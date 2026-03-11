#!/usr/bin/env python3
"""Process raw kinase binding data into ML-ready format.

Loads data from BindingDB JSON files or Davis CSV, then:
  1. Filters for valid measurements (IC50, Ki, Kd)
  2. Validates and canonicalizes SMILES with RDKit
  3. Filters affinity range (0.01 nM - 1 mM)
  4. Deduplicates via geometric mean aggregation
  5. Converts to generic pActivity scale (-log10 affinity in molar units)
  6. Splits by protein target (70/15/15 train/val/test)

Usage:
    python scripts/process_dataset.py
    python scripts/process_dataset.py --source davis
    python scripts/process_dataset.py --source bindingdb
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

AFFINITY_TYPES = ["IC50", "KI", "KD"]
MIN_AFFINITY_NM = 0.01
MAX_AFFINITY_NM = 1_000_000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42


def load_bindingdb_json_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all BindingDB JSON response files into a single DataFrame."""
    json_files = sorted(raw_dir.glob("bindingdb_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No bindingdb_*.json files found in {raw_dir}")

    metadata_path = raw_dir / "bindingdb_target_metadata.json"
    target_metadata: Dict[str, Dict[str, Any]] = {}
    if metadata_path.exists():
        with open(metadata_path) as fh:
            target_metadata = json.load(fh)

    all_records: List[Dict[str, Any]] = []
    for json_file in json_files:
        target_id = json_file.stem.replace("bindingdb_", "")
        with open(json_file) as fh:
            data = json.load(fh)

        records = _parse_bindingdb_response(data, target_id, target_metadata.get(target_id, {}))
        all_records.extend(records)
        logger.info("Loaded %d records from %s", len(records), json_file.name)

    df = pd.DataFrame(all_records)
    logger.info("Total raw BindingDB records loaded: %d", len(df))
    return df


def _parse_bindingdb_response(
    data: Any,
    target_id: str,
    target_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse a single BindingDB JSON response into flat records."""
    records: List[Dict[str, Any]] = []
    target_metadata = target_metadata or {}

    if isinstance(data, dict):
        entries = (
            data.get("affinities")
            or data.get("data")
            or data.get("bindingdb")
            or data.get("getLigandsByUniprotsResponse", {}).get("affinities")
            or data.get("getLindsByUniprotsResponse", {}).get("affinities")
            or [data]
        )
    elif isinstance(data, list):
        entries = data
    else:
        logger.warning("Unexpected BindingDB response type: %s", type(data))
        return records

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        smiles = (
            entry.get("smiles")
            or entry.get("smile")
            or entry.get("ligand_smiles")
            or entry.get("Ligand SMILES")
        )
        affinity_type = (
            entry.get("affinity_type")
            or entry.get("affinityType")
            or entry.get("Affinity Type")
        )
        affinity_value = (
            entry.get("affinity")
            or entry.get("affinityValue")
            or entry.get("Affinity Value")
        )
        target_label = (
            entry.get("target_name")
            or entry.get("query")
            or entry.get("Target Name")
            or entry.get("target")
            or target_metadata.get("target_name")
            or target_id
        )

        if not smiles or not affinity_type or affinity_value is None:
            continue

        try:
            records.append(
                {
                    "smiles": str(smiles),
                    "target_id": target_id,
                    "target_label": str(target_label),
                    "target_sequence": target_metadata.get("target_sequence"),
                    "affinity_type": str(affinity_type).upper().strip(),
                    "affinity_nm": float(affinity_value),
                    "source": "bindingdb",
                }
            )
        except (TypeError, ValueError):
            continue

    return records


def load_davis_csv(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the Davis dataset CSV into the normalized schema."""
    filepath = raw_dir / "davis_dataset.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Davis dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    result = pd.DataFrame(
        {
            "smiles": df["Drug"],
            "target_id": df["Target_ID"],
            "target_label": df["Target_ID"],
            "target_sequence": df.get("Target_Sequence", df.get("Target")),
            "affinity_type": "KD",
            "affinity_nm": df["Y"],
            "source": "davis",
        }
    )
    logger.info("Loaded Davis dataset: %d rows", len(result))
    return result


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert a SMILES string to RDKit canonical form."""
    from rdkit import Chem

    if not smiles or not isinstance(smiles, str):
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def affinity_nm_to_pactivity(affinity_nm: float) -> float:
    """Convert affinity in nM to pActivity (-log10 of molar concentration)."""
    return -np.log10(affinity_nm * 1e-9)


def filter_and_clean(
    df: pd.DataFrame,
    min_nm: float = MIN_AFFINITY_NM,
    max_nm: float = MAX_AFFINITY_NM,
    allowed_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter and clean binding affinity data."""
    if allowed_types is None:
        allowed_types = AFFINITY_TYPES

    stats: Dict[str, int] = {"initial_count": len(df)}

    df = df.dropna(subset=["smiles", "target_id", "affinity_nm", "affinity_type"])
    stats["after_drop_na"] = len(df)

    df = df.copy()
    df["affinity_type"] = df["affinity_type"].astype(str).str.upper().str.strip()
    df = df[df["affinity_type"].isin(allowed_types)]
    stats["after_type_filter"] = len(df)

    df = df[(df["affinity_nm"] >= min_nm) & (df["affinity_nm"] <= max_nm)]
    stats["after_range_filter"] = len(df)

    logger.info("Validating SMILES with RDKit...")
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    invalid_mask = df["canonical_smiles"].isna()
    stats["invalid_smiles_count"] = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()
    stats["after_smiles_validation"] = len(df)

    df["smiles"] = df["canonical_smiles"]
    df = df.drop(columns=["canonical_smiles"])
    df["target_sequence"] = df["target_sequence"].fillna("").astype(str)
    df["target_label"] = df["target_label"].fillna(df["target_id"]).astype(str)

    logger.info("Filtering stats: %s", stats)
    return df, stats


def deduplicate_and_aggregate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deduplicate by target, ligand, and assay type using geometric mean."""
    stats: Dict[str, int] = {"before_dedup": len(df)}

    group_cols = ["smiles", "target_id", "affinity_type"]
    duplicates = df.groupby(group_cols).size()
    stats["duplicate_groups"] = int((duplicates > 1).sum())

    def geometric_mean(series: pd.Series) -> float:
        return float(np.exp(np.mean(np.log(series.values))))

    def first_non_empty(series: pd.Series) -> str:
        for value in series.astype(str):
            if value:
                return value
        return ""

    df_agg = df.groupby(group_cols, as_index=False).agg(
        affinity_nm=("affinity_nm", geometric_mean),
        measurement_count=("affinity_nm", "size"),
        source=("source", "first"),
        target_label=("target_label", "first"),
        target_sequence=("target_sequence", first_non_empty),
    )

    stats["after_dedup"] = len(df_agg)
    logger.info(
        "Deduplication: %d -> %d (%d groups had duplicates)",
        stats["before_dedup"],
        stats["after_dedup"],
        stats["duplicate_groups"],
    )
    return df_agg, stats


def compute_pactivity(df: pd.DataFrame) -> pd.DataFrame:
    """Add pActivity columns to the DataFrame."""
    df = df.copy()
    df["p_activity"] = df["affinity_nm"].apply(affinity_nm_to_pactivity)
    df["activity_label"] = df["affinity_type"].map(
        {
            "IC50": "pIC50",
            "KI": "pKi",
            "KD": "pKd",
        }
    )
    return df


def split_by_target(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Split dataset into train/val/test by target identifier."""
    rng = np.random.RandomState(random_seed)
    targets = df["target_id"].unique()
    target_counts = df.groupby("target_id").size().sort_values(ascending=False)

    shuffled_targets = rng.permutation(targets)
    total_rows = len(df)
    train_budget = int(total_rows * train_frac)
    val_budget = int(total_rows * val_frac)

    train_targets: List[str] = []
    val_targets: List[str] = []
    test_targets: List[str] = []
    train_count = 0
    val_count = 0

    for target_id in shuffled_targets:
        count = int(target_counts[target_id])
        if train_count < train_budget:
            train_targets.append(str(target_id))
            train_count += count
        elif val_count < val_budget:
            val_targets.append(str(target_id))
            val_count += count
        else:
            test_targets.append(str(target_id))

    if not val_targets and len(train_targets) > 1:
        val_targets.append(train_targets.pop())
    if not test_targets and len(train_targets) > 1:
        test_targets.append(train_targets.pop())

    assignments = {
        "train": train_targets,
        "val": val_targets,
        "test": test_targets,
    }

    train_df = df[df["target_id"].isin(train_targets)].copy()
    val_df = df[df["target_id"].isin(val_targets)].copy()
    test_df = df[df["target_id"].isin(test_targets)].copy()

    logger.info(
        "Split: train=%d (%d targets), val=%d (%d targets), test=%d (%d targets)",
        len(train_df),
        len(train_targets),
        len(val_df),
        len(val_targets),
        len(test_df),
        len(test_targets),
    )
    return train_df, val_df, test_df, assignments


def save_processed_data(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_assignments: Dict[str, List[str]],
    filter_stats: Dict[str, int],
    dedup_stats: Dict[str, int],
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save processed data, metadata, and target summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_columns = [
        "smiles",
        "target_id",
        "target_label",
        "target_sequence",
        "affinity_type",
        "activity_label",
        "affinity_nm",
        "p_activity",
        "measurement_count",
        "source",
    ]
    full_df = full_df[ordered_columns]
    train_df = train_df[ordered_columns]
    val_df = val_df[ordered_columns]
    test_df = test_df[ordered_columns]

    full_df.to_csv(output_dir / "kinase_binding_data.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    target_summary = (
        full_df.groupby("target_id", as_index=False)
        .agg(
            target_label=("target_label", "first"),
            num_measurements=("target_id", "size"),
            num_unique_ligands=("smiles", "nunique"),
            has_sequence=("target_sequence", lambda x: bool(x.astype(str).str.len().max())),
        )
        .sort_values(["num_measurements", "target_id"], ascending=[False, True])
    )
    target_summary.to_csv(output_dir / "target_summary.csv", index=False)

    stats = {
        "filter_stats": filter_stats,
        "dedup_stats": dedup_stats,
        "affinity_type_counts": {
            key: int(value)
            for key, value in full_df["affinity_type"].value_counts().sort_index().items()
        },
        "split_stats": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_targets": len(split_assignments["train"]),
            "val_targets": len(split_assignments["val"]),
            "test_targets": len(split_assignments["test"]),
        },
    }
    with open(output_dir / "processing_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    with open(output_dir / "target_split_assignments.json", "w") as fh:
        json.dump(split_assignments, fh, indent=2)

    logger.info("All processed data saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw kinase binding data into ML-ready format"
    )
    parser.add_argument(
        "--source",
        choices=["bindingdb", "davis", "auto"],
        default="auto",
        help="Data source (default: auto-detect)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Raw data directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if args.source == "auto":
        json_files = list(args.raw_dir.glob("bindingdb_*.json"))
        davis_file = args.raw_dir / "davis_dataset.csv"
        if davis_file.exists():
            logger.info("Auto-detected Davis CSV")
            df = load_davis_csv(args.raw_dir)
        elif json_files:
            logger.info("Auto-detected BindingDB JSON files")
            df = load_bindingdb_json_files(args.raw_dir)
        else:
            raise FileNotFoundError(
                f"No raw data found in {args.raw_dir}. Run download_data.py first."
            )
    elif args.source == "bindingdb":
        df = load_bindingdb_json_files(args.raw_dir)
    else:
        df = load_davis_csv(args.raw_dir)

    df, filter_stats = filter_and_clean(df)
    df, dedup_stats = deduplicate_and_aggregate(df)
    df = compute_pactivity(df)
    train_df, val_df, test_df, assignments = split_by_target(df)
    save_processed_data(
        df,
        train_df,
        val_df,
        test_df,
        assignments,
        filter_stats,
        dedup_stats,
        output_dir=args.output_dir,
    )

    print("\n=== Processing Summary ===")
    print(f"Total records:   {len(df)}")
    print(f"Unique ligands:  {df['smiles'].nunique()}")
    print(f"Unique targets:  {df['target_id'].nunique()}")
    print(
        f"pActivity range: {df['p_activity'].min():.2f} - {df['p_activity'].max():.2f}"
    )
    print(
        f"Affinity types:  {', '.join(sorted(df['affinity_type'].unique().tolist()))}"
    )
    print(f"Train: {len(train_df)} rows, {len(assignments['train'])} targets")
    print(f"Val:   {len(val_df)} rows, {len(assignments['val'])} targets")
    print(f"Test:  {len(test_df)} rows, {len(assignments['test'])} targets")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
