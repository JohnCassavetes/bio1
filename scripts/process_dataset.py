#!/usr/bin/env python3
"""Process raw kinase binding data into ML-ready format.

Loads data from BindingDB JSON files or Davis CSV, then:
  1. Filters for valid measurements (IC50, Ki, Kd)
  2. Validates and canonicalizes SMILES with RDKit
  3. Filters affinity range (0.01 nM - 1 mM)
  4. Deduplicates via geometric mean aggregation
  5. Converts to pIC50 scale
  6. Splits by protein target (70/15/15 train/val/test)

Usage:
    python scripts/process_dataset.py                   # auto-detect source
    python scripts/process_dataset.py --source davis     # explicit Davis
    python scripts/process_dataset.py --source bindingdb # explicit BindingDB
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration ---

BASE_DIR = Path(__file__).resolve().parent.parent  # bio1/
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

AFFINITY_TYPES = ["IC50", "KI", "KD"]
MIN_AFFINITY_NM = 0.01
MAX_AFFINITY_NM = 1_000_000  # 1 mM
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42


# --- Loaders ---


def load_bindingdb_json_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all BindingDB JSON response files into a single DataFrame."""
    json_files = sorted(raw_dir.glob("bindingdb_*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No bindingdb_*.json files found in {raw_dir}"
        )

    all_records: List[Dict[str, Any]] = []
    for jf in json_files:
        uniprot_id = jf.stem.replace("bindingdb_", "")
        with open(jf) as fh:
            data = json.load(fh)

        records = _parse_bindingdb_response(data, uniprot_id)
        all_records.extend(records)
        logger.info("Loaded %d records from %s", len(records), jf.name)

    df = pd.DataFrame(all_records)
    logger.info("Total raw records loaded: %d", len(df))
    return df


def _parse_bindingdb_response(
    data: Any,
    uniprot_id: str,
) -> List[Dict[str, Any]]:
    """Parse a single BindingDB JSON response into flat records.

    Handles multiple possible response formats from the API.
    """
    records: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        entries = data.get("affinities", data.get("data", [data]))
    elif isinstance(data, list):
        entries = data
    else:
        logger.warning(
            "Unexpected response type for %s: %s", uniprot_id, type(data)
        )
        return records

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        # Try common field names from BindingDB API
        smiles = (
            entry.get("smiles")
            or entry.get("ligand_smiles")
            or entry.get("Ligand SMILES")
        )
        aff_type = entry.get("affinity_type") or entry.get("affinityType")
        aff_value = entry.get("affinity") or entry.get("affinityValue")

        if smiles and aff_type and aff_value:
            try:
                records.append(
                    {
                        "smiles": str(smiles),
                        "uniprot_id": uniprot_id,
                        "affinity_type": str(aff_type).upper().strip(),
                        "affinity_nm": float(aff_value),
                        "source": "bindingdb",
                    }
                )
            except (ValueError, TypeError):
                continue

    return records


def load_davis_csv(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the Davis dataset CSV (from TDC fallback).

    TDC Davis format has columns: Drug_ID, Drug, Target_ID, Target, Y
    where Y is Kd in nM.
    """
    filepath = raw_dir / "davis_dataset.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Davis dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    result = pd.DataFrame(
        {
            "smiles": df["Drug"],
            "uniprot_id": df["Target_ID"],
            "affinity_type": "KD",
            "affinity_nm": df["Y"],
            "source": "davis",
        }
    )
    logger.info("Loaded Davis dataset: %d rows", len(result))
    return result


# --- Processing ---


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert a SMILES string to RDKit canonical form. Returns None if invalid."""
    from rdkit import Chem

    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def affinity_to_pic50(affinity_nm: float) -> float:
    """Convert affinity in nanomolar to pIC50 (-log10 of molar concentration).

    Higher pIC50 = more potent compound.
    Example: 10 nM -> pIC50 = 8.0, 1000 nM -> pIC50 = 6.0
    """
    return -np.log10(affinity_nm * 1e-9)


def filter_and_clean(
    df: pd.DataFrame,
    min_nm: float = MIN_AFFINITY_NM,
    max_nm: float = MAX_AFFINITY_NM,
    allowed_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter and clean binding affinity data.

    Steps: drop NaN -> filter affinity types -> filter range -> validate SMILES.
    Returns cleaned DataFrame and filtering statistics.
    """
    if allowed_types is None:
        allowed_types = AFFINITY_TYPES

    stats: Dict[str, int] = {"initial_count": len(df)}

    # Drop missing values
    df = df.dropna(subset=["smiles", "affinity_nm"])
    stats["after_drop_na"] = len(df)

    # Filter affinity types
    df["affinity_type"] = df["affinity_type"].str.upper().str.strip()
    df = df[df["affinity_type"].isin(allowed_types)]
    stats["after_type_filter"] = len(df)

    # Filter affinity range
    df = df[(df["affinity_nm"] >= min_nm) & (df["affinity_nm"] <= max_nm)]
    stats["after_range_filter"] = len(df)

    # Validate and canonicalize SMILES
    logger.info("Validating SMILES with RDKit...")
    df = df.copy()
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    invalid_mask = df["canonical_smiles"].isna()
    stats["invalid_smiles_count"] = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()
    stats["after_smiles_validation"] = len(df)

    df["smiles"] = df["canonical_smiles"]
    df = df.drop(columns=["canonical_smiles"])

    logger.info("Filtering stats: %s", stats)
    return df, stats


def deduplicate_and_aggregate(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Deduplicate by (smiles, uniprot_id, affinity_type) using geometric mean.

    Geometric mean is appropriate because affinity measurements are
    log-normally distributed.
    """
    stats: Dict[str, int] = {"before_dedup": len(df)}

    group_cols = ["smiles", "uniprot_id", "affinity_type"]
    duplicates = df.groupby(group_cols).size()
    stats["duplicate_groups"] = int((duplicates > 1).sum())

    def geometric_mean(series: pd.Series) -> float:
        return float(np.exp(np.mean(np.log(series.values))))

    df_agg = df.groupby(group_cols, as_index=False).agg(
        affinity_nm=("affinity_nm", geometric_mean),
        measurement_count=("affinity_nm", "size"),
        source=("source", "first"),
    )

    stats["after_dedup"] = len(df_agg)
    logger.info(
        "Deduplication: %d -> %d (%d groups had duplicates)",
        stats["before_dedup"],
        stats["after_dedup"],
        stats["duplicate_groups"],
    )
    return df_agg, stats


def compute_pic50(df: pd.DataFrame) -> pd.DataFrame:
    """Add pIC50 column to the DataFrame."""
    df = df.copy()
    df["pic50"] = df["affinity_nm"].apply(affinity_to_pic50)
    return df


def split_by_target(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """Split dataset into train/val/test by protein target.

    All measurements for a given target go to exactly one split.
    This prevents data leakage and tests generalization to unseen targets.
    Uses greedy assignment to approximate desired fractions.
    """
    rng = np.random.RandomState(random_seed)
    targets = df["uniprot_id"].unique()
    target_counts = df.groupby("uniprot_id").size().sort_values(ascending=False)

    shuffled = rng.permutation(targets)
    total = len(df)
    train_budget = int(total * train_frac)
    val_budget = int(total * val_frac)

    train_targets: List[str] = []
    val_targets: List[str] = []
    test_targets: List[str] = []
    train_count = val_count = 0

    for t in shuffled:
        count = target_counts[t]
        if train_count < train_budget:
            train_targets.append(t)
            train_count += count
        elif val_count < val_budget:
            val_targets.append(t)
            val_count += count
        else:
            test_targets.append(t)

    # Ensure no split is empty
    if not val_targets and len(train_targets) > 1:
        val_targets.append(train_targets.pop())
    if not test_targets and len(train_targets) > 1:
        test_targets.append(train_targets.pop())

    assignments = {
        "train": [str(t) for t in train_targets],
        "val": [str(t) for t in val_targets],
        "test": [str(t) for t in test_targets],
    }

    train_df = df[df["uniprot_id"].isin(train_targets)]
    val_df = df[df["uniprot_id"].isin(val_targets)]
    test_df = df[df["uniprot_id"].isin(test_targets)]

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
    """Save all processed data and metadata to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(output_dir / "kinase_binding_data.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    stats = {
        "filter_stats": filter_stats,
        "dedup_stats": dedup_stats,
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


# --- CLI ---


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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    # 1. Load raw data
    if args.source == "auto":
        json_files = list(args.raw_dir.glob("bindingdb_*.json"))
        davis_file = args.raw_dir / "davis_dataset.csv"
        if json_files:
            logger.info("Auto-detected BindingDB JSON files")
            df = load_bindingdb_json_files(args.raw_dir)
        elif davis_file.exists():
            logger.info("Auto-detected Davis CSV")
            df = load_davis_csv(args.raw_dir)
        else:
            raise FileNotFoundError(
                f"No raw data found in {args.raw_dir}. "
                "Run download_data.py first."
            )
    elif args.source == "bindingdb":
        df = load_bindingdb_json_files(args.raw_dir)
    else:
        df = load_davis_csv(args.raw_dir)

    # 2. Filter and clean
    df, filter_stats = filter_and_clean(df)

    # 3. Deduplicate and aggregate
    df, dedup_stats = deduplicate_and_aggregate(df)

    # 4. Compute pIC50
    df = compute_pic50(df)

    # 5. Split by target
    train_df, val_df, test_df, assignments = split_by_target(df)

    # 6. Save
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

    # Summary
    print("\n=== Processing Summary ===")
    print(f"Total records:   {len(df)}")
    print(f"Unique ligands:  {df['smiles'].nunique()}")
    print(f"Unique targets:  {df['uniprot_id'].nunique()}")
    print(f"pIC50 range:     {df['pic50'].min():.2f} - {df['pic50'].max():.2f}")
    print(f"Train: {len(train_df)} rows, {len(assignments['train'])} targets")
    print(f"Val:   {len(val_df)} rows, {len(assignments['val'])} targets")
    print(f"Test:  {len(test_df)} rows, {len(assignments['test'])} targets")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
