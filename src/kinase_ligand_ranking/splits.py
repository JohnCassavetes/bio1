"""Benchmark split generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from kinase_ligand_ranking.sequence_identity import (
    SequenceIdentityConfig,
    cluster_targets_by_identity,
    compute_identity_table,
)


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: Dict[str, object]
    artifacts: Optional[Dict[str, pd.DataFrame]] = None


def generate_split_bundle(
    df: pd.DataFrame,
    *,
    split_type: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_seed: int = 42,
    sequence_identity_threshold: float = 0.6,
    sequence_kmer_size: int = 3,
) -> SplitBundle:
    """Generate a train/val/test split bundle for a named strategy."""
    split_type = split_type.lower()
    extra_manifest: Dict[str, object] = {}
    artifacts: Dict[str, pd.DataFrame] = {}
    if split_type == "random":
        assignment = _random_row_assignment(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
        )
        grouping = "row"
    elif split_type == "cold_target":
        assignment = _group_assignment(
            df,
            group_column="target_id",
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
        )
        grouping = "target_id"
    elif split_type == "cold_ligand":
        assignment = _group_assignment(
            df,
            group_column="smiles",
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
        )
        grouping = "smiles"
    elif split_type == "scaffold":
        scaffold_df = df.copy()
        scaffold_df["scaffold_id"] = scaffold_df["smiles"].apply(smiles_to_scaffold)
        assignment = _group_assignment(
            scaffold_df,
            group_column="scaffold_id",
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
        )
        grouping = "scaffold_id"
        df = scaffold_df
    elif split_type == "both_new":
        assignment = _both_new_assignment(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
        )
        grouping = "target_id+smiles"
    elif split_type == "mutation_holdout":
        assignment = _mutation_holdout_assignment(
            df,
            random_seed=random_seed,
        )
        grouping = "mutation_family"
    elif split_type == "sequence_identity":
        assignment, extra_manifest, artifacts = _sequence_identity_assignment(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed,
            identity_threshold=sequence_identity_threshold,
            kmer_size=sequence_kmer_size,
        )
        grouping = "sequence_identity_cluster"
    else:
        raise ValueError(f"Unsupported split type: {split_type}")

    retained_mask = assignment.isin(["train", "val", "test"])
    train_df = df[assignment == "train"].copy()
    val_df = df[assignment == "val"].copy()
    test_df = df[assignment == "test"].copy()

    manifest = {
        "split_type": split_type,
        "grouping": grouping,
        "random_seed": random_seed,
        "retained_rows": int(retained_mask.sum()),
        "discarded_rows": int((~retained_mask).sum()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_targets": int(train_df["target_id"].nunique()),
        "val_targets": int(val_df["target_id"].nunique()),
        "test_targets": int(test_df["target_id"].nunique()),
        "train_ligands": int(train_df["smiles"].nunique()),
        "val_ligands": int(val_df["smiles"].nunique()),
        "test_ligands": int(test_df["smiles"].nunique()),
    }
    manifest.update(extra_manifest)
    return SplitBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        manifest=manifest,
        artifacts=artifacts or None,
    )


def smiles_to_scaffold(smiles: str) -> str:
    """Convert a SMILES string to a Bemis-Murcko scaffold identifier."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return smiles
    return Chem.MolToSmiles(scaffold)


def target_family(target_id: str, known_targets: Iterable[str] | None = None) -> str:
    """Normalize a target ID to its base family/wild-type name."""
    target_id = str(target_id)
    family = target_id.split("(")[0]
    if known_targets is not None and family.endswith("p") and family[:-1] in set(known_targets):
        family = family[:-1]
    return family


def _random_row_assignment(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    random_seed: int,
) -> pd.Series:
    rng = np.random.RandomState(random_seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    train_cut = int(len(indices) * train_frac)
    val_cut = train_cut + int(len(indices) * val_frac)
    assignment = pd.Series(index=df.index, dtype="object")
    assignment.iloc[indices[:train_cut]] = "train"
    assignment.iloc[indices[train_cut:val_cut]] = "val"
    assignment.iloc[indices[val_cut:]] = "test"
    return assignment


def _group_assignment(
    df: pd.DataFrame,
    *,
    group_column: str,
    train_frac: float,
    val_frac: float,
    random_seed: int,
) -> pd.Series:
    rng = np.random.RandomState(random_seed)
    group_counts = df.groupby(group_column).size().sort_values(ascending=False)
    groups = list(group_counts.index)
    rng.shuffle(groups)

    total_rows = len(df)
    train_budget = int(total_rows * train_frac)
    val_budget = int(total_rows * val_frac)
    assignment_map: Dict[str, str] = {}
    train_rows = 0
    val_rows = 0

    for group in groups:
        count = int(group_counts[group])
        if train_rows < train_budget:
            split = "train"
            train_rows += count
        elif val_rows < val_budget:
            split = "val"
            val_rows += count
        else:
            split = "test"
        assignment_map[str(group)] = split

    return df[group_column].astype(str).map(assignment_map)


def _both_new_assignment(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    random_seed: int,
) -> pd.Series:
    target_assignment = _group_assignment(
        df,
        group_column="target_id",
        train_frac=train_frac,
        val_frac=val_frac,
        random_seed=random_seed,
    )
    ligand_assignment = _group_assignment(
        df,
        group_column="smiles",
        train_frac=train_frac,
        val_frac=val_frac,
        random_seed=random_seed + 17,
    )

    assignment = pd.Series(index=df.index, dtype="object")
    for split in ["train", "val", "test"]:
        mask = (target_assignment == split) & (ligand_assignment == split)
        assignment.loc[mask] = split

    # Rows falling into mixed quadrants are excluded so the test split remains
    # strictly both-new.
    assignment = assignment.fillna("discard")
    return assignment


def _mutation_holdout_assignment(
    df: pd.DataFrame,
    *,
    random_seed: int,
) -> pd.Series:
    known_targets = set(df["target_id"].astype(str).unique())
    mutation_df = df.copy()
    mutation_df["target_family"] = mutation_df["target_id"].apply(
        lambda value: target_family(value, known_targets=known_targets)
    )
    mutation_df["is_variant"] = mutation_df["target_id"] != mutation_df["target_family"]

    family_stats = mutation_df.groupby("target_family").agg(
        has_wildtype=("is_variant", lambda values: bool((~values).any())),
        has_variant=("is_variant", lambda values: bool(values.any())),
    )
    candidate_families = set(
        family_stats[(family_stats["has_wildtype"]) & (family_stats["has_variant"])].index
    )

    assignment = pd.Series("train", index=df.index, dtype="object")
    candidate_variant_rows = mutation_df[
        mutation_df["is_variant"] & mutation_df["target_family"].isin(candidate_families)
    ]
    if candidate_variant_rows.empty:
        return assignment

    variant_assignment = _group_assignment(
        candidate_variant_rows,
        group_column="target_family",
        train_frac=0.0,
        val_frac=0.5,
        random_seed=random_seed,
    )
    assignment.loc[candidate_variant_rows.index] = variant_assignment.values
    return assignment


def _sequence_identity_assignment(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    random_seed: int,
    identity_threshold: float,
    kmer_size: int,
) -> Tuple[pd.Series, Dict[str, object], Dict[str, pd.DataFrame]]:
    target_rows = (
        df[["target_id", "target_sequence"]]
        .drop_duplicates(subset=["target_id"])
        .assign(target_sequence=lambda frame: frame["target_sequence"].fillna("").astype(str))
    )

    identity_config = SequenceIdentityConfig(threshold=identity_threshold)
    identity_table = compute_identity_table(target_rows, config=identity_config)
    cluster_table = cluster_targets_by_identity(identity_table, threshold=identity_threshold)
    cluster_map = dict(
        zip(
            cluster_table["target_id"].astype(str),
            cluster_table["sequence_identity_cluster"].astype(str),
        )
    )

    cluster_df = df.copy()
    cluster_df["sequence_identity_cluster"] = cluster_df["target_id"].map(cluster_map)
    assignment = _group_assignment(
        cluster_df,
        group_column="sequence_identity_cluster",
        train_frac=train_frac,
        val_frac=val_frac,
        random_seed=random_seed,
    )
    manifest = {
        "sequence_identity_threshold": float(identity_threshold),
        "sequence_identity_metric": "global_alignment_percent_identity",
        "sequence_identity_gap_open": float(identity_config.gap_open),
        "sequence_identity_gap_extend": float(identity_config.gap_extend),
        "sequence_identity_cluster_count": int(cluster_table["sequence_identity_cluster"].nunique()),
        "largest_sequence_identity_cluster": int(
            cluster_table.groupby("sequence_identity_cluster").size().max()
        ),
    }
    return assignment, manifest, {
        "target_sequence_identity": identity_table,
        "target_sequence_identity_clusters": cluster_table,
    }
