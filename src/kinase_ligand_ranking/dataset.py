"""Dataset utilities for processed kinase-ligand CSV files."""

from pathlib import Path
from typing import Dict

import pandas as pd

REQUIRED_COLUMNS = [
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


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a processed dataset CSV and validate the expected schema."""
    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    df = df.copy()
    df["target_sequence"] = df["target_sequence"].fillna("").astype(str)
    df["target_id"] = df["target_id"].astype(str)
    df["affinity_type"] = df["affinity_type"].astype(str)
    return df


def dataset_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Return a compact summary dictionary for reporting."""
    return {
        "rows": int(len(df)),
        "targets": int(df["target_id"].nunique()),
        "ligands": int(df["smiles"].nunique()),
        "min_p_activity": float(df["p_activity"].min()),
        "max_p_activity": float(df["p_activity"].max()),
    }
