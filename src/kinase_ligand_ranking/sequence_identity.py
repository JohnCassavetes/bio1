"""Exact sequence-identity utilities based on global pairwise alignment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from Bio import pairwise2
except ImportError:  # pragma: no cover - environment-dependent.
    pairwise2 = None


@dataclass
class SequenceIdentityConfig:
    threshold: float = 0.6
    gap_open: float = -2.0
    gap_extend: float = -0.5
    workers: int = max(1, (os.cpu_count() or 2) - 1)


def require_biopython() -> None:
    if pairwise2 is None:
        raise ImportError(
            "Biopython is required for exact sequence-identity splits. "
            "Install the pinned environment or `biopython`."
        )


def compute_identity_table(
    target_sequences: pd.DataFrame,
    *,
    config: SequenceIdentityConfig,
) -> pd.DataFrame:
    """Compute all-vs-all global-alignment identity for unique targets."""
    require_biopython()
    unique_targets = (
        target_sequences[["target_id", "target_sequence"]]
        .drop_duplicates("target_id")
        .assign(target_sequence=lambda frame: frame["target_sequence"].fillna("").astype(str))
        .sort_values("target_id")
    )
    sequence_map = {
        str(row.target_id): str(row.target_sequence)
        for row in unique_targets.itertuples(index=False)
    }

    target_ids = list(sequence_map.keys())
    pairs = [
        (
            left_id,
            right_id,
            sequence_map[left_id],
            sequence_map[right_id],
            config.gap_open,
            config.gap_extend,
        )
        for left_index, left_id in enumerate(target_ids)
        for right_id in target_ids[left_index:]
    ]
    if config.workers <= 1:
        pair_rows = [_identity_job(pair) for pair in pairs]
    else:
        with Pool(processes=config.workers) as pool:
            pair_rows = pool.map(_identity_job, pairs, chunksize=32)

    rows: List[Dict[str, object]] = []
    for left_id, right_id, identity in pair_rows:
        rows.append({"target_id_1": left_id, "target_id_2": right_id, "sequence_identity": identity})
        if left_id != right_id:
            rows.append({"target_id_1": right_id, "target_id_2": left_id, "sequence_identity": identity})
    return pd.DataFrame(rows).sort_values(["target_id_1", "target_id_2"]).reset_index(drop=True)


def pairwise_sequence_identity(
    sequence_a: str,
    sequence_b: str,
    *,
    gap_open: float,
    gap_extend: float,
) -> float:
    require_biopython()
    normalized_a = (sequence_a or "").upper()
    normalized_b = (sequence_b or "").upper()
    if not normalized_a and not normalized_b:
        return 1.0
    if not normalized_a or not normalized_b:
        return 0.0

    aligned_a, aligned_b, _, _, _ = _cached_global_alignment(
        normalized_a,
        normalized_b,
        gap_open,
        gap_extend,
    )
    alignment_length = len(aligned_a)
    if alignment_length == 0:
        return 0.0
    matches = sum(char_a == char_b for char_a, char_b in zip(aligned_a, aligned_b))
    return matches / alignment_length


@lru_cache(maxsize=20000)
def _cached_global_alignment(
    sequence_a: str,
    sequence_b: str,
    gap_open: float,
    gap_extend: float,
) -> Tuple[str, str, float, int, int]:
    alignments = pairwise2.align.globalms(
        sequence_a,
        sequence_b,
        1.0,
        0.0,
        gap_open,
        gap_extend,
        one_alignment_only=True,
    )
    if not alignments:
        return "", "", 0.0, 0, 0
    return alignments[0]


def _identity_job(job: Tuple[str, str, str, str, float, float]) -> Tuple[str, str, float]:
    left_id, right_id, sequence_a, sequence_b, gap_open, gap_extend = job
    identity = pairwise_sequence_identity(
        sequence_a,
        sequence_b,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return left_id, right_id, identity


def cluster_targets_by_identity(
    identity_table: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Cluster targets by connected components at a fixed identity threshold."""
    target_ids = sorted(
        set(identity_table["target_id_1"].astype(str)).union(set(identity_table["target_id_2"].astype(str)))
    )
    parent = {target_id: target_id for target_id in target_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    qualifying_edges = identity_table[identity_table["sequence_identity"] >= threshold]
    for row in qualifying_edges.itertuples(index=False):
        union(str(row.target_id_1), str(row.target_id_2))

    roots = {target_id: find(target_id) for target_id in target_ids}
    cluster_members: Dict[str, List[str]] = {}
    for target_id, root in roots.items():
        cluster_members.setdefault(root, []).append(target_id)

    cluster_labels = {
        target_id: f"cluster_{cluster_index:03d}"
        for cluster_index, members in enumerate(
            sorted(cluster_members.values(), key=lambda items: (-len(items), items[0]))
        )
        for target_id in sorted(members)
    }
    rows = []
    for target_id in target_ids:
        rows.append(
            {
                "target_id": target_id,
                "sequence_identity_cluster": cluster_labels[target_id],
            }
        )
    return pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)


def nearest_train_identity(
    identity_table: pd.DataFrame,
    *,
    train_targets: Iterable[str],
    eval_targets: Iterable[str],
) -> pd.DataFrame:
    train_set = {str(target) for target in train_targets}
    eval_set = {str(target) for target in eval_targets}
    rows: List[Dict[str, object]] = []
    for eval_target in sorted(eval_set):
        subset = identity_table[
            (identity_table["target_id_1"].astype(str) == eval_target)
            & (identity_table["target_id_2"].astype(str).isin(train_set))
        ]
        if subset.empty:
            rows.append(
                {
                    "target_id": eval_target,
                    "nearest_train_target_id": None,
                    "nearest_train_sequence_identity": 0.0,
                }
            )
            continue
        best_row = subset.sort_values(
            ["sequence_identity", "target_id_2"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "target_id": eval_target,
                "nearest_train_target_id": str(best_row["target_id_2"]),
                "nearest_train_sequence_identity": float(best_row["sequence_identity"]),
            }
        )
    return pd.DataFrame(rows)


def save_identity_artifacts(
    *,
    identity_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_table.to_csv(output_dir / "target_sequence_identity.csv", index=False)
    cluster_table.to_csv(output_dir / "target_sequence_identity_clusters.csv", index=False)
