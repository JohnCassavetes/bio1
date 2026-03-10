"""Feature builders for kinase-ligand ranking baselines."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
AFFINITY_TYPES = ["IC50", "KD", "KI"]
SOURCES = ["bindingdb", "davis"]


def _build_fingerprint_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def smiles_to_fingerprint(
    smiles: str,
    *,
    generator,
    n_bits: int,
) -> np.ndarray:
    """Convert a SMILES string into a Morgan fingerprint bit vector."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fingerprint = generator.GetFingerprint(molecule)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def sequence_to_features(sequence: str) -> np.ndarray:
    """Encode a protein sequence using amino-acid composition and length."""
    sequence = (sequence or "").upper()
    features = np.zeros((len(AMINO_ACIDS) + 2,), dtype=np.float32)
    if not sequence:
        return features

    valid_residue_count = 0
    for residue in sequence:
        index = AMINO_ACID_INDEX.get(residue)
        if index is None:
            continue
        features[index] += 1.0
        valid_residue_count += 1

    if valid_residue_count:
        features[: len(AMINO_ACIDS)] /= valid_residue_count

    sequence_length = len(sequence)
    features[-2] = np.log1p(sequence_length)
    features[-1] = valid_residue_count / max(sequence_length, 1)
    return features


def categorical_one_hot(value: str, categories: Sequence[str]) -> np.ndarray:
    """One-hot encode a categorical value against a fixed vocabulary."""
    encoded = np.zeros((len(categories),), dtype=np.float32)
    try:
        encoded[categories.index(value)] = 1.0
    except ValueError:
        pass
    return encoded


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    fingerprint_bits: int = 1024,
    fingerprint_radius: int = 2,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a dense design matrix from processed kinase-ligand rows."""
    generator = _build_fingerprint_generator(
        radius=fingerprint_radius,
        n_bits=fingerprint_bits,
    )

    fingerprint_rows: List[np.ndarray] = []
    sequence_rows: List[np.ndarray] = []
    assay_rows: List[np.ndarray] = []
    source_rows: List[np.ndarray] = []

    for row in df.itertuples(index=False):
        fingerprint_rows.append(
            smiles_to_fingerprint(
                row.smiles,
                generator=generator,
                n_bits=fingerprint_bits,
            )
        )
        sequence_rows.append(sequence_to_features(row.target_sequence))
        assay_rows.append(categorical_one_hot(row.affinity_type, AFFINITY_TYPES))
        source_rows.append(categorical_one_hot(row.source, SOURCES))

    matrix = np.concatenate(
        [
            np.asarray(fingerprint_rows, dtype=np.float32),
            np.asarray(sequence_rows, dtype=np.float32),
            np.asarray(assay_rows, dtype=np.float32),
            np.asarray(source_rows, dtype=np.float32),
        ],
        axis=1,
    )

    metadata = {
        "fingerprint_bits": fingerprint_bits,
        "fingerprint_radius": fingerprint_radius,
        "sequence_feature_count": len(AMINO_ACIDS) + 2,
        "assay_feature_count": len(AFFINITY_TYPES),
        "source_feature_count": len(SOURCES),
        "total_features": int(matrix.shape[1]),
    }
    return matrix, metadata
