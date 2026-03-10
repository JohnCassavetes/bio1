#!/usr/bin/env python3
"""Download kinase-ligand binding data.

Primary source: Davis kinase dataset from DeepDTA GitHub repository.
  - 442 kinases x 68 ligands with Kd measurements
  - Reference: Davis et al., Nature Biotechnology 29, 1046-1051 (2011)

Alternative: BindingDB REST API (may be intermittently unavailable).

Usage:
    python scripts/download_data.py                    # Davis (default)
    python scripts/download_data.py --source davis     # Davis explicitly
    python scripts/download_data.py --source bindingdb # BindingDB API
"""

import io
import json
import pickle
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration ---

BASE_DIR = Path(__file__).resolve().parent.parent  # bio1/
RAW_DIR = BASE_DIR / "data" / "raw"

KINASE_TARGETS = {
    "EGFR": "P00533",
    "ABL1": "P00519",
    "CDK2": "P24941",
    "BRAF": "P15056",
    "SRC": "P12931",
    "VEGFR2": "P35968",
    "JAK2": "O60674",
    "AURKA": "O14965",
}

# DeepDTA Davis dataset URLs (well-established, stable)
DAVIS_LIGANDS_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/ligands_can.txt"
)
DAVIS_PROTEINS_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/proteins.txt"
)
DAVIS_AFFINITY_URL = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/Y"
)

BINDINGDB_API_URL = (
    "https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
)
AFFINITY_CUTOFF_NM = 10_000
REQUEST_TIMEOUT_SEC = 30
MAX_RETRIES = 3


# --- Davis dataset download ---


def download_davis(output_dir: Path = RAW_DIR) -> Path:
    """Download the Davis kinase dataset from DeepDTA GitHub.

    The Davis dataset contains Kd values for 442 kinases x 68 ligands.
    Downloads three files (ligands, proteins, affinity matrix) and
    combines them into a single CSV.

    Returns path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download ligands (JSON: {pubchem_cid: SMILES})
    logger.info("Downloading Davis ligands...")
    resp = requests.get(DAVIS_LIGANDS_URL, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    ligands: Dict[str, str] = json.loads(resp.text)
    ligand_ids = list(ligands.keys())
    ligand_smiles = list(ligands.values())
    logger.info("  %d ligands loaded", len(ligand_ids))

    # Download proteins (JSON: {gene_name: sequence})
    logger.info("Downloading Davis proteins...")
    resp = requests.get(DAVIS_PROTEINS_URL, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    proteins: Dict[str, str] = json.loads(resp.text)
    protein_ids = list(proteins.keys())
    protein_seqs = list(proteins.values())
    logger.info("  %d proteins loaded", len(protein_ids))

    # Download affinity matrix (numpy pickle from Python 2 era)
    logger.info("Downloading Davis affinity matrix...")
    resp = requests.get(DAVIS_AFFINITY_URL, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    raw = pickle.loads(resp.content, encoding="latin1")
    affinity_matrix = np.array(raw, dtype=np.float64)
    logger.info("  Affinity matrix shape: %s", affinity_matrix.shape)

    # Build flat DataFrame: one row per (ligand, protein, Kd)
    records: List[Dict[str, Any]] = []
    for i, (lid, smi) in enumerate(zip(ligand_ids, ligand_smiles)):
        for j, pid in enumerate(protein_ids):
            kd = float(affinity_matrix[i, j])
            records.append(
                {
                    "Drug_ID": lid,
                    "Drug": smi,
                    "Target_ID": pid,
                    "Target": protein_seqs[j][:50],  # truncate for storage
                    "Y": kd,
                }
            )

    df = pd.DataFrame(records)

    # Save
    filepath = output_dir / "davis_dataset.csv"
    df.to_csv(filepath, index=False)
    logger.info("Davis dataset saved to %s (%d rows)", filepath, len(df))

    # Also save raw files for reference
    with open(output_dir / "davis_ligands.json", "w") as fh:
        json.dump(ligands, fh, indent=2)
    with open(output_dir / "davis_proteins.json", "w") as fh:
        json.dump(proteins, fh, indent=2)

    # Save download log
    log = {
        "timestamp": datetime.now().isoformat(),
        "source": "davis",
        "url_base": "https://github.com/hkmztrk/DeepDTA",
        "num_ligands": len(ligand_ids),
        "num_proteins": len(protein_ids),
        "num_rows": len(df),
        "affinity_matrix_shape": list(affinity_matrix.shape),
    }
    with open(output_dir / "download_log.json", "w") as fh:
        json.dump(log, fh, indent=2)

    return filepath


# --- BindingDB API (alternative) ---


def fetch_ligands_for_uniprot(
    uniprot_id: str,
    cutoff_nm: int = AFFINITY_CUTOFF_NM,
    timeout: int = REQUEST_TIMEOUT_SEC,
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """Query BindingDB REST API for ligands binding a given UniProt target.

    Note: This API endpoint may be intermittently unavailable.
    Retries with exponential backoff on transient failures.
    """
    params = {
        "uniprot": uniprot_id,
        "cutoff": cutoff_nm,
        "code": 0,
        "response": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                BINDINGDB_API_URL, params=params, timeout=timeout
            )
            if resp.status_code == 200:
                return resp.json()
            elif 400 <= resp.status_code < 500:
                logger.warning(
                    "Client error %d for %s — skipping",
                    resp.status_code,
                    uniprot_id,
                )
                return None
            else:
                logger.warning(
                    "Server error %d for %s, attempt %d/%d",
                    resp.status_code,
                    uniprot_id,
                    attempt + 1,
                    max_retries,
                )
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning(
                "Network error for %s: %s, attempt %d/%d",
                uniprot_id,
                exc,
                attempt + 1,
                max_retries,
            )

        time.sleep(2 ** (attempt + 1))

    logger.error("All retries exhausted for %s", uniprot_id)
    return None


def download_bindingdb(
    targets: Dict[str, str] = KINASE_TARGETS,
    cutoff_nm: int = AFFINITY_CUTOFF_NM,
    output_dir: Path = RAW_DIR,
) -> Dict[str, Any]:
    """Download binding data for kinase targets from BindingDB API.

    Queries each target sequentially. Continues on failure for individual
    targets. Returns a download log.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source": "bindingdb",
        "cutoff_nm": cutoff_nm,
        "targets": {},
    }

    for name, uid in targets.items():
        logger.info("Fetching data for %s (%s)...", name, uid)
        data = fetch_ligands_for_uniprot(uid, cutoff_nm=cutoff_nm)

        if data is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"bindingdb_{uid}.json"
            with open(filepath, "w") as fh:
                json.dump(data, fh, indent=2)
            count = len(data) if isinstance(data, list) else 1
            log["targets"][name] = {
                "uniprot_id": uid,
                "status": "success",
                "file": str(filepath),
                "ligand_count": count,
            }
            logger.info("  -> Saved %d records to %s", count, filepath)
        else:
            log["targets"][name] = {
                "uniprot_id": uid,
                "status": "failed",
                "file": None,
                "ligand_count": 0,
            }
            logger.warning("  -> FAILED for %s", name)

        time.sleep(2.0)

    # Save log and metadata
    log_path = output_dir / "download_log.json"
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)

    targets_df = pd.DataFrame(
        [{"kinase_name": k, "uniprot_id": v} for k, v in targets.items()]
    )
    targets_df.to_csv(output_dir / "kinase_targets.csv", index=False)

    successes = sum(
        1 for t in log["targets"].values() if t["status"] == "success"
    )
    logger.info(
        "Download complete: %d/%d targets succeeded", successes, len(targets)
    )
    return log


# --- CLI ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download kinase binding data"
    )
    parser.add_argument(
        "--source",
        choices=["davis", "bindingdb"],
        default="davis",
        help="Data source (default: davis)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=AFFINITY_CUTOFF_NM,
        help=f"Affinity cutoff in nM for BindingDB (default: {AFFINITY_CUTOFF_NM})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory (default: {RAW_DIR})",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if args.source == "davis":
        download_davis(output_dir=args.output_dir)
    elif args.source == "bindingdb":
        log = download_bindingdb(
            cutoff_nm=args.cutoff, output_dir=args.output_dir
        )
        all_failed = all(
            t["status"] == "failed" for t in log["targets"].values()
        )
        if all_failed:
            logger.error(
                "All BindingDB queries failed. Use the Davis dataset instead:\n"
                "  python scripts/download_data.py --source davis"
            )


if __name__ == "__main__":
    main()
