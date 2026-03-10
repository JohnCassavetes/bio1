#!/usr/bin/env python3
"""Download kinase-ligand binding data.

Default source: Davis kinase dataset from the DeepDTA repository.
  - 442 kinase targets x 68 ligands with Kd measurements
  - Includes target identifiers and protein sequences

Optional source: BindingDB REST API for a smaller curated kinase panel.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --source davis
    python scripts/download_data.py --source bindingdb
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
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


def download_davis(output_dir: Path = RAW_DIR) -> Path:
    """Download the Davis kinase dataset and flatten it into a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Davis ligands...")
    response = requests.get(DAVIS_LIGANDS_URL, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    ligands: Dict[str, str] = json.loads(response.text)
    ligand_ids = list(ligands.keys())
    ligand_smiles = list(ligands.values())
    logger.info("  %d ligands loaded", len(ligand_ids))

    logger.info("Downloading Davis proteins...")
    response = requests.get(DAVIS_PROTEINS_URL, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    proteins: Dict[str, str] = json.loads(response.text)
    protein_ids = list(proteins.keys())
    protein_sequences = list(proteins.values())
    logger.info("  %d proteins loaded", len(protein_ids))

    logger.info("Downloading Davis affinity matrix...")
    response = requests.get(DAVIS_AFFINITY_URL, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        affinity_matrix = np.array(
            pickle.loads(response.content, encoding="latin1"),
            dtype=np.float64,
        )
    logger.info("  Affinity matrix shape: %s", affinity_matrix.shape)

    records: List[Dict[str, Any]] = []
    for ligand_index, (ligand_id, smiles) in enumerate(zip(ligand_ids, ligand_smiles)):
        for target_index, target_id in enumerate(protein_ids):
            records.append(
                {
                    "Drug_ID": ligand_id,
                    "Drug": smiles,
                    "Target_ID": target_id,
                    "Target_Sequence": protein_sequences[target_index],
                    "Y": float(affinity_matrix[ligand_index, target_index]),
                }
            )

    df = pd.DataFrame(records)
    filepath = output_dir / "davis_dataset.csv"
    df.to_csv(filepath, index=False)
    logger.info("Davis dataset saved to %s (%d rows)", filepath, len(df))

    with open(output_dir / "davis_ligands.json", "w") as fh:
        json.dump(ligands, fh, indent=2)
    with open(output_dir / "davis_proteins.json", "w") as fh:
        json.dump(proteins, fh, indent=2)

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


def fetch_ligands_for_uniprot(
    uniprot_id: str,
    cutoff_nm: int = AFFINITY_CUTOFF_NM,
    timeout: int = REQUEST_TIMEOUT_SEC,
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """Query BindingDB for ligands that bind a target UniProt identifier."""
    params = {
        "uniprot": uniprot_id,
        "cutoff": cutoff_nm,
        "code": 0,
        "response": "application/json",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                BINDINGDB_API_URL,
                params=params,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()
            if 400 <= response.status_code < 500:
                logger.warning(
                    "Client error %d for %s; skipping target",
                    response.status_code,
                    uniprot_id,
                )
                return None

            logger.warning(
                "Server error %d for %s on attempt %d/%d",
                response.status_code,
                uniprot_id,
                attempt + 1,
                max_retries,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning(
                "Network error for %s on attempt %d/%d: %s",
                uniprot_id,
                attempt + 1,
                max_retries,
                exc,
            )

        time.sleep(2 ** (attempt + 1))

    logger.error("All retries exhausted for %s", uniprot_id)
    return None


def download_bindingdb(
    targets: Dict[str, str] = KINASE_TARGETS,
    cutoff_nm: int = AFFINITY_CUTOFF_NM,
    output_dir: Path = RAW_DIR,
) -> Dict[str, Any]:
    """Download a curated kinase panel from BindingDB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source": "bindingdb",
        "cutoff_nm": cutoff_nm,
        "targets": {},
    }

    for target_name, uniprot_id in targets.items():
        logger.info("Fetching BindingDB data for %s (%s)...", target_name, uniprot_id)
        data = fetch_ligands_for_uniprot(uniprot_id, cutoff_nm=cutoff_nm)

        if data is not None:
            filepath = output_dir / f"bindingdb_{uniprot_id}.json"
            with open(filepath, "w") as fh:
                json.dump(data, fh, indent=2)

            count = len(data) if isinstance(data, list) else 1
            log["targets"][target_name] = {
                "uniprot_id": uniprot_id,
                "status": "success",
                "file": str(filepath),
                "ligand_count": count,
            }
            logger.info("  Saved %d records to %s", count, filepath)
        else:
            log["targets"][target_name] = {
                "uniprot_id": uniprot_id,
                "status": "failed",
                "file": None,
                "ligand_count": 0,
            }
            logger.warning("  Failed for %s", target_name)

        time.sleep(2.0)

    with open(output_dir / "download_log.json", "w") as fh:
        json.dump(log, fh, indent=2)

    pd.DataFrame(
        [{"kinase_name": name, "uniprot_id": uid} for name, uid in targets.items()]
    ).to_csv(output_dir / "kinase_targets.csv", index=False)

    successes = sum(
        1 for target in log["targets"].values() if target["status"] == "success"
    )
    logger.info("Download complete: %d/%d targets succeeded", successes, len(targets))
    return log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download kinase binding data")
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if args.source == "davis":
        download_davis(output_dir=args.output_dir)
        return

    log = download_bindingdb(cutoff_nm=args.cutoff, output_dir=args.output_dir)
    if all(target["status"] == "failed" for target in log["targets"].values()):
        logger.error(
            "All BindingDB queries failed. Use the Davis dataset instead:\n"
            "  python scripts/download_data.py --source davis"
        )


if __name__ == "__main__":
    main()
