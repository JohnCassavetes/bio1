# Kinase Ligand Ranking Pipeline

Uncertainty-aware machine learning system for ranking small-molecule ligands
by predicted binding affinity to kinase protein targets. Designed to help
prioritize candidate molecules for experimental testing.

## Project Overview

This pipeline:
1. Takes a kinase protein target
2. Evaluates candidate ligands (small molecules)
3. Predicts binding affinity scores (pIC50)
4. Ranks ligands by predicted binding strength
5. Estimates prediction uncertainty
6. Outputs a shortlist of top candidate molecules with confidence intervals

## Repository Structure

```
bio1/
  README.md           - This file
  requirements.txt    - Python dependencies
  docs/               - Project documentation and plans
  memory/             - Agent memory / context files
  data/               - Raw and processed datasets
    raw/              - Downloaded data from BindingDB
    processed/        - Cleaned, filtered, split datasets
  models/             - Saved model checkpoints
  scripts/            - Data download, processing, training scripts
  notebooks/          - Exploratory analysis notebooks
  results/            - Evaluation metrics, predictions
  figures/            - Plots and visualizations
  paper/              - Manuscript drafts
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download kinase binding data (BindingDB API)
python scripts/download_data.py

# Or use the Davis kinase dataset as fallback
python scripts/download_data.py --source davis

# Process the dataset
python scripts/process_dataset.py
```

## Data Sources

- **BindingDB** (primary): REST API for protein-ligand binding measurements
  - https://www.bindingdb.org
- **Davis Kinase Dataset** (fallback): 442 kinases x 68 ligands with Kd values
  - Available via Therapeutics Data Commons (TDC)

## Target Kinases

| Kinase   | UniProt ID | Relevance                      |
|----------|------------|--------------------------------|
| EGFR     | P00533     | Non-small cell lung cancer     |
| ABL1     | P00519     | Chronic myeloid leukemia       |
| CDK2     | P24941     | Cell cycle regulation          |
| BRAF     | P15056     | Melanoma                       |
| SRC      | P12931     | Multiple cancers               |
| VEGFR2   | P35968     | Angiogenesis                   |
| JAK2     | O60674     | Myeloproliferative disorders   |
| Aurora A | O14965     | Mitotic kinase                 |

## Pipeline

```
protein + ligand dataset
        |
molecule feature extraction (RDKit)
        |
protein representation (embeddings)
        |
graph neural network (PyTorch Geometric)
        |
binding affinity prediction (pIC50)
        |
ligand ranking
        |
uncertainty estimation (MC Dropout)
        |
top candidate molecules + confidence intervals
```

## Evaluation Metrics

- **Spearman correlation** - rank correlation between predicted and actual affinities
- **Top-k enrichment** - fraction of true actives in top-k predictions vs. random
- **ROC-AUC** - discrimination between active and inactive compounds

## Key Tools

- [RDKit](https://www.rdkit.org) - molecule processing and featurization
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) - graph neural networks
- [BindingDB](https://www.bindingdb.org) - protein-ligand binding data

## Disclaimer

This is a computational screening tool for candidate prioritization.
It does not claim to discover drugs. All outputs should be validated
experimentally before drawing conclusions.
