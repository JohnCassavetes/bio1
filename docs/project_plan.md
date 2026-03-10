# Project Plan And Status

## Current Status

The repository now has a complete baseline workflow:

1. `scripts/download_data.py` downloads the Davis kinase dataset
2. `scripts/process_dataset.py` normalizes affinity data into a reusable schema
3. `scripts/train_baseline.py` trains and evaluates a target-aware baseline
4. `scripts/predict_rank.py` scores new ligand-target pairs

Implemented outputs include:

- processed train/validation/test CSV files
- saved model artifact in `models/baseline/`
- evaluation reports in `results/baseline/`
- smoke-tested inference path

## Design Decisions

### 1. Generic Target Variable

The project uses `p_activity = -log10(affinity in molar units)` as the modeling
target and preserves the original assay family in `affinity_type`.

Reason:

- `pIC50` is only correct for `IC50`
- the Davis dataset is `Kd`
- mixed assay projects should not hide the assay label

### 2. Split By Target

All rows for a target go into exactly one split.

Reason:

- random ligand-target splits inflate apparent generalization
- the project goal is ranking on unseen kinase targets

### 3. Sequence-Aware Baseline

The baseline uses ligand fingerprints plus protein-sequence composition
features rather than target one-hot encodings.

Reason:

- one-hot target IDs break on unseen targets
- sequence-derived features support target-level generalization

### 4. Calibrated Uncertainty

Bootstrap ensemble variance is scaled using the validation split so prediction
interval coverage is meaningful.

Reason:

- raw ensemble spread was under-dispersed
- the project explicitly claims uncertainty-aware ranking

## Near-Term Work

### Phase 1: Stronger Baselines

- add tree-based and shallow neural baselines
- compare RMSE, Spearman, ROC-AUC, and enrichment

### Phase 2: Better Protein Features

- replace amino-acid composition with pretrained embeddings
- compare sequence truncation vs full-sequence embeddings

### Phase 3: Better Chemistry Features

- compare ECFP variants and count fingerprints
- add graph-based ligand encoders

### Phase 4: Experiment Management

- add run manifests and parameterized output folders
- record dataset hashes and config files with each run

### Phase 5: External Data Expansion

- validate BindingDB ingestion against current live API responses
- add source-specific schema tests before merging datasets
