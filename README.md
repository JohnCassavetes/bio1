# Kinase Ligand Ranking Baseline

Target-aware baseline pipeline for ranking small-molecule ligands against kinase
targets by predicted binding strength. The repository now runs end to end:
download data, process it into a consistent schema, train a baseline model,
evaluate ranking quality, and score new ligand-target pairs with uncertainty.

## What The Project Does

The current baseline is designed for candidate prioritization, not drug
discovery claims.

It provides:

1. A reproducible data pipeline for kinase-ligand affinity data
2. A consistent `p_activity` target across `Kd`, `Ki`, and `IC50` measurements
3. A target-aware baseline model using Morgan fingerprints plus protein-sequence features
4. Ranking metrics on held-out kinase targets
5. Uncertainty estimates calibrated on validation data

## Current Workflow

```text
raw affinity data
    -> cleaning + canonical SMILES
    -> target-aware train/val/test split
    -> ligand fingerprints + protein sequence features
    -> ridge ensemble regressor
    -> ranked ligands + uncertainty
```

## Repository Layout

```text
bio1/
  README.md
  requirements.txt
  data/
    raw/
    processed/
  docs/
    methodology.md
    project_plan.md
    results_baseline.md
  models/
    baseline/
  results/
    baseline/
  scripts/
    download_data.py
    process_dataset.py
    train_baseline.py
    predict_rank.py
  src/kinase_ligand_ranking/
  tests/
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download the Davis kinase dataset
python scripts/download_data.py --source davis

# 2. Clean and split the dataset
python scripts/process_dataset.py --source davis

# 3. Train and evaluate the baseline
python scripts/train_baseline.py

# 4. Score new ligand-target pairs
python scripts/predict_rank.py --input path/to/candidates.csv
```

## Data Semantics

The processed dataset keeps the original assay family in `affinity_type`, but
the regression target is `p_activity = -log10(affinity in molar units)`.

This matters because:

- `Kd`, `Ki`, and `IC50` are not interchangeable assay labels
- the previous `pic50` naming was wrong for mixed-type data
- the repo now preserves both the raw assay type and the generic transformed target

Processed CSV columns include:

- `smiles`
- `target_id`
- `target_label`
- `target_sequence`
- `affinity_type`
- `activity_label`
- `affinity_nm`
- `p_activity`
- `measurement_count`
- `source`

## Baseline Model

The current model is a bootstrap ensemble of ridge regressors trained on:

- Morgan fingerprints for ligands
- amino-acid composition and length-derived target sequence features
- assay-type one-hot features
- source one-hot features

Model selection uses the validation split to choose ridge regularization, then
refits on train+validation before final evaluation on held-out test targets.

## Latest Local Results

Artifacts were regenerated locally on March 10, 2026 with:

```bash
python scripts/download_data.py --source davis
python scripts/process_dataset.py --source davis
python scripts/train_baseline.py
```

Held-out test metrics from [metrics.json](/Users/a/Desktop/bio1/results/baseline/metrics.json):

- RMSE: `0.779`
- Global Spearman: `0.454`
- ROC-AUC at `p_activity >= 6.0`: `0.797`
- Mean per-target Spearman: `0.465`
- Mean top-10% enrichment: `3.424x`
- 95% interval coverage after calibration: `0.952`

## Inference Input Format

`scripts/predict_rank.py` expects a CSV with:

- required: `smiles`, `target_id`
- optional: `target_sequence`, `affinity_type`, `source`

Predictions are written with:

- `predicted_p_activity`
- `prediction_std`

## Tests

```bash
python -m unittest discover -s tests
```

## Roadmap

- Add stronger nonlinear baselines
- Add explicit BindingDB support validation once API schema is stabilized
- Add richer protein features beyond amino-acid composition
- Add experiment tracking and hyperparameter sweeps
