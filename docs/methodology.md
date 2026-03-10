# Baseline Methodology

## Dataset

The default dataset is the Davis kinase panel:

- 68 ligands
- 442 kinase targets
- 30,056 ligand-target affinity measurements
- assay family: `Kd`

Raw data is downloaded from the DeepDTA-hosted Davis files and flattened into a
single CSV with ligand SMILES, target identifier, target sequence, and affinity.

## Processing

The processing pipeline does the following:

1. Drops rows with missing ligand, target, or affinity data
2. Validates and canonicalizes SMILES using RDKit
3. Filters to a bounded affinity range of `0.01 nM` to `1 mM`
4. Deduplicates repeated measurements by geometric mean
5. Converts affinity to `p_activity = -log10(affinity in molar units)`
6. Splits the dataset by target into train/validation/test partitions

## Feature Representation

Each ligand-target row is encoded with:

- Morgan fingerprint bits for the ligand (`1024` bits, radius `2`)
- amino-acid composition of the target sequence
- log sequence length and valid-residue fraction
- assay-type one-hot features
- source one-hot features

## Model

The baseline model is a bootstrap ensemble of ridge regressors.

Training procedure:

1. Build train and validation feature matrices
2. Sweep a small ridge-regularization grid on validation RMSE
3. Prefer the larger alpha when multiple values are effectively tied
4. Fit an ensemble on train data to calibrate predictive uncertainty on validation
5. Refit the final ensemble on train+validation with the selected alpha
6. Evaluate once on held-out test targets

## Comparison Models

The repository now supports a benchmark suite with:

- `ligand_only_ridge`: ligand-only ridge ensemble
- `ridge_ensemble`: ligand + target feature ridge ensemble
- `dual_tower_uq`: interaction-aware projected cross-feature tree ensemble with calibrated uncertainty

The comparison suite is intended to separate:

- the value of target information
- the value of interaction-aware modeling
- the value of calibrated uncertainty

## Evaluation

The project reports:

- RMSE on `p_activity`
- global Spearman correlation
- ROC-AUC with active threshold `p_activity >= 6.0`
- mean per-target Spearman
- mean top-10% enrichment
- calibrated 95% prediction-interval coverage

Per-target metrics are written to `results/baseline/test_per_target_metrics.csv`.

## Decision-Oriented Evaluation

The repository also evaluates budget-constrained prioritization policies using
saved validation and test predictions.

Implemented policies:

- mean-only ranking
- probability-of-activity ranking
- risk-adjusted ranking with validation-tuned `mean - lambda * std`

Outputs are written to:

- `results/baseline/budget_policy_metrics.csv`
- `results/baseline/budget_policy_metrics.json`
