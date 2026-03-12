# KinBench-UQ

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

## Why This Benchmark Exists

Scientists use AI models to help decide which kinase-targeting compounds are
worth testing in the lab. The problem is that many standard evaluations are too
easy: proteins in the test set can still be very similar to proteins the model
already saw during training. That can make a model look stronger than it really
is.

An easy way to think about it is:

> if you train someone to recognize dogs, but the test only shows them dogs
> that look almost identical to the ones they already studied, the test will
> overstate how well they generalize

KinBench-UQ exists to make that evaluation stricter and more realistic for
kinase-specific candidate prioritization. In particular, it adds
sequence-identity-aware splitting, mutation-family evaluation, external
validation, and generated analysis artifacts that make leakage and split
difficulty visible.

So the main contribution is not simply "a better model." It is also "a better
exam" for evaluating kinase prioritization models.

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
    benchmark_protocol.md
    benchmark_results.md
    methodology.md
    project_plan.md
    results_baseline.md
  paper/
    draft.md
  models/
    baseline/
  results/
    baseline/
  scripts/
    download_data.py
    process_dataset.py
    generate_benchmark_splits.py
    train_baseline.py
    predict_rank.py
    evaluate_budgeted_policies.py
    evaluate_conformal.py
    run_benchmark_suite.py
  src/kinase_ligand_ranking/
  tests/
```

## Reproducible Environment

The intended paper-grade environment is pinned for Python 3.12.

```bash
conda env create -f environment.yml
conda activate kinbench-uq
```

If you prefer `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download the Davis kinase dataset
python scripts/download_data.py --source davis

# 2. Clean and split the dataset
python scripts/process_dataset.py --source davis

# 3. Generate benchmark split families
python scripts/generate_benchmark_splits.py

# 4. Train and evaluate the baseline
python scripts/train_baseline.py

# 5. Evaluate budget-constrained decision policies
python scripts/evaluate_budgeted_policies.py

# 6. Evaluate split-conformal uncertainty
python scripts/evaluate_conformal.py

# 7. Download and build the external validation set
python scripts/download_data.py --source bindingdb
python scripts/build_external_validation.py

# 8. Run the full benchmark suite
python scripts/run_benchmark_suite.py \
  --splits random cold_target cold_ligand scaffold both_new sequence_identity mutation_holdout \
  --models ligand_only_ridge ridge_ensemble dual_tower_uq deepdta_exact graphdta_gcn_exact

# 9. Run external-source validation
python scripts/run_external_validation.py \
  --models ligand_only_ridge ridge_ensemble dual_tower_uq deepdta_exact graphdta_gcn_exact

# 10. Generate manuscript figures and tables from results/*
python scripts/generate_paper_assets.py

# 11. Score new ligand-target pairs
python scripts/predict_rank.py --input path/to/candidates.csv
```

One-command submission pipeline:

```bash
python scripts/run_submission_pipeline.py --device cpu
```

## Scientist Entry Points

If you only want the main files for reproducing the paper or comparing models,
start here:

- benchmark runner: [scripts/run_benchmark_suite.py](scripts/run_benchmark_suite.py)
- external validation runner: [scripts/run_external_validation.py](scripts/run_external_validation.py)
- split generation: [scripts/generate_benchmark_splits.py](scripts/generate_benchmark_splits.py)
- manuscript asset generation: [scripts/generate_paper_assets.py](scripts/generate_paper_assets.py)
- one-command pipeline: [scripts/run_submission_pipeline.py](scripts/run_submission_pipeline.py)
- new-candidate scoring: [scripts/predict_rank.py](scripts/predict_rank.py)

Main model code:

- repo-native model: [neural_modeling.py](src/kinase_ligand_ranking/neural_modeling.py)
  class: `DualTowerUncertaintyRanker`
- literature baselines: [literature_models.py](src/kinase_ligand_ranking/literature_models.py)
  models: `deepdta_exact`, `graphdta_gcn_exact`
- feature construction: [features.py](src/kinase_ligand_ranking/features.py)
- split definitions: [splits.py](src/kinase_ligand_ranking/splits.py)

If you only want to compare methods under the main benchmark, run:

```bash
python scripts/run_benchmark_suite.py \
  --splits random cold_target sequence_identity mutation_holdout \
  --models ligand_only_ridge ridge_ensemble dual_tower_uq deepdta_exact graphdta_gcn_exact \
  --results-dir results/benchmark \
  --device cpu
```

For a step-by-step reproducibility path, see
[docs/reproduce_paper.md](docs/reproduce_paper.md).

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

Held-out test metrics from [metrics.json](results/baseline/metrics.json):

- RMSE: `0.779`
- Global Spearman: `0.454`
- ROC-AUC at `p_activity >= 6.0`: `0.797`
- Mean per-target Spearman: `0.465`
- Mean top-10% enrichment: `3.424x`
- 95% interval coverage after calibration: `0.952`

Decision-oriented results from [budget_policy_metrics.json](results/baseline/budget_policy_metrics.json):

- budget-1 hit rate: `0.785`
- budget-3 hit rate: `0.646`
- budget-5 hit rate: `0.600`
- budget-10 hit rate: `0.506`
- uncertainty/error Spearman: `0.431`

The current calibrated uncertainty is informative about error, but the simple
risk-adjusted selection policy does not yet beat mean-only ranking on the Davis
test split. That is now part of the benchmark story rather than something being
hidden.

Conformal uncertainty results from [conformal_metrics.json](results/baseline/conformal_metrics.json):

- normalized conformal at `alpha=0.10`: coverage `0.895`, mean width `2.032`
- normalized conformal at `alpha=0.05`: coverage `0.952`, mean width `3.289`
- at `alpha=0.10`, the model abstains on about `69.5%` of cases while making a large confident-inactive region

Benchmark comparison results:

- ridge baselines across `random`, `cold_target`, `cold_ligand`, `scaffold`, and `both_new` are summarized in [results/benchmark_ridge/summary.csv](results/benchmark_ridge/summary.csv)
- the interaction-aware `dual_tower_uq` model is summarized in [results/benchmark_dual_tower/summary.csv](results/benchmark_dual_tower/summary.csv)
- the mutation-transfer benchmark is summarized in [results/benchmark_mutation/summary.csv](results/benchmark_mutation/summary.csv)
- the split manifests live in [data/benchmark/manifest.json](data/benchmark/manifest.json)
- exact-architecture DeepDTA and GraphDTA-style baselines run through [scripts/run_benchmark_suite.py](scripts/run_benchmark_suite.py)
- external-source validation is produced by [scripts/run_external_validation.py](scripts/run_external_validation.py)
- manuscript evidence is generated by [scripts/generate_paper_assets.py](scripts/generate_paper_assets.py)

Current takeaways:

- ligand/scaffold/both-new splits are materially harsher than random splits for the ridge baselines
- target-held-out is not automatically the hardest split on Davis
- the interaction-aware model materially outperforms the ridge baselines on the currently run `random` and `cold_target` splits
- on the new `mutation_holdout` split, `dual_tower_uq` is much stronger than the ridge baselines

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

## Current Protocol Upgrades

- exact target-sequence-identity split generation using global pairwise alignment
- mutation-family analysis and leakage summaries as generated artifacts
- exact-architecture `deepdta_exact` and `graphdta_gcn_exact` comparison baselines
- Davis-to-BindingDB external validation
- generated manuscript tables and figures from `results/*`
