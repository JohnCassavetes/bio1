# Paper Reproduction Guide

This document is the shortest practical path for a scientist or reviewer who
wants to reproduce the benchmark artifacts in this repository.

## 1. Environment

Preferred:

```bash
conda env create -f environment.yml
conda activate kinbench-uq
```

Alternative:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 2. Main Scientist-Facing Entry Points

Use these files if you want to understand or rerun the benchmark:

- main benchmark runner: `scripts/run_benchmark_suite.py`
- external validation runner: `scripts/run_external_validation.py`
- benchmark split generation: `scripts/generate_benchmark_splits.py`
- paper asset generation: `scripts/generate_paper_assets.py`
- one-command pipeline: `scripts/run_submission_pipeline.py`
- prediction on new candidates: `scripts/predict_rank.py`

Main model code:

- repo-native model: `src/kinase_ligand_ranking/neural_modeling.py`
  class: `DualTowerUncertaintyRanker`
- literature baselines: `src/kinase_ligand_ranking/literature_models.py`
  models: `deepdta_exact`, `graphdta_gcn_exact`
- feature encoding: `src/kinase_ligand_ranking/features.py`
- split definitions: `src/kinase_ligand_ranking/splits.py`

## 3. Minimal Full Reproduction

If you want the shortest end-to-end path:

```bash
python scripts/run_submission_pipeline.py --device cpu
```

This performs:

1. Davis download
2. Davis processing
3. benchmark split generation
4. baseline benchmark runs
5. BindingDB external dataset build
6. external validation
7. paper-asset generation

## 4. Explicit Step-by-Step Reproduction

### 4.1 Build the main Davis benchmark

```bash
python scripts/download_data.py --source davis
python scripts/process_dataset.py --source davis
python scripts/generate_benchmark_splits.py
```

### 4.2 Run the main benchmark comparisons

```bash
python scripts/run_benchmark_suite.py \
  --splits random cold_target sequence_identity mutation_holdout \
  --models ligand_only_ridge ridge_ensemble dual_tower_uq deepdta_exact graphdta_gcn_exact \
  --results-dir results/benchmark \
  --device cpu
```

### 4.3 Build the external validation set

```bash
python scripts/download_data.py --source bindingdb
python scripts/build_external_validation.py
```

### 4.4 Run external validation

If you want the clean final state, run all benchmark models together:

```bash
python scripts/run_external_validation.py \
  --models ligand_only_ridge ridge_ensemble dual_tower_uq deepdta_exact graphdta_gcn_exact \
  --results-dir results/external_validation/bindingdb \
  --device cpu
```

### 4.5 Generate paper figures and tables

```bash
python scripts/generate_paper_assets.py \
  --results-dir results/benchmark \
  --external-results-dir results/external_validation/bindingdb
```

## 5. Output Files That Matter Most

Main benchmark:

- `results/benchmark/summary.csv`
- `results/benchmark/summary.json`

External validation:

- `results/external_validation/bindingdb/summary.csv`
- `results/external_validation/bindingdb/summary.json`

Generated analysis:

- `results/benchmark_analysis/benchmark_overview.csv`
- `results/benchmark_analysis/leakage_summary.csv`
- `results/benchmark_analysis/mutation_family_analysis.csv`

Generated paper evidence:

- `paper/generated_results.md`
- `figures/split_rmse_comparison.png`
- `figures/split_spearman_comparison.png`
- `figures/sequence_identity_leakage.png`
- `figures/mutation_family_delta.png`
- `figures/external_validation_rmse.png`

## 6. What the Current Benchmark Actually Compares

The main benchmark compares:

- `ligand_only_ridge`
- `ridge_ensemble`
- `dual_tower_uq`
- `deepdta_exact`
- `graphdta_gcn_exact`

The central interpretation is:

- `DeepDTA` is strongest on the easier `random` split
- `dual_tower_uq` is strongest on `cold_target`
- `DeepDTA` is best on `sequence_identity` by RMSE, while `dual_tower_uq` is slightly better there by Spearman
- `dual_tower_uq` is strongest on `mutation_holdout`

## 7. What Reviewers Can Verify

This repository is designed so the paper-facing numbers are tied to saved
artifacts rather than hand-written tables.

A reviewer can verify:

- model predictions in `test_predictions.csv`
- split metrics in `metrics.json`
- merged summaries in `summary.csv`
- generated paper summaries in `paper/generated_results.md`

## 8. Practical Notes

- `GraphDTA` and `DeepDTA` are much slower than the ridge baselines
- CPU-only runs on older laptops can take many hours
- `sequence_identity` split generation is slower than the other split families
- `summary.csv` files are rewritten from the models requested in each run, so
  avoid mixing partial reruns into the same output directory unless you intend
  to replace the full summary
