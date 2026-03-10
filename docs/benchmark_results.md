# Benchmark Results

## Split Families Generated

Benchmark manifests are stored in `data/benchmark/manifest.json`.

Current split families:

- `random`
- `cold_target`
- `cold_ligand`
- `scaffold`
- `both_new`

The `both_new` split discards mixed target-ligand quadrants so the held-out test
set remains strictly unseen in both dimensions.

## Ridge Baseline Comparison

Results from `results/benchmark_ridge/summary.csv`:

| Split | Best ridge-family model | RMSE | Spearman |
|------|------|------|------|
| random | ridge_ensemble | 0.768 | 0.486 |
| cold_target | ridge_ensemble | 0.730 | 0.456 |
| cold_ligand | ligand_only_ridge | 0.824 | 0.267 |
| scaffold | ridge_ensemble | 0.814 | 0.209 |
| both_new | ridge_ensemble | 0.812 | 0.292 |

Key observation:

- performance depends strongly on split family
- ligand/scaffold/both-new splits are much harsher than random
- target-held-out is not automatically the harshest Davis split

This is important for the benchmark thesis because it means a single split
family can create a misleading picture of model quality.

## Interaction-Aware Model Comparison

Results from `results/benchmark_dual_tower/summary.csv`:

| Split | Model | RMSE | Spearman | ROC-AUC |
|------|------|------|------|------|
| random | dual_tower_uq | 0.596 | 0.634 | 0.908 |
| cold_target | dual_tower_uq | 0.597 | 0.587 | 0.899 |

Compared with the ridge baselines:

- on `random`, `dual_tower_uq` improves RMSE from `0.768` to `0.596`
- on `cold_target`, `dual_tower_uq` improves RMSE from `0.730` to `0.597`
- rank correlation and ROC-AUC also improve on both evaluated splits

## Uncertainty and Decision Utility

The decision-policy diagnostics are currently reported for the original baseline
workflow in `results/baseline/budget_policy_metrics.json`.

Current lesson:

- calibrated uncertainty is informative about prediction error
- naive risk-adjusted ranking does not automatically improve budgeted hit rate

This is a useful benchmark finding because it separates:

1. predictive performance
2. uncertainty calibration
3. practical selection utility

Those dimensions should not be collapsed into a single “better model” claim.
