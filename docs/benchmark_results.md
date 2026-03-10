# Benchmark Results

## Split Families Generated

Benchmark manifests are stored in `data/benchmark/manifest.json`.

Current split families:

- `random`
- `cold_target`
- `cold_ligand`
- `scaffold`
- `both_new`
- `mutation_holdout`

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
| mutation_holdout | ridge_ensemble | 1.158 | 0.463 |

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

## Mutation-Transfer Benchmark

Results from `results/benchmark_mutation/summary.csv`:

| Split | Model | RMSE | Spearman | ROC-AUC |
|------|------|------|------|------|
| mutation_holdout | ligand_only_ridge | 1.203 | 0.388 | 0.729 |
| mutation_holdout | ridge_ensemble | 1.158 | 0.463 | 0.764 |
| mutation_holdout | dual_tower_uq | 0.545 | 0.818 | 0.971 |

This is currently the strongest scientist-facing result in the repo. It suggests
that transfer from wild-type targets to mutant kinase variants is a materially
different problem from ordinary held-out-target evaluation, and that the
interaction-aware model is much stronger on that setting than the ridge
baselines.

## Uncertainty and Decision Utility

The decision-policy diagnostics are currently reported for the original baseline
workflow in `results/baseline/budget_policy_metrics.json`.

Current lesson:

- calibrated uncertainty is informative about prediction error
- naive risk-adjusted ranking does not automatically improve budgeted hit rate
- conformal intervals give a clearer abstain-vs-decide picture than raw standard deviations alone

Conformal summary from `results/baseline/conformal_metrics.json`:

- normalized conformal at `alpha=0.10`: coverage `0.895`, mean interval width `2.032`
- normalized conformal at `alpha=0.05`: coverage `0.952`, mean interval width `3.289`

This is a useful benchmark finding because it separates:

1. predictive performance
2. uncertainty calibration
3. practical selection utility

Those dimensions should not be collapsed into a single “better model” claim.
