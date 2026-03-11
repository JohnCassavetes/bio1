# Benchmark Protocol

## Objective

The benchmark is designed to answer a scientist-facing question:

**Under a fixed assay budget, which method most reliably prioritizes compounds
for a kinase target while representing uncertainty honestly?**

This is intentionally different from only minimizing regression error.

## Dataset Semantics

The benchmark keeps assay labels explicit.

- raw affinity field: `affinity_nm`
- assay family: `affinity_type`
- transformed target: `p_activity`
- derived label: `activity_label`

Mixed-assay evaluation should be treated carefully. The current Davis workflow
is assay-homogeneous (`Kd`), but the schema is prepared for future mixed-source
benchmarks.

## Current Split

The current benchmark now treats `cold_target` as the baseline in-domain split,
but it is no longer the only realistic setting.

Rationale:

- random interaction-level splits are too optimistic
- target-held-out evaluation better reflects generalization to unseen kinases

## Implemented Split Families

The repository now generates:

1. random interaction split
2. cold-target split
3. cold-ligand split
4. both-new split
5. ligand-similarity-aware split
   implemented here as scaffold-based ligand grouping
6. target-sequence-identity-aware split
   implemented via exact global pairwise alignment identity clustering
7. mutation-holdout split

External validation:

1. train on Davis, validate on Davis validation, test on processed BindingDB kinase panel

## Core Metrics

### Standard Predictive Metrics

- RMSE on `p_activity`
- global Spearman correlation
- mean per-target Spearman
- ROC-AUC for `p_activity >= 6.0`
- top-10% enrichment

### Budget-Constrained Metrics

For budgets such as `1`, `3`, `5`, and `10` compounds per target:

- hit rate at budget
- mean selected `p_activity`
- regret against oracle top-k selection

### Uncertainty Metrics

- calibrated 95% interval coverage
- mean predicted standard deviation
- rank correlation between predicted uncertainty and empirical error
- selective-performance diagnostics
- split-conformal and normalized-conformal interval quality

## Policy Evaluation

The repo currently evaluates:

- mean-only ranking
- probability-of-activity ranking
- risk-adjusted ranking with validation-tuned `mean - lambda * std`

The repo also supports comparison models:

- `ligand_only_ridge`
- `ridge_ensemble`
- `dual_tower_uq`
- `deepdta_exact`
- `graphdta_gcn_exact`

The main lesson from the initial baseline is that calibrated uncertainty can be
informative without automatically yielding a better decision policy. That is why
the benchmark separates:

- predictive quality
- calibration quality
- decision utility
