# KinBench-UQ: Realistic Kinase-Ligand Prioritization with Exact Literature Baselines, External Validation, and Generated Evidence

## Abstract

Drug-target affinity prediction for kinases is often evaluated on optimistic
interaction-level splits and summarized with aggregate regression metrics that
do not directly answer the experimental prioritization question. We present
**KinBench-UQ**, a benchmark and baseline suite for kinase-ligand candidate
prioritization that centers realistic generalization, mutation transfer,
external validation, and uncertainty-aware evaluation. The benchmark enforces
assay-aware semantics, includes random, cold-target, cold-ligand, scaffold,
both-new, exact sequence-identity-aware, and mutation-holdout settings, and
adds a Davis-to-BindingDB external validation protocol. On the currently
generated key splits, the interaction-aware `dual_tower_uq` model is the
strongest repo-native baseline, reaching RMSE/Spearman of `0.596/0.634` on the
random split, `0.597/0.587` on cold-target, `0.701/0.461` on the exact
sequence-identity split, and `0.545/0.818` on mutation holdout. On external
validation, the same model improves RMSE from `1.602` for the target-aware
ridge ensemble to `1.484` and improves mean per-target Spearman from `0.155` to
`0.342`. The repository also reports budget-constrained selection utility,
conformal uncertainty diagnostics, per-family mutation analysis, and
nearest-train sequence-identity leakage summaries. All tables and figures in
this draft are generated from `results/*` by `scripts/generate_paper_assets.py`;
the generated artifact bundle is written to `paper/generated_results.md`.

## 1. Introduction

Kinase inhibitor prioritization is not just a regression problem. In practice, a
scientist must choose which compounds to assay first under a fixed budget, often
for targets or target variants that are not cleanly represented by the training
distribution. Standard DTA benchmarks such as Davis and KIBA have been crucial
for the field, but they also encourage a narrow evaluation habit: optimize RMSE
or concordance under splits that may retain substantial target or sequence
similarity between training and test sets. That framing is insufficient for a
candidate-prioritization paper.

KinBench-UQ is designed around a stricter question:

> Under realistic train/test separation, which model best prioritizes kinase
> compounds while preserving calibrated uncertainty and remaining reproducible
> under a common protocol?

This paper makes four contributions.

1. It defines a realistic kinase prioritization benchmark with exact
   sequence-identity-aware splitting, mutation-family transfer, and external
   Davis-to-BindingDB evaluation.
2. It reruns exact-architecture DeepDTA-style and GraphDTA-style baselines
   under the same protocol as the repo baselines, avoiding weak comparisons to
   incompatible published numbers.
3. It evaluates predictive quality, leakage, mutation-family behavior, and
   budget-constrained selection utility in one reporting framework.
4. It makes the manuscript evidence generated rather than hand-curated: the
   reported tables and figures come directly from versioned result artifacts.

## 2. Related Work

### 2.1 Drug-Target Affinity Modeling

DeepDTA established the modern sequence-and-SMILES CNN baseline for Davis and
KIBA. GraphDTA extended this line by replacing fixed ligand encodings with
molecular graph neural networks while keeping the target branch sequence-based.
Those methods remain essential comparison points, but their original reported
numbers are not directly comparable under stricter benchmark protocols.

### 2.2 Realistic Evaluation

Recent DTA work increasingly emphasizes similarity-aware evaluation and more
realistic kinase formulations. These studies motivate the present benchmark but
do not by themselves provide a unified, target-aware prioritization protocol
with exact literature reruns, mutation-family analysis, external validation,
and generated manuscript evidence.

### 2.3 Uncertainty and Selection Utility

Uncertainty estimation and conformal prediction have become common in DTA and
DTI modeling. KinBench-UQ treats that literature as a motivation for stronger
evaluation rather than as a standalone novelty claim. In this benchmark,
uncertainty matters only if it improves calibration, abstention behavior, or
selection utility under assay budgets.

## 3. Benchmark Design

### 3.1 Data Semantics

The benchmark stores assay family explicitly in `affinity_type` and uses
`p_activity = -log10(affinity in molar units)` as the unified regression target.
This prevents silent label collapse when future external datasets include mixed
`Kd`, `Ki`, or `IC50` measurements.

### 3.2 Datasets

KinBench-UQ uses:

- Davis as the primary training and in-domain benchmark dataset.
- BindingDB-derived kinase assays as an external-source evaluation set,
  processed into the same schema and filtered for assay comparability.

### 3.3 Split Families

The benchmark includes:

- random interaction split
- cold-target split
- cold-ligand split
- scaffold split
- both-new split
- exact sequence-identity-aware split
- mutation-holdout split

The sequence-identity split clusters targets by exact global pairwise alignment
identity rather than a sequence-similarity proxy. The mutation-holdout split
keeps wild-type family context in training while reserving mutant family members
for evaluation. In addition to split definitions, the benchmark exports
nearest-train identity leakage summaries for each test target.

## 4. Models

KinBench-UQ evaluates five comparison models under one benchmark runner.

1. `ligand_only_ridge`
2. `ridge_ensemble`
3. `dual_tower_uq`
4. `deepdta_exact`
5. `graphdta_gcn_exact`

The first three are repo-native baselines. The last two are exact-architecture
literature reproductions run under the same train/validation/test protocol and
artifact format. The objective is benchmark comparability, not architecture
novelty.

## 5. Evaluation Protocol

### 5.1 Predictive Metrics

- RMSE
- global Spearman correlation
- mean per-target Spearman
- ROC-AUC at `p_activity >= 6.0`
- top-10% enrichment

### 5.2 Decision Metrics

For fixed budgets per target, the benchmark reports:

- hit rate at budget
- mean selected `p_activity`
- regret relative to oracle top-k selection

### 5.3 Uncertainty Diagnostics

- calibrated Gaussian interval coverage
- split-conformal and normalized-conformal interval quality
- uncertainty-error rank correlation
- abstention-region summaries

### 5.4 Leakage and Mutation Analysis

The benchmark also exports:

- nearest-train sequence identity for each evaluation target
- split-level leakage summaries
- per-family mutation metrics
- wild-type versus mutant activity deltas

## 6. Results

The manuscript does not rely on hand-maintained tables. All result summaries
are generated from the artifact tree:

- `results/benchmark/summary.csv`
- `results/benchmark/summary.json`
- `results/external_validation/bindingdb/summary.csv`
- `results/benchmark_analysis/*.csv`
- `figures/*.png`
- `paper/generated_results.md`

The canonical regeneration step is:

```bash
python scripts/generate_paper_assets.py
```

### 6.1 Split Difficulty and Model Ranking

The generated summary in `paper/generated_results.md` shows that split choice
materially changes both absolute error and relative model ranking. Across the
four key paper splits currently regenerated in this workspace, `dual_tower_uq`
is the best repo-native model by RMSE:

- random: RMSE `0.596`, Spearman `0.634`
- cold-target: RMSE `0.597`, Spearman `0.587`
- sequence-identity: RMSE `0.701`, Spearman `0.461`
- mutation-holdout: RMSE `0.545`, Spearman `0.818`

The sequence-identity split is materially harsher than the random split for all
models. For the ridge ensemble, RMSE worsens from `0.768` on random to `0.714`
on sequence identity, even though both are evaluated on Davis. This is the main
benchmark claim: exact similarity control changes the apparent difficulty of the
task.

### 6.2 Leakage Analysis

The generated leakage summary in `results/benchmark_analysis/leakage_summary.csv`
shows that the sequence-identity split is the only currently regenerated split
with a materially reduced nearest-train identity ceiling (`mean 0.367`, `max
0.581`). By contrast, random, scaffold, mutation-holdout, and cold-ligand
evaluation still admit exact or near-exact train-test neighbors at the target
level. This does not invalidate those splits, but it clarifies what they do and
do not test.

### 6.3 Mutation-Family Transfer

Mutation-family analysis in
`results/benchmark_analysis/mutation_family_analysis.csv` shows that the
mutation-holdout setting is not uniform across kinase families. The strongest
repo-native model (`dual_tower_uq`) maintains low family-level RMSE for several
families, including `PIK3CA` (`0.150`), `LRRK2` (`0.334`), and `BRAF` (`0.389`),
while the ridge baselines degrade substantially on larger families such as `KIT`
and `FLT3`. These family-level outputs are important because clinically relevant
resistance evaluation is usually family-specific, not just aggregate.

### 6.4 External Validation

External evaluation against the processed BindingDB kinase panel yields a
substantially harder regime than in-domain Davis testing. In the generated
external summary:

- `dual_tower_uq`: RMSE `1.484`, mean per-target Spearman `0.342`
- `ridge_ensemble`: RMSE `1.602`, mean per-target Spearman `0.155`
- `ligand_only_ridge`: RMSE `1.641`, mean per-target Spearman `0.166`

These results are not yet competitive with in-domain Davis performance, but
they are scientifically more credible for a candidate-prioritization story. The
paper should emphasize this external regime rather than over-indexing on random
split results.

### 6.5 Figures and Tables

The manuscript evidence should draw from generated figures only:

- `figures/split_rmse_comparison.png`
- `figures/split_spearman_comparison.png`
- `figures/sequence_identity_leakage.png`
- `figures/mutation_family_delta.png`
- `figures/external_validation_rmse.png`

The generated tables live in `paper/generated_results.md` and are sourced from:

- `results/benchmark/summary.csv`
- `results/external_validation/bindingdb/summary.csv`
- `results/benchmark_analysis/*.csv`

## 7. Reproducibility

The intended environment is pinned in:

- `environment.yml`
- `requirements-lock.txt`

The intended end-to-end command is:

```bash
python scripts/run_submission_pipeline.py --device cpu
```

That pipeline downloads data, builds the external validation set, generates
split families, runs benchmark models, runs external validation, and regenerates
paper assets.

## 8. Limitations

The benchmark still depends on public retrospective datasets and does not
replace prospective medicinal-chemistry validation. External-source processing
also inherits the noise and assay heterogeneity of upstream databases. In
addition, exact-architecture literature baselines are part of the benchmark
protocol, but their regenerated result files should be treated as protocol
artifacts only after they have been fully executed and frozen in the manuscript
bundle.

## References

1. Davis MI, Hunt JP, Herrgard S, et al. Comprehensive analysis of kinase
   inhibitor selectivity. *Nature Biotechnology* (2011).
2. Ozturk H, Ozkirimli E, Ozgur A. DeepDTA: deep drug-target binding affinity
   prediction. *Bioinformatics* (2018).
3. Nguyen T, Le H, Quinn T, et al. GraphDTA: predicting drug-target binding
   affinity with graph neural networks. *Bioinformatics* (2021).
4. Additional references should be finalized in the venue bibliography file once
   the generated result section is frozen.
