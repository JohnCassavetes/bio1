# KinBench-UQ: A Realistic Kinase Affinity Benchmark with Calibrated Uncertainty for Budget-Constrained Compound Prioritization

## Abstract

Drug-target affinity prediction for kinases is commonly evaluated with metrics
such as RMSE or rank correlation on benchmark datasets such as Davis and KIBA.
However, standard evaluations are often optimistic, collapse distinct assay
semantics, and say little about whether a model helps a scientist choose which
compounds to test under a fixed experimental budget. We present **KinBench-UQ**,
a benchmark and evaluation framework for kinase affinity prediction that centers
realistic generalization, calibration, and decision utility. KinBench-UQ keeps
assay types explicit and implements multiple split families, including
ligand-held-out, scaffold-based, both-new, and mutation-holdout evaluation. As
comparison models, we implement ligand-only ridge, ligand-plus-target ridge, and
an interaction-aware projected cross-feature ensemble with calibrated
uncertainty. On Davis, the harsher split families substantially change the
conclusions suggested by standard evaluation: for the ridge ensemble, RMSE rises
from `0.768` on random splits to `0.814` on scaffold splits and `0.812` on
both-new splits. On the mutation-holdout benchmark, the interaction-aware model
substantially outperforms the ridge baselines, reaching RMSE `0.545`, Spearman
`0.818`, and ROC-AUC `0.971`. We also show that uncertainty is informative about
error but does not automatically improve ranking under naive risk-adjusted
selection, motivating conformal and selective evaluation alongside classical
predictive metrics. These results position KinBench-UQ as a benchmark for
realistic kinase prioritization rather than another optimistic affinity
leaderboard.

## 1. Introduction

Machine learning for drug-target affinity (DTA) prediction is already a mature
literature. Sequence-and-SMILES architectures such as DeepDTA demonstrated that
learned models could predict affinities on Davis and KIBA, and graph-based
variants such as GraphDTA later improved model expressivity by using molecular
graphs rather than fixed fingerprints alone. Recent reviews document a broad
ecosystem of DTA methods, losses, and encoders across proteins and ligands.

Despite that progress, three gaps remain especially important for kinase
screening:

1. **Optimistic evaluation**. Standard benchmark scores may overstate practical
   generalization because train and test samples can remain too similar.
2. **Weak decision framing**. RMSE and Spearman do not directly answer the lab
   question: which compounds should be tested first under a fixed assay budget?
3. **Underspecified uncertainty**. Predictive confidence is often reported, but
   calibration and decision utility are less often evaluated explicitly.

These gaps are now visible in recent literature. A 2025 study on DTA evaluation
argues that similarity-aware data splitting changes the picture materially and
that conventional protocols can be misleading. A 2025 conformal-prediction
study for DTI shows that uncertainty estimation is an active area, but it does
not solve the kinase-specific benchmark problem by itself. A 2025
modification-aware DAVIS benchmark further highlights that even standard kinase
datasets still admit more realistic problem formulations. KinBench-UQ is aimed
at the overlap of those concerns: kinase realism, calibrated uncertainty, and
budgeted prioritization.

This paper positions KinBench-UQ as a benchmark and evaluation contribution
with a stronger interaction-aware baseline rather than as a pure architecture
paper. Our thesis is:

> **Current kinase affinity benchmarks are too optimistic and not
> decision-oriented; realistic evaluation should jointly measure predictive
> performance, calibration, and compound prioritization under assay budgets.**

## 2. Related Work

### 2.1 Drug-Target Affinity Prediction

DeepDTA established the modern Davis/KIBA deep-learning baseline by learning
from protein sequences and SMILES strings directly. GraphDTA later replaced the
ligand string encoder with molecular graphs and became a widely reused
comparison point. A 2024 review in *Frontiers in Pharmacology* summarizes the
field and makes clear that standard DTA prediction is already heavily studied.

### 2.2 Realistic Evaluation and Benchmarking

Recent work increasingly questions default DTA evaluation. The 2025 preprint
*DTA Models’ Performance Under Similarity-Aware Splits* argues that random or
insufficiently strict splits can overestimate generalization. The 2025 preprint
*Advancing Kinase Inhibitor Benchmarking via a Modification-Aware DAVIS
Benchmark Dataset* argues for more realistic kinase formulations that account
for sequence modifications and mutant structure.

### 2.3 Uncertainty Estimation

Uncertainty estimation in drug-target modeling is also established rather than
novel by itself. Rakhshaninejad et al. (2025) study conformal prediction for
drug-target interaction prediction and show that calibrated uncertainty can be
made operational. This motivates uncertainty as a benchmark dimension, but not
as a sufficient research contribution on its own.

## 3. KinBench-UQ Benchmark Design

### 3.1 Principles

KinBench-UQ is built around four principles:

1. **Assay-aware data semantics**. `Kd`, `Ki`, and `IC50` should not be merged
   silently under a single label.
2. **Realistic generalization**. The default split should prevent target
   leakage.
3. **Decision-oriented evaluation**. Benchmarks should answer budgeted
   selection questions, not only regression questions.
4. **Calibration-aware reporting**. A model with uncertainty must show whether
   that uncertainty is actually informative.

### 3.2 Current Implemented Benchmark

The current repository implements the first KinBench-UQ slice on the Davis
dataset together with multiple split families:

- dataset: Davis kinase panel
- targets: 442 kinase targets
- ligands: 68 molecules
- interactions: 30,056
- assay family: `Kd`
- splits: random, cold-target, cold-ligand, scaffold, and both-new
- mutation-aware split: mutation-holdout evaluation from wild-type to variant targets

The pipeline preserves:

- `target_id`
- `target_sequence`
- `affinity_type`
- `affinity_nm`
- `p_activity`

### 3.3 Planned Benchmark Extensions

The next benchmark extensions should include:

- target-sequence-identity-aware splits
- mutation-aware splits
- external validation across sources

## 4. Methods

### 4.1 Baseline Model

The repository currently supports three comparison models:

1. **Ligand-only ridge ensemble**
2. **Ligand-plus-target ridge ensemble**
3. **Interaction-aware projected cross-feature ensemble** with calibrated
   uncertainty (`dual_tower_uq`)

The third model is designed to be more expressive while remaining fast enough
to run across benchmark variants on CPU. It augments ligand and target features
with projected interaction features before fitting an ensemble regressor.

### 4.2 Calibration

Bootstrap spread alone was under-dispersed on validation, so we apply a scalar
uncertainty calibration step to match nominal interval coverage. This makes
confidence intervals meaningful enough to evaluate rather than decorative.

### 4.3 Decision Policies

We evaluate three prioritization policies:

1. **Mean-only ranking**
   Score = predicted `p_activity`
2. **Probability-of-activity ranking**
   Score = calibrated probability that `p_activity >= 6.0`
3. **Risk-adjusted ranking**
   Score = predicted `p_activity - lambda * prediction_std`, with `lambda`
   selected on validation

## 5. Evaluation Protocol

### 5.1 Predictive Metrics

- RMSE
- global Spearman correlation
- mean per-target Spearman
- ROC-AUC at `p_activity >= 6.0`
- top-10% enrichment

### 5.2 Budget-Constrained Metrics

For assay budgets of 1, 3, 5, and 10 compounds per target:

- hit rate at budget
- mean selected `p_activity`
- regret relative to oracle top-k selection

### 5.3 Uncertainty Diagnostics

- calibrated 95% interval coverage
- mean predictive standard deviation
- Spearman correlation between predictive standard deviation and absolute error
- split-conformal and normalized-conformal interval quality

## 6. Current Results

### 6.1 Split Family Effects

The ridge-family baselines already show that split family matters materially.

For the ridge ensemble:

- random split: RMSE `0.768`, Spearman `0.486`
- cold-target split: RMSE `0.730`, Spearman `0.456`
- cold-ligand split: RMSE `0.825`, Spearman `0.290`
- scaffold split: RMSE `0.814`, Spearman `0.209`
- both-new split: RMSE `0.812`, Spearman `0.292`

This reveals an important benchmark insight: on Davis, target-held-out is not
automatically the harshest evaluation. Ligand-generalization and both-new
settings are substantially harder for the ridge baselines.

### 6.1b Mutation-Transfer Split

We additionally construct a mutation-holdout benchmark in which wild-type target
families remain available for training while mutant variants are reserved for
validation and test. This setting is scientifically meaningful for kinases
because clinically relevant resistance variants are common and often drive
screening decisions.

### 6.2 Stronger Interaction-Aware Model

The interaction-aware `dual_tower_uq` model improves substantially on the two
evaluated splits so far:

- random split: RMSE `0.596`, Spearman `0.634`, ROC-AUC `0.908`
- cold-target split: RMSE `0.597`, Spearman `0.587`, ROC-AUC `0.899`

Relative to the best ridge-family comparison:

- random RMSE improves from `0.768` to `0.596`
- cold-target RMSE improves from `0.730` to `0.597`

### 6.3 Predictive Performance on the Original Target-Held-Out Slice

On the original target-held-out Davis workflow, the ridge baseline achieves:

- RMSE: `0.778949`
- global Spearman: `0.453520`
- ROC-AUC: `0.796891`
- mean per-target Spearman: `0.465384`
- mean top-10% enrichment: `3.423712`
- calibrated 95% interval coverage: `0.952489`

### 6.4 Decision-Oriented Results

For budgets of 1, 3, 5, and 10 compounds per target, mean-only ranking is the
strongest currently implemented policy on the test set:

- budget-1 hit rate: `0.784615`
- budget-3 hit rate: `0.646154`
- budget-5 hit rate: `0.600000`
- budget-10 hit rate: `0.506154`

Probability-of-activity ranking and the current validation-tuned risk-adjusted
ranking do not improve hit rate on this split. This is a meaningful result:
calibrated uncertainty does **not** automatically translate into better
selection policy.

### 6.4b Conformal Evaluation

Conformal evaluation makes the uncertainty story sharper. On the original
baseline workflow, normalized conformal intervals achieve:

- `alpha = 0.10`: coverage `0.895`, mean width `2.032`
- `alpha = 0.05`: coverage `0.952`, mean width `3.289`

This makes it possible to report explicit abstention and decision regions rather
than only raw predictive standard deviations.

### 6.5 Why Uncertainty Still Matters

Even though the simple uncertainty-aware policies do not yet outperform mean
ranking, the uncertainty estimates are informative:

- Spearman correlation between `prediction_std` and absolute error: `0.431368`
- lowest-uncertainty quartile mean absolute error: `0.278880`
- highest-uncertainty quartile mean absolute error: `0.777631`

This supports KinBench-UQ’s claim that calibration should be evaluated as its
own dimension and should inform abstention, triage, and scientist trust.

## 7. Discussion

This draft does **not** claim direct superiority over DeepDTA, GraphDTA, or
other published methods yet. That comparison would be scientifically weak unless
those baselines are rerun under the same KinBench-UQ protocol. Published Davis
numbers from standard literature are not directly comparable to our realistic
benchmark framing.

Instead, the current contribution is:

- a cleaned kinase-focused benchmark with multiple split families
- an assay-aware data representation
- a decision-oriented evaluation protocol
- comparison-ready baseline models
- an uncertainty-calibrated interaction-aware model that improves on the ridge
  baselines in the currently run settings
- a mutation-transfer benchmark that is much harder for the ridge baselines and
  more relevant to kinase screening than generic random splits
- an honest demonstration that uncertainty is informative, but that naive
  uncertainty-aware policies do not automatically outperform mean ranking

That honesty is a strength rather than a weakness. If future models improve
budgeted prioritization while remaining calibrated, KinBench-UQ will make that
gain measurable.

## 8. Limitations and Next Steps

The current draft has four immediate limitations:

1. Literature baselines have not yet been rerun under the KinBench-UQ protocol.
2. Sequence-identity-aware and mutation-aware splits are not yet implemented.
3. The current models use sequence-derived features rather than pretrained
   protein embeddings.
4. The current application study is retrospective rather than prospective.

Next steps are therefore:

- implement similarity-aware and mutation-aware splits
- rerun DeepDTA-like, GraphDTA-like, and stronger modern baselines under the
  same protocol
- test richer decision policies and selective prediction
- expand to external validation and scientist-facing case studies

## References

1. Davis MI, Hunt JP, Herrgard S, et al. Comprehensive analysis of kinase
   inhibitor selectivity. *Nature Biotechnology* (2011).
2. Ozturk H, Ozkirimli E, Ozgur A. DeepDTA: deep drug-target binding affinity
   prediction. *Bioinformatics* (2018). [Link](https://doi.org/10.1093/bioinformatics/bty593)
3. Nguyen T, Le H, Quinn T, et al. GraphDTA: predicting drug-target binding
   affinity with graph neural networks. *Bioinformatics* (2021).
   [Link](https://doi.org/10.1093/bioinformatics/btaa921)
4. Du Y, Wang C, Wang Z, et al. Recent advances in deep learning methods for
   drug-target affinity prediction. *Frontiers in Pharmacology* (2024).
   [Link](https://www.frontiersin.org/articles/10.3389/fphar.2024.1375522/full)
5. Su Y, Liu Y, Zhang X, et al. DTA Models’ Performance Under
   Similarity-Aware Splits: Challenging Assumptions in Predicting Drug-Target
   Binding Affinity. arXiv (2025).
   [Link](https://arxiv.org/abs/2504.09481)
6. Rakhshaninejad M, Evers A, de Ruiter A, et al. Conformal Prediction for
   Uncertainty Estimation in Drug-Target Interaction Prediction. *PMLR* (2025).
   [Link](https://proceedings.mlr.press/v266/rakhshaninejad25a.html)
7. Advancing Kinase Inhibitor Benchmarking via a Modification-Aware DAVIS
   Benchmark Dataset. arXiv (2025). [Link](https://arxiv.org/abs/2512.00708)
