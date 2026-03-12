# KinBench-UQ in Simple Language

## What problem are we trying to solve?

We want to help answer a practical question:

> If a scientist cares about a kinase, which small molecules should they test first?

That is the real goal of this project. We are not trying to claim that the computer can invent a drug on its own. We are trying to help make a smarter shortlist.

## Why does this matter?

In real lab work, time and money are limited. Scientists cannot test every possible molecule. A ranking system that puts better candidates near the top can save effort and help focus experiments.

## The core problem

Scientists are trying to find drugs that work on kinases, which are proteins
involved in diseases such as cancer and Parkinson's. They use AI models to
predict which drug candidates are worth testing in the lab.

The problem is that many existing tests for these AI models are too easy. That
can make the models look better than they really are in real-life settings.

## A simple analogy

Imagine you are training a student to recognize dogs they have never seen
before. But the test keeps showing them dogs that look almost identical to the
ones they already studied. The student scores very highly, but then struggles
when shown a truly unfamiliar breed.

That would be a misleading test.

That is close to what happens in kinase AI benchmarks when proteins that are
too similar to the training set still appear in testing. The model can look
smarter than it really is.

## What does this project do?

The system looks at two things:

- a small molecule
- a kinase target

It then guesses how strongly they might bind and uses that guess to rank the molecules from most promising to least promising.

## What did we think was missing in older work?

A lot of older work was mainly about getting good scores on standard datasets. That is useful, but it can be too optimistic.

Sometimes a model looks good because:

- the test is too easy
- the test targets are too similar to the training targets
- the paper only reports one kind of split
- the results are hard to reproduce

We wanted a benchmark that asks a harder and more realistic question:

> Does a model still look good when the test is stricter and closer to real candidate prioritization?

## What KinBench does differently

KinBench is mainly a stricter test.

It tries to make sure the model is being tested on proteins that are not just
near-copies of proteins it already saw during training. It also tests mutant
kinases, which matters because drug resistance often happens through mutations.

So the main point is not:

> "we built the best AI for drug discovery"

The main point is closer to:

> "we built a better exam for grading AI models used in kinase drug discovery"

And when we use that harder exam, the rankings between models change.

## What did we build?

We built a benchmark called `KinBench-UQ`.

The name means:

- `Kin` = kinase
- `Bench` = benchmark
- `UQ` = uncertainty quantification

The benchmark does more than just run one model on one dataset. It:

- cleans and standardizes the data
- creates several kinds of train/test splits
- compares multiple models under the same setup
- checks whether train and test targets are too similar
- evaluates mutation-related cases
- tests on outside data
- generates paper tables and figures directly from saved result files

## What models did we compare?

We compared several prediction systems.

The most important ones are:

- `dual_tower_uq`, which is our main repo model
- `GraphDTA`, a known published comparison model
- `DeepDTA`, another known published comparison model

This matters because it means we are not only comparing ourselves against weak toy baselines.

## What makes our main model different?

Our main model, `dual_tower_uq`, tries to look at both sides of the problem:

- the molecule
- the kinase

It then combines those views to make a prediction. It also tries to report uncertainty, not just a single score.

In simple terms, it is trying to act like a smarter matching and ranking system.

## How did we test the models?

We did not use only one easy test.

We used several kinds of evaluation:

- `random`: an easier split
- `cold_target`: the model must handle targets it did not train on
- `sequence_identity`: the model is prevented from getting easy help from very similar targets
- `mutation_holdout`: the model is tested on mutation-related cases
- external validation: train on Davis, test on outside kinase data from BindingDB

This is important because a model that only looks good on an easy split is less convincing.

## What do the scores mean?

The two easiest ones to understand are:

- `RMSE`: lower is better; it means the guesses were closer to the true values
- `Spearman`: higher is better; it means the model ranked the compounds in a better order

That second one matters a lot here because the whole point is candidate ranking.

## What did we find?

The results are not a simple story of "our model wins everything."

That is actually a good sign, because it makes the benchmark more believable.

The current results say:

- `DeepDTA` is strongest on the easier `random` split
- `DeepDTA` is also slightly best on `sequence_identity` by one error measure
- `dual_tower_uq` is best on `cold_target`
- `dual_tower_uq` is clearly best on `mutation_holdout`
- `GraphDTA` is competitive, but weaker than the top two models on the main harder comparisons

## What are the key numbers?

Across the four main benchmark splits:

- `random`: `DeepDTA` RMSE `0.548`, `dual_tower_uq` RMSE `0.596`, `GraphDTA` RMSE `0.591`
- `cold_target`: `dual_tower_uq` RMSE `0.597`, `DeepDTA` RMSE `0.614`, `GraphDTA` RMSE `0.690`
- `sequence_identity`: `DeepDTA` RMSE `0.681`, `dual_tower_uq` RMSE `0.701`, `GraphDTA` RMSE `0.780`
- `mutation_holdout`: `dual_tower_uq` RMSE `0.545`, `GraphDTA` RMSE `0.736`, `DeepDTA` RMSE `0.764`

Looking at ranking quality (`Spearman`):

- `random`: `DeepDTA` `0.658`, `dual_tower_uq` `0.634`
- `cold_target`: `dual_tower_uq` `0.587`, `DeepDTA` `0.563`
- `sequence_identity`: `dual_tower_uq` `0.461`, `DeepDTA` `0.446`
- `mutation_holdout`: `dual_tower_uq` `0.818`, `GraphDTA` `0.749`, `DeepDTA` `0.739`

So the simple interpretation is:

- `DeepDTA` is very strong on the easier or more standard settings
- `dual_tower_uq` stays stronger on some of the harder and more realistic settings

## What did we learn from the stricter split?

The `sequence_identity` split is important because it tries to prevent the model from getting easy help from targets that are too similar to what it already saw.

The leakage summary shows:

- most of the easier splits still contain exact or near-exact target neighbors
- the `sequence_identity` split reduces the mean nearest-train identity to about `0.367`
- its maximum nearest-train identity is about `0.581`

That means it is doing a better job of testing real generalization.

## What did we learn from mutation analysis?

Mutation-related performance is not uniform across families.

Our main model did especially well on several important mutation families, including:

- `PIK3CA`
- `LRRK2`
- `BRAF`
- `MET`
- `FLT3`

This matters because real kinase work often involves important variants, not only one clean target sequence.

## What about outside data?

We also tested on an external BindingDB-based kinase set after training on Davis.

That is a harder and more realistic check than staying inside Davis.

On this outside test:

- `dual_tower_uq` reached RMSE `1.484`
- `ridge_ensemble` reached RMSE `1.602`
- `ligand_only_ridge` reached RMSE `1.641`

The external scores are worse than the in-domain ones, which is expected. But they are also more honest about how hard generalization really is.

## Why should someone trust these claims?

Because the numbers are tied to saved outputs in the repo.

The project stores:

- per-run prediction files
- per-run metric files
- merged benchmark summaries
- generated analysis tables
- generated figures

The manuscript fragment is generated from those artifacts, rather than typed by hand.

So if someone wants to check the claims, they can inspect:

- `results/benchmark/summary.csv`
- `results/benchmark/summary.json`
- `results/benchmark_analysis/benchmark_overview.csv`
- `results/benchmark_analysis/leakage_summary.csv`
- `results/benchmark_analysis/mutation_family_analysis.csv`
- `results/external_validation/bindingdb/summary.csv`
- `paper/generated_results.md`

## What is the honest conclusion right now?

The project is now much stronger than a simple benchmark demo.

It has:

- harder and more realistic tests
- recognizable literature baselines
- external validation
- mutation-family analysis
- reproducible paper artifacts

The current evidence does not say "our model is best at everything."

The stronger and more honest claim is:

> We built a more realistic and reproducible kinase candidate-prioritization benchmark, and our main model remains especially strong on some of the harder evaluation settings.

That is a better paper story than winning only on an easy split.

## Why this matters in the real world

If a drug company or research lab uses only an easy benchmark, it can end up
trusting an AI model too much.

KinBench helps give a more honest picture of whether a model is likely to hold
up when the task is genuinely difficult. That matters before spending large
amounts of time and money on lab experiments.
