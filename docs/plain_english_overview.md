# KinBench-UQ In Plain English

## What We Are Trying To Do

We are trying to build a fair test for models that rank small molecules for
kinase targets.

In normal language:

- a **kinase target** is the protein we care about
- a **small molecule ligand** is the chemical we might test against it
- the model tries to guess **how strongly the molecule will bind**
- then we use that guess to **rank which molecules should be tested first**

So this is not mainly about claiming drug discovery success. It is about
**candidate prioritization**:

> If a scientist only has budget to test a few compounds, can the model help
> choose the better ones first?

## What The Repo Does

The repo takes raw binding data, cleans it, trains models, and evaluates them
under different testing setups.

The important idea is that not all test setups are equally honest.

Some test splits are too easy because the model may still see targets in testing
that are very similar to targets it already saw during training.

So the repo now checks:

- normal random splits
- harder target-held-out splits
- harder ligand-held-out splits
- scaffold and both-new splits
- exact sequence-identity-aware target splits
- mutation-family splits
- external validation on a different source dataset

## What We Found

The main finding is:

**the benchmark setup changes the story a lot.**

A model can look good on an easier split and noticeably worse on a more honest
split.

We also found:

- sequence similarity leakage matters
- mutation transfer is a real and distinct challenge
- external validation is much harder than in-domain Davis testing
- the interaction-aware repo model (`dual_tower_uq`) is currently the strongest
  of the regenerated repo-native models on the key splits we ran

In short:

> the project is no longer just “train one model on Davis.”
> it is now a more realistic evaluation system for kinase-ligand ranking.

## A CS-Friendly Version

If you are from CS, the easiest mental model is:

- this repo is a **benchmark harness**
- the task is **ranking / regression for protein-ligand pairs**
- the real contribution is **evaluation protocol design**
- the main bug in older-style evaluation is **distribution leakage**

So the repo now asks:

1. Are train and test really separated in a meaningful way?
2. Does the model still work on variants and unseen target families?
3. Does it still work on outside data?
4. Are the results generated reproducibly, not hand-written into the paper?

That is why this repo matters more as a **measurement system** than as a single
novel model.

## A Totally Non-Technical Version

Imagine you trained a movie recommender by letting it study almost the same
users and movies in both practice mode and exam mode. The exam would look easier
than real life.

That is the problem we are trying to avoid here.

We want to test these molecule-ranking models in a way that is closer to the
real question:

> “If I give you a new kinase problem, can you still help me decide which
> compounds to test first?”

## Is It Publishable?

Yes, the project is now much more publishable than before because:

- it has a stronger benchmark story
- it has real external validation
- it has leakage analysis
- it has generated figures and tables
- it has a reproducible pipeline

But one major step is still needed for the strongest paper claim:

- finish the full exact reruns of the literature baselines
  (`deepdta_exact` and `graphdta_gcn_exact`) across all claimed splits

So the honest summary is:

> publishable direction: yes
> already fully finished top-tier benchmark package: not quite yet

## What The Important Scripts Do

### Data

- `scripts/download_data.py`
  Downloads raw data.
  Supports Davis and BindingDB.

- `scripts/process_dataset.py`
  Cleans raw data and builds the standard processed dataset used by the repo.

- `scripts/build_external_validation.py`
  Builds the processed external validation set from BindingDB.

### Splits And Benchmarking

- `scripts/generate_benchmark_splits.py`
  Creates the benchmark split folders.
  This now includes the exact sequence-identity split.

- `scripts/run_benchmark_suite.py`
  Runs benchmark models across the requested split families.

- `scripts/run_external_validation.py`
  Trains on Davis and tests on the external BindingDB set.

### Baseline Workflow

- `scripts/train_baseline.py`
  Trains the main target-aware ridge baseline.

- `scripts/predict_rank.py`
  Scores new molecule-target pairs and ranks them.

### Decision / Uncertainty

- `scripts/evaluate_budgeted_policies.py`
  Checks how useful the model is when you only get to test a few compounds.

- `scripts/evaluate_conformal.py`
  Evaluates uncertainty intervals and abstention-style diagnostics.

### Paper / Reporting

- `scripts/generate_paper_assets.py`
  Generates analysis CSVs, figures, and the manuscript evidence fragment from
  result files.

- `scripts/run_submission_pipeline.py`
  Runs the full paper pipeline end to end.

## The Main Output Folders

- `data/benchmark/`
  Benchmark split files.

- `data/external/bindingdb/`
  Processed external validation data.

- `results/benchmark/`
  Main benchmark outputs.

- `results/external_validation/bindingdb/`
  External validation outputs.

- `results/benchmark_analysis/`
  Generated analysis tables.

- `figures/`
  Generated paper figures.

- `paper/generated_results.md`
  Generated manuscript evidence summary.

## If You Only Want The Shortest Possible Summary

We built a system for testing kinase-ligand ranking models more honestly.

Instead of only asking “does the model fit Davis well?”, we now ask:

- does it still work on harder splits?
- does it still work on mutant families?
- does it still work on outside data?
- are the paper figures/tables produced automatically from results?

That is the real upgrade.
