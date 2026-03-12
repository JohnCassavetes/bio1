# Super Simple Project Explanation

## What are we trying to do?

We are trying to help answer a simple question:

> If you have a kinase you care about, which small molecules should you test first?

The goal is to make a ranked list, so the best candidates are near the top.

This is useful because in real lab work, people usually cannot test everything. They need a smarter shortlist.

## What is a kinase?

A kinase is just a kind of protein in the body.

Many diseases involve kinases, so scientists often want to find molecules that bind to them well.

## What is this repo doing?

This repo is building a system that:

- looks at a molecule
- looks at a kinase
- guesses how strongly they might bind
- ranks the molecules from most promising to least promising

So instead of saying "this is definitely a drug," it says something more realistic:

> "These are the compounds most worth testing first."

## What makes our project better than a basic version?

A basic project might only ask:

> Can we get a decent score on one dataset?

Our project is trying to ask a better question:

> Does the system still work when the test is harder and more realistic?

That matters because some models look good only when the test is too easy.

## What did we add?

We made the project much more serious.

Now it:

- tests harder situations, not just easy ones
- checks whether the model is secretly getting help from things that look too similar to the training data
- tests how well it works on mutation-related cases
- tests it on outside data, not just the original dataset
- automatically creates tables and figures from the saved results

That makes it much more believable as a paper project.

## What are the main steps?

### Step 1: Download the data

We first download the starting data.

This is just collecting the raw information about molecules, kinases, and measured binding values.

### Step 2: Clean the data

Then we clean it up so everything follows one format.

This means:

- keeping only usable entries
- putting the values into a consistent form
- saving everything in organized files

### Step 3: Make easier and harder tests

Then we create different test setups.

Some are easier.
Some are much harder and more realistic.

This matters because if you only use an easy test, the model can seem better than it really is.

### Step 4: Run different models

Then we compare several prediction systems.

Some are simple.
Some are stronger.
Some are based on well-known past papers.

This helps show whether our method is actually good, instead of only looking good because we chose a weak comparison.

### Step 5: Make the paper outputs

After the runs finish, the repo turns the results into:

- summary tables
- charts
- paper-ready result files

So the paper is based on actual outputs, not hand-typed numbers.

## What models do we have right now?

There are a few models in the repo.

A model is just the decision-making system that looks at the molecule and the kinase and tries to guess how strong the match is.

You can think of each model as a different kind of ranking engine.

Some are simple.
Some are more advanced.
Some come from past research papers and are used as known comparison points.

So yes:

- `GraphDTA` is a model
- `DeepDTA` is a model
- `dual_tower_uq` is our main model in this repo

The most important one from our own repo is:

- `dual_tower_uq`

The most important outside comparison we already finished is:

- `GraphDTA`

There is also another older comparison model called:

- `DeepDTA`

At the time of this note, `DeepDTA` has not yet been fully finished in the final benchmark.

## What do those model names mean in simple words?

### `ligand_only_ridge`

This is a simple baseline.

It mostly asks:

> If I only look at the molecule, can I still make a useful guess?

This is important because if a fancy model cannot beat a simple one, that is not impressive.

### `ridge_ensemble`

This is a stronger but still fairly standard baseline.

It looks at both the molecule and the kinase in a more traditional machine-learning way.

You can think of it as a solid classical comparison model.

### `dual_tower_uq`

This is our main model.

It has one part that looks at the molecule and one part that looks at the kinase, then combines those two views to make a prediction.

The `UQ` part means it also tries to express uncertainty, not just a single score.

So in plain words:

> Our model tries to judge both sides of the match, not just one side.

### `GraphDTA`

This is a known model from earlier research.

It is important because it is a real outside comparison, not something we invented just to make ourselves look good.

It is one of the better-known baseline models in this area.

### `DeepDTA`

This is another well-known older comparison model from earlier research.

It is useful mostly because reviewers recognize it and expect to see it as a comparison point.

## What is our model, exactly?

Our main model is `dual_tower_uq`.

That is the model we are most interested in.

The other models are there so we can answer this question honestly:

> Is our model actually strong, or does it only look good because we did not compare it to anything real?

So the point of `GraphDTA` and `DeepDTA` is to give a fair outside comparison.

## How should a normal person think about these models?

An easy way to think about it is:

- the molecule is one side of the match
- the kinase is the other side of the match
- the model is trying to decide how good the match is

Then it puts molecules in order from "most promising" to "least promising."

So this is closer to a recommendation system than a magical discovery machine.

It is like:

> given one target, recommend the molecules most worth testing first

## What did we find so far?

The short version is:

> Our main model still looks strong, even when the tests get harder.

That is important.

It means the model is not only doing well on an easy setup. It is still competitive when we make the task more realistic.

## How did `GraphDTA` do?

`GraphDTA` is a real, well-known comparison model from past research.

That makes it a meaningful test.

The result was:

- `GraphDTA` did fine
- but our main model still looked better overall on the harder tests

That is a good sign for the paper.

## What is the proof?

The proof is that the repo saved the actual outputs from the runs.

We are not just writing numbers into a document by hand. The repo contains:

- the saved predictions for each run
- the saved metric files for each run
- the combined benchmark summary
- the generated figures and tables

So when we say a model did well or badly, that claim comes from files that were created by the code.

## Where can someone check the proof?

The easiest places to check are:

- `results/benchmark/summary.csv`
- `results/benchmark/summary.json`
- `results/benchmark/random/graphdta_gcn_exact/test_predictions.csv`
- `results/benchmark/cold_target/graphdta_gcn_exact/test_predictions.csv`
- `results/benchmark/sequence_identity/graphdta_gcn_exact/test_predictions.csv`
- `results/benchmark/mutation_holdout/graphdta_gcn_exact/test_predictions.csv`
- `results/benchmark_analysis/benchmark_overview.csv`
- `paper/generated_results.md`

These files show the runs that actually happened.

## What do the current results say?

Before `DeepDTA`, the main story is:

- on the easy `random` split, `GraphDTA` is very competitive
- on the harder splits, our main model still looks better overall

## How do you read the results?

The results are answering:

> Which model makes better guesses?

The easiest way to read them is:

- lower `RMSE` is better
- higher `Spearman` is better

You do not need to know the math deeply.

Very roughly:

- `RMSE` means "how far off were the guesses?"
- `Spearman` means "did the model put the compounds in roughly the right order?"

That second one matters a lot here, because the real goal is ranking candidates.

So if two models are close on `RMSE`, but one is better on `Spearman`, that can still matter a lot for prioritizing what to test first.

Some key numbers:

- `random`: `GraphDTA` RMSE `0.591`, `dual_tower_uq` RMSE `0.596`
- `cold_target`: `GraphDTA` RMSE `0.690`, `dual_tower_uq` RMSE `0.597`
- `sequence_identity`: `GraphDTA` RMSE `0.780`, `dual_tower_uq` RMSE `0.701`
- `mutation_holdout`: `GraphDTA` RMSE `0.736`, `dual_tower_uq` RMSE `0.545`

In simple words:

- `GraphDTA` is not weak
- but our main model still looks stronger when the task gets more realistic

## What do the split names mean?

These names describe how hard the test is.

### `random`

This is the easier test.

It is closer to mixing everything up and holding some part out.

Models often look better here.

### `cold_target`

This means the test includes kinase targets that the model did not get to train on.

That is harder and more realistic.

### `sequence_identity`

This is an even stricter version.

It tries to stop the model from getting an unfair advantage from targets that are too similar to ones it already saw.

So if a model still does well here, that is a stronger sign.

### `mutation_holdout`

This checks whether the model still works on mutation-related cases.

That matters because real biology often includes important variants, not just a single clean target version.

## How should someone interpret the current numbers?

The simple interpretation is:

- if a model only looks good on `random`, that is less impressive
- if it stays strong on `cold_target`, `sequence_identity`, and `mutation_holdout`, that is more impressive

That is why our current result is encouraging.

`GraphDTA` is competitive on the easier test, but our main model still holds up better on the tougher tests.

That is a stronger result than just winning on an easy split.

## How did we get those results?

Very simply:

1. We downloaded and cleaned the kinase-ligand data.
2. We built several test setups, including harder ones.
3. We ran multiple models on the same setups.
4. We saved the predictions and metrics to disk.
5. We generated the tables and figures from those saved result files.

That means the paper-facing outputs are tied to the experiment outputs.

## Why are we not lying about the results?

Because the evidence is saved in the repo and can be checked.

The important protection against "making things up" is:

- the numbers come from run outputs
- the paper summary is generated from those outputs
- the plots are generated from those outputs
- the benchmark compares multiple models, not only our own

So if a number changes, it changes because the code was rerun and the result files changed.

That does not mean the project is perfect.
It just means the claims are tied to actual artifacts, not hand-written storytelling.

## What is the main message right now?

The main message is:

> We built a more realistic and reproducible way to test kinase ranking systems, and our main method performs well in that stronger setup.

So the value is not only the model.

The value is also the benchmark itself:

- better tests
- fairer comparisons
- clearer outputs
- easier reproduction

## Is this publishable?

Yes, it is much closer now.

Before, it was more like a solid project.
Now, it looks much more like a real paper candidate.

The main missing piece is that `DeepDTA` still needs to be fully run if we want the strongest comparison package.

So the honest version is:

> It is close to paper-ready, but one major comparison is still unfinished.

## What is left to do?

The biggest remaining task is:

- finish the `DeepDTA` comparison run

After that:

- add those results into the main benchmark
- regenerate the figures and tables
- make sure the paper text matches the final result files

## Very short summary

We are building a tool that helps rank which molecules should be tested first for kinase targets.

We improved it by making the tests more realistic, adding outside validation, and comparing against a strong known baseline.

So far, the results say our main model still looks strong, especially on the harder tests, and the project is much closer to being publishable.
