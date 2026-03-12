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

There are a few baseline models in the repo.

The most important one from our own repo is:

- `dual_tower_uq`

The most important outside comparison we already finished is:

- `GraphDTA`

There is also another older comparison model called:

- `DeepDTA`

At the time of this note, `DeepDTA` has not yet been fully finished in the final benchmark.

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
