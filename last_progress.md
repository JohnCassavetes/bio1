We turned the repo from “a solid baseline project” into a much more serious benchmark system for kinase-ligand candidate prioritization.

The repo now has:

- exact sequence-identity-aware splits based on real alignment identity, not a proxy
- mutation-family split analysis and leakage summaries
- Davis-to-BindingDB external validation
- generated figures, tables, and manuscript evidence pulled from `results/*`
- a reproducible environment and one-command submission pipeline

The benchmark story is now much stronger because it tests harder and more realistic conditions instead of only an easier in-distribution setup.

Current benchmark status:

- repo-native baselines are complete
- `graphdta_gcn_exact` has been fully run on the four main paper splits: `random`, `cold_target`, `sequence_identity`, and `mutation_holdout`
- `GraphDTA` has been merged into `results/benchmark`, and the paper assets were regenerated from the merged benchmark outputs
- the current generated evidence is in `results/benchmark/summary.csv`, `results/benchmark_analysis/*`, `figures/*`, and `paper/generated_results.md`

Current result takeaway:

- `GraphDTA` is a legitimate literature-style comparison baseline and is competitive
- the repo-native `dual_tower_uq` model still performs better overall on the harder splits, especially `cold_target`, `sequence_identity`, and `mutation_holdout`
- `GraphDTA` narrowly edges `dual_tower_uq` on `random` RMSE, but not on the stronger overall benchmark story

What is still missing:

- `deepdta_exact` has not yet been fully run and frozen under the same four-split protocol
- the manuscript should either wait for `DeepDTA` or be edited so it does not claim both literature baselines are already complete

So the project is now meaningfully closer to submission. The main remaining decision is whether to finish `DeepDTA` for the safest comparison package, or narrow the paper claims to the evidence that already exists.
