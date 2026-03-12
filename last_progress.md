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
- `deepdta_exact` has been fully run on the same four main paper splits
- both literature baselines have been merged into `results/benchmark`, and the paper assets were regenerated from the merged benchmark outputs
- the current generated evidence is in `results/benchmark/summary.csv`, `results/benchmark_analysis/*`, `figures/*`, and `paper/generated_results.md`

Current result takeaway:

- both `GraphDTA` and `DeepDTA` are legitimate literature-style comparison baselines and are competitive
- `DeepDTA` is currently strongest on the `random` split and also slightly ahead of `dual_tower_uq` on `sequence_identity` RMSE
- the repo-native `dual_tower_uq` model still performs best on the harder `cold_target` and `mutation_holdout` settings and remains stronger on several ranking-oriented metrics
- the benchmark story is now much stronger because the main claims are backed by completed exact literature reruns rather than only repo-native baselines

What is still missing:

- the manuscript text should be tightened to reflect the full five-model benchmark results, especially the fact that `DeepDTA` is now a strong competitor on `random` and `sequence_identity`
- if desired, external validation could be extended to the literature baselines too, but that is now a strengthening step rather than a core benchmark blocker

So the project is now materially closer to submission. The core benchmark package is in place, and the main remaining work is paper polish plus any optional extra validation runs.
