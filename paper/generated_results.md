# Generated Benchmark Evidence

All tables below are generated from result files under `results/`.

## Primary Tables

split,model,rmse,spearman,roc_auc,mean_per_target_spearman
cold_target,graphdta_gcn_exact,0.69,0.478,0.829,0.516
mutation_holdout,graphdta_gcn_exact,0.736,0.749,0.942,0.626
random,graphdta_gcn_exact,0.591,0.61,0.892,0.579
sequence_identity,graphdta_gcn_exact,0.78,0.385,0.763,0.424

split,graphdta_gcn_exact
cold_target,0.69
mutation_holdout,0.736
random,0.591
sequence_identity,0.78

split,graphdta_gcn_exact
cold_target,0.478
mutation_holdout,0.749
random,0.61
sequence_identity,0.385

split,test_targets,mean_nearest_train_identity,median_nearest_train_identity,max_nearest_train_identity
both_new,65,0.571,0.532,1.0
cold_ligand,442,1.0,1.0,1.0
cold_target,65,0.571,0.532,1.0
mutation_holdout,26,1.0,1.0,1.0
random,442,1.0,1.0,1.0
scaffold,442,1.0,1.0,1.0
sequence_identity,59,0.367,0.391,0.581

model,mutation_family,rows,targets,variant_targets,wildtype_train_rows,wildtype_train_targets,rmse,spearman,variant_mean_p_activity,wildtype_train_mean_p_activity,variant_minus_wildtype_p_activity,mean_nearest_train_identity
graphdta_gcn_exact,LRRK2,68,1,1,68,1,0.465,0.769,5.567,5.544,0.023,1.0
graphdta_gcn_exact,PIK3CA,612,9,9,68,1,0.598,0.386,5.173,5.179,-0.005,1.0
graphdta_gcn_exact,FLT3,408,6,6,68,1,0.766,0.754,6.23,6.358,-0.128,1.0
graphdta_gcn_exact,BRAF,68,1,1,68,1,0.79,0.565,5.417,5.359,0.058,1.0
graphdta_gcn_exact,KIT,476,7,7,68,1,0.843,0.798,6.257,6.563,-0.306,1.0
graphdta_gcn_exact,MET,136,2,2,68,1,0.873,0.633,5.583,5.59,-0.007,1.0

evaluation,model,rows,targets,rmse,spearman,roc_auc,prediction_interval_95_coverage,mean_prediction_std,mean_per_target_spearman,mean_top_10pct_enrichment,mean_per_target_roc_auc
davis_to_external,dual_tower_uq,1617,8,1.484,0.245,0.602,0.987,3.954,0.342,1.499,0.683
davis_to_external,ridge_ensemble,1617,8,1.602,0.071,0.514,0.685,0.824,0.155,1.37,0.605
davis_to_external,ligand_only_ridge,1617,8,1.641,0.129,0.602,0.566,0.608,0.166,1.366,0.611

## Analysis Artifacts

- [best_by_split.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/best_by_split.csv)
- [benchmark_overview.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/benchmark_overview.csv)
- [leakage_summary.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/leakage_summary.csv)
- [leakage_per_target.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/leakage_per_target.csv)
- [mutation_family_analysis.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/mutation_family_analysis.csv)
- [external_validation_summary.csv](/Users/a/Desktop/bio1/results/benchmark_analysis/external_validation_summary.csv)

## Figures

![Split RMSE](/Users/a/Desktop/bio1/figures/split_rmse_comparison.png)
![Split Spearman](/Users/a/Desktop/bio1/figures/split_spearman_comparison.png)
![Identity Leakage](/Users/a/Desktop/bio1/figures/sequence_identity_leakage.png)
![Mutation Family Delta](/Users/a/Desktop/bio1/figures/mutation_family_delta.png)
![External Validation RMSE](/Users/a/Desktop/bio1/figures/external_validation_rmse.png)
