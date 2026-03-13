# Generated Benchmark Evidence

All tables below are generated from result files under `results/`.

## Primary Tables

split,model,rmse,spearman,roc_auc,mean_per_target_spearman
cold_target,dual_tower_uq,0.597,0.587,0.899,0.571
mutation_holdout,dual_tower_uq,0.545,0.818,0.971,0.714
random,deepdta_exact,0.548,0.658,0.92,0.634
sequence_identity,deepdta_exact,0.681,0.446,0.805,0.446

split,deepdta_exact,dual_tower_uq,graphdta_gcn_exact,ridge_ensemble
cold_target,0.614,0.597,0.69,0.73
mutation_holdout,0.764,0.545,0.736,1.158
random,0.548,0.596,0.591,0.768
sequence_identity,0.681,0.701,0.78,0.714

split,deepdta_exact,dual_tower_uq,graphdta_gcn_exact,ridge_ensemble
cold_target,0.563,0.587,0.478,0.456
mutation_holdout,0.739,0.818,0.749,0.463
random,0.658,0.634,0.61,0.486
sequence_identity,0.446,0.461,0.385,0.43

split,test_targets,mean_nearest_train_identity,median_nearest_train_identity,max_nearest_train_identity
both_new,65,0.571,0.532,1.0
cold_ligand,442,1.0,1.0,1.0
cold_target,65,0.571,0.532,1.0
mutation_holdout,26,1.0,1.0,1.0
random,442,1.0,1.0,1.0
scaffold,442,1.0,1.0,1.0
sequence_identity,59,0.367,0.391,0.581

model,mutation_family,rows,targets,variant_targets,wildtype_train_rows,wildtype_train_targets,rmse,spearman,variant_mean_p_activity,wildtype_train_mean_p_activity,variant_minus_wildtype_p_activity,mean_nearest_train_identity
deepdta_exact,LRRK2,68,1,1,68,1,0.488,0.752,5.567,5.544,0.023,1.0
deepdta_exact,PIK3CA,612,9,9,68,1,0.624,0.428,5.173,5.179,-0.005,1.0
deepdta_exact,FLT3,408,6,6,68,1,0.821,0.776,6.23,6.358,-0.128,1.0
deepdta_exact,BRAF,68,1,1,68,1,0.83,0.357,5.417,5.359,0.058,1.0
deepdta_exact,KIT,476,7,7,68,1,0.854,0.812,6.257,6.563,-0.306,1.0
deepdta_exact,MET,136,2,2,68,1,0.893,0.631,5.583,5.59,-0.007,1.0
dual_tower_uq,PIK3CA,612,9,9,68,1,0.15,0.458,5.173,5.179,-0.005,1.0
dual_tower_uq,LRRK2,68,1,1,68,1,0.334,0.819,5.567,5.544,0.023,1.0
dual_tower_uq,BRAF,68,1,1,68,1,0.389,0.75,5.417,5.359,0.058,1.0
dual_tower_uq,MET,136,2,2,68,1,0.411,0.834,5.583,5.59,-0.007,1.0
dual_tower_uq,FLT3,408,6,6,68,1,0.52,0.903,6.23,6.358,-0.128,1.0
dual_tower_uq,KIT,476,7,7,68,1,0.871,0.817,6.257,6.563,-0.306,1.0

evaluation,model,rows,targets,rmse,spearman,roc_auc,prediction_interval_95_coverage,mean_prediction_std,mean_per_target_spearman,mean_top_10pct_enrichment,mean_per_target_roc_auc
davis_to_external,deepdta_exact,1617,8,1.342,0.37,0.725,,,0.361,1.489,0.775
davis_to_external,dual_tower_uq,1617,8,1.484,0.245,0.602,0.987,3.954,0.342,1.499,0.683
davis_to_external,graphdta_gcn_exact,1617,8,1.538,0.096,0.627,,,0.138,1.353,0.597
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
