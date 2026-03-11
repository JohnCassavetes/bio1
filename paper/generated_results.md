# Generated Benchmark Evidence

All tables below are generated from result files under `results/`.

## Primary Tables

split,model,rmse,spearman,roc_auc,mean_per_target_spearman
cold_target,dual_tower_uq,0.597,0.587,0.899,0.571
mutation_holdout,dual_tower_uq,0.545,0.818,0.971,0.714
random,dual_tower_uq,0.596,0.634,0.908,0.617
sequence_identity,dual_tower_uq,0.701,0.461,0.809,0.493

split,dual_tower_uq,ridge_ensemble
cold_target,0.597,0.73
mutation_holdout,0.545,1.158
random,0.596,0.768
sequence_identity,0.701,0.714

split,dual_tower_uq,ridge_ensemble
cold_target,0.587,0.456
mutation_holdout,0.818,0.463
random,0.634,0.486
sequence_identity,0.461,0.43

split,test_targets,mean_nearest_train_identity,median_nearest_train_identity,max_nearest_train_identity
both_new,65,0.571,0.532,1.0
cold_ligand,442,1.0,1.0,1.0
cold_target,65,0.571,0.532,1.0
mutation_holdout,26,1.0,1.0,1.0
random,442,1.0,1.0,1.0
scaffold,442,1.0,1.0,1.0
sequence_identity,59,0.367,0.391,0.581

model,mutation_family,rows,targets,variant_targets,wildtype_train_rows,wildtype_train_targets,rmse,spearman,variant_mean_p_activity,wildtype_train_mean_p_activity,variant_minus_wildtype_p_activity,mean_nearest_train_identity
dual_tower_uq,PIK3CA,612,9,9,68,1,0.15,0.458,5.173,5.179,-0.005,1.0
dual_tower_uq,LRRK2,68,1,1,68,1,0.334,0.819,5.567,5.544,0.023,1.0
dual_tower_uq,BRAF,68,1,1,68,1,0.389,0.75,5.417,5.359,0.058,1.0
dual_tower_uq,MET,136,2,2,68,1,0.411,0.834,5.583,5.59,-0.007,1.0
dual_tower_uq,FLT3,408,6,6,68,1,0.52,0.903,6.23,6.358,-0.128,1.0
dual_tower_uq,KIT,476,7,7,68,1,0.871,0.817,6.257,6.563,-0.306,1.0
ligand_only_ridge,LRRK2,68,1,1,68,1,0.742,0.679,5.567,5.544,0.023,1.0
ligand_only_ridge,PIK3CA,612,9,9,68,1,0.851,0.017,5.173,5.179,-0.005,1.0
ligand_only_ridge,MET,136,2,2,68,1,0.942,0.566,5.583,5.59,-0.007,1.0
ligand_only_ridge,BRAF,68,1,1,68,1,0.949,0.071,5.417,5.359,0.058,1.0
ligand_only_ridge,FLT3,408,6,6,68,1,1.442,0.708,6.23,6.358,-0.128,1.0
ligand_only_ridge,KIT,476,7,7,68,1,1.485,0.56,6.257,6.563,-0.306,1.0

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
