# Baseline Results

These results were generated locally on March 10, 2026 by running:

```bash
python scripts/download_data.py --source davis
python scripts/process_dataset.py --source davis
python scripts/train_baseline.py
```

## Data Summary

- rows: `30,056`
- ligands: `68`
- targets: `442`
- train rows: `21,080`
- validation rows: `4,556`
- test rows: `4,420`

## Model Configuration

- model: ridge ensemble
- selected alpha: `10.0`
- ensemble size: `8`
- bootstrap fraction: `0.8`
- fingerprint bits: `1024`
- fingerprint radius: `2`
- uncertainty scale: `18.928209`

## Held-Out Test Metrics

- RMSE: `0.778949`
- global Spearman: `0.453520`
- ROC-AUC: `0.796891`
- mean per-target Spearman: `0.465384`
- mean top-10% enrichment: `3.423712`
- 95% interval coverage: `0.952489`

## Interpretation

This is a credible baseline, not a state-of-the-art model.

What the numbers suggest:

- ranking quality is materially better than random
- the sequence-aware representation carries information across unseen targets
- there is still large room to improve regression accuracy and rank correlation

For artifact details, see:

- `results/baseline/metrics.json`
- `results/baseline/test_predictions.csv`
- `results/baseline/test_per_target_metrics.csv`
- `results/baseline/test_top_ranked_hits.csv`
