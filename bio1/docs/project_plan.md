# Kinase Ligand Ranking Pipeline - Project Plan

## 1. Problem Statement

Given a kinase protein target and a set of candidate small molecules, predict
binding affinity (pIC50) and rank molecules by predicted potency. Include
uncertainty estimates so researchers know which predictions to trust.

This is a **candidate prioritization** tool, not a drug discovery claim.

## 2. Dataset Description

### 2.1 Primary Source: BindingDB

- **API**: REST endpoint `getLigandsByUniprots`
- **URL**: https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprots
- **Targets**: 8 well-studied kinases (EGFR, ABL1, CDK2, BRAF, SRC, VEGFR2, JAK2, Aurora A)
- **Measurement types**: IC50, Ki, Kd (nanomolar)
- **Affinity cutoff**: 10,000 nM (10 uM)
- **Expected size**: ~5,000-20,000 measurements depending on target

### 2.2 Fallback Source: Davis Kinase Dataset

- **Source**: Therapeutics Data Commons (TDC)
- **Content**: 442 kinases x 68 ligands, Kd measurements
- **Size**: ~25,772 drug-target pairs
- **Reference**: Davis et al., Nature Biotechnology 29, 1046-1051 (2011)

### 2.3 Data Processing Pipeline

```
Raw data (JSON/CSV)
    |
    v
Filter (exact measurements, valid SMILES, affinity range 0.01 nM - 1 mM)
    |
    v
Canonicalize SMILES (RDKit)
    |
    v
Deduplicate (geometric mean aggregation)
    |
    v
Convert to pIC50 (-log10 of IC50 in molar)
    |
    v
Split by protein target (70/15/15 train/val/test)
```

### 2.4 Key Design Decisions

1. **Split by target, not random**: Prevents data leakage. Tests generalization
   to unseen kinases rather than memorization of known ligand-target pairs.

2. **Geometric mean for aggregation**: Affinity measurements are log-normally
   distributed, so geometric mean is the appropriate central tendency.

3. **pIC50 scale**: -log10(IC50 in M). Higher = more potent. Converts
   multiplicative relationships to additive, better for regression models.

4. **Canonical SMILES**: Ensures same molecule always has the same string
   representation regardless of how it was drawn or encoded.

## 3. Pipeline Phases

### Phase 1: Repository Setup (complete)
- Directory structure, dependencies, documentation

### Phase 2: Data Acquisition (complete)
- `scripts/download_data.py`: Fetch from BindingDB API or Davis via TDC
- `scripts/process_dataset.py`: Clean, filter, deduplicate, split

### Phase 3: Feature Engineering
- Molecular fingerprints (Morgan/ECFP via RDKit, radius=2, 1024 bits)
- Molecular graphs for GNN (atom features + bond features via RDKit)
- Protein features (learnable kinase embeddings or pre-computed ESM-2)

### Phase 4: Model Training
- Baseline: MLP on Morgan fingerprints + kinase embedding
- GNN: AttentiveFP or GCN on molecular graphs + kinase embedding
- Loss: MSE on pIC50

### Phase 5: Ligand Ranking
- Predict pIC50 for all candidate ligands against a target kinase
- Sort by predicted score (descending = most potent first)

### Phase 6: Uncertainty Estimation
- Monte Carlo dropout: run N forward passes with dropout active at inference
- Compute mean prediction and standard deviation per ligand
- Output: predicted pIC50, uncertainty (std), confidence interval

### Phase 7: Evaluation
- Spearman rank correlation (primary ranking metric)
- Top-k enrichment factor (virtual screening performance)
- ROC-AUC (active vs. inactive discrimination)
- RMSE on pIC50 (regression accuracy)
- Generate plots: predicted vs. actual, uncertainty calibration, enrichment curves

### Phase 8: Documentation
- Methodology description in `docs/methodology.md`
- Results summary in `paper/draft.md`
- Updated README with final instructions

## 4. Evaluation Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Spearman rho | Rank correlation | > 0.5 |
| Top-10% enrichment | Active compounds in top predictions | > 2x random |
| ROC-AUC | Active/inactive discrimination | > 0.7 |
| RMSE (pIC50) | Prediction accuracy | < 1.5 |

## 5. Tools and Dependencies

- **RDKit**: Molecule parsing, validation, fingerprints, graph features
- **PyTorch**: Model training and MC dropout inference
- **PyTorch Geometric**: Graph neural network layers
- **pandas/numpy/scipy**: Data processing and metrics
- **scikit-learn**: Train/test utilities and additional metrics
- **matplotlib/seaborn**: Visualization

## 6. References

- BindingDB: Liu et al., Nucleic Acids Research (2007)
- Davis dataset: Davis et al., Nature Biotechnology 29, 1046-1051 (2011)
- MC Dropout: Gal & Ghahramani, ICML (2016)
- AttentiveFP: Xiong et al., Journal of Medicinal Chemistry (2020)
- Morgan fingerprints: Rogers & Hahn, Journal of Chemical Information and Modeling (2010)
