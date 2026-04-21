# REGINA
\textbf{R}egularized \textbf{E}ncoder with Latent Cycle-\textbf{G}AN for \textbf{I}n-vitro \textbf{N}eural Cell Perturbation \textbf{A}pproximation

## Benchmark metrics:
- Pearson correlation (top 50 vs genome wide)
- Pearson squared (top 50 vs genome wide)
- Directional error (top 50 vs genome wide)
- Wasserstein distance
- L1 distance
- Energy


## Steps to do:
### Bence
- Modify dataloader
- Train our model on data
- Finalize eval
- Start eval on GEARS results
- Write about datasets

### Regina
- Write inference code for models
    - Output should be same as in GEARS inference
    - Anndata.X = results
    - Anndata.gene = perturbation_name
    - Ndraws should be 100
- Write about datasets
