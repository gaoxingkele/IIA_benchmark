# Experiments

No paper-score reproduction is asserted by collection generation alone.

## SOTA Wave 2 grouped multi-dataset validation (E2/P1)

- Training-only spectral silhouette selects among n=1..4 before a 100-epoch LSTM on each grouped TEP/NPP/FCC split; unigram wins every run.
- Mean balanced accuracy is 0.8450/0.6250/0.9896 with seed SD 0.1039/0.1311/0.0045; paired classification credit passes 3/9.
- Full-progress robustness AUC is 0.7783/0.5348/0.7151; the TEP fit averages 423.35 seconds.
- KPCA fault isolation is explicitly not applicable because all three payloads lack a normal-operation class.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: modified equations, VAM data, KPCA normal/fault protocol, paper hyperparameters, and reference scores remain gated.
