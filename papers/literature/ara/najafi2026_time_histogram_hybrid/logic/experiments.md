# Experiments

No paper-score reproduction is asserted by collection generation alone.

## SOTA Wave 2 three-phase neural validation (E2/P1)

- Autoencoder pretraining, Transformer pretraining, joint fine-tuning, and positive learned attenuation activate for all TEP/NPP/FCC seeds.
- Mean balanced accuracy is 0.7017/0.2652/0.3698 and robustness AUC is 0.6850/0.2159/0.2599.
- The method loses all 9 classification and all 9 robustness paired comparisons against the class-core parent; the negative transfer is retained.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: full equations, paper hyperparameters/split, companion score, and source tables remain gated.
