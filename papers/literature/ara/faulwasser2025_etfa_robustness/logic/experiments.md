# Experiments

No paper-score reproduction is asserted by collection generation alone.

## SOTA Wave 2 corruption validation (E2/P1)

- Six classifiers are tested on grouped TEP/NPP/FCC episodes under missing, spurious, timing, detector-delay, and mixed test-only corruptions.
- Two severities, three observation prefixes, two Monte Carlo draws per outer seed, and seeds 1103/2207/3301 yield a reproducible severity/AUC ledger.
- Clean and corruption rankings differ; for example FCC modified TF-IDF has BA 0.9896 but robustness AUC 0.7151, while CASIM has 0.9922/0.8318.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: ETFA full text, original payload, perturbation mapping, and reference scores remain unavailable.
