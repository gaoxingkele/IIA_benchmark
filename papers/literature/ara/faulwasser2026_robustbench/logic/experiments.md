# Experiments

No paper-score reproduction is asserted by collection generation alone.

## SOTA Wave 2 bounded RobustBench execution (E2/P1)

- The local protocol executes five corruption families, severities 0.1/0.2, progress 0.25/0.5/1.0, six classifiers, three real alarm datasets, and three outer seeds.
- Every point stores clean score, perturbed mean/SD/95% Monte Carlo interval, degradation, and normalized robustness AUC.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json` and Figures 1-2 in the Wave 2 harness directory.
- Boundary: the SSRN PDF endpoint remains HTTP 403; exact source datasets, severity mapping, repetitions, and paper tables are not claimed.
