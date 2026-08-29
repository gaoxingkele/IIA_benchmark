# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC rising-edge alarms with complete-run grouping; training and calibration runs form five folds, test runs remain untouched.
- Base classifier: local CTFH; alpha 0.05.
- Result: coverage 0.965625, mean set size 7.671875/16, singleton rate 0.0. Prefix sets are smaller than the label space but do not yield singleton early decisions.
- Evidence: `configs/experiments/fcc_cross_conformal_ctfh_uncertainty_validation.json` and `experiments/reports/fcc_cross_conformal_ctfh_uncertainty_validation.json`.
- Boundary: one transfer split, not a source-paper score reproduction or clean multi-seed E3 result.
