# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm states; complete runs 1-60 train and 81-100 test for all 16 abnormal-situation classes.
- Registered model: full local CASIM configuration, 672 features and 10 classifiers.
- Result: balanced accuracy 1.000000, macro-F1 1.000000, all 16 classes predicted; wall time 205.15 s.
- Evidence: `configs/experiments/fcc_casim_state_validation.json` and `experiments/reports/fcc_casim_state_validation.json`.
- Boundary: a single fixed transfer split can be easy; this does not reproduce the source paper's protocol or grant paired-significance/E3 credit.

## TEP five-class payload validation (E2/P2)

- Exact public payload with seed-1103 120/40/40 complete-sample split per class.
- Full registered CASIM (672 features, 10 classifiers) completed in 191.38 s and obtained balanced accuracy/macro-F1 1.000000 across all five classes.
- G0 class-centroid accuracy is already 0.975, so the fixed split is easy and perfect separation alone is not evidence of universal superiority.
- Evidence: `configs/experiments/tep_alarm_casim_state_validation.json` and `experiments/reports/tep_alarm_casim_state_validation.json`.
- Boundary: exact public data is available, but the paper's repeated split, online grid, reference table, and official capsule equivalence remain open.
