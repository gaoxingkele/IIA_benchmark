# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC activation sequences, complete-run 60/20/20 split for all 16 scenarios.
- Method: maximum-entropy next-alarm model with time-distance features.
- Result: 3,550 evaluated transitions, vocabulary coverage 0.919927, top-1 accuracy 0.187324, top-3 accuracy 0.412958, NLL 3.155777, and Brier score 0.017883.
- Evidence: `configs/experiments/fcc_max_entropy_next_alarm_validation.json` and `experiments/reports/fcc_max_entropy_next_alarm_validation.json`.
- Boundary: simultaneous alarms require a declared deterministic ordering; the source paper event log, exact constraints, and paper score remain gated.
