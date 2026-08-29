# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC representation-pair validation (E2/P1)

- Dataset/split: official FCC alarms, complete runs 1-60 train, 61-80 calibration, 81-100 test, 16 classes.
- State representation: 62 consensus hashes, 12 predicted classes, balanced accuracy 0.512500, macro-F1 0.464254.
- Rising-edge representation: 45 hashes, 10 predicted classes, balanced accuracy 0.365625, macro-F1 0.324116.
- Evidence: `experiments/reports/fcc_ctfh_state_validation.json` and `experiments/reports/fcc_ctfh_rising_edge_validation.json`.
- Interpretation/boundary: both activate, unlike PRONTO's zero hashes, but edge conversion loses useful duration information. These transfer results do not reproduce the paper's private protocol or score.
