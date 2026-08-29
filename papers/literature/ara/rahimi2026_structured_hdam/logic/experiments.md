# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm states, complete-run 60/20/20 split for 16 abnormal-situation classes.
- Result: balanced accuracy 0.993750, macro-F1 0.993734, all classes predicted, minimum template stability 0.641034.
- Evidence: `configs/experiments/fcc_hdam_state_validation.json` and `experiments/reports/fcc_hdam_state_validation.json`.
- Interpretation/boundary: this reverses the PRONTO mismatch result (balanced accuracy 0.101010) and shows task-matched alarm episodes activate the model. It remains a single transfer split rather than the source-paper protocol or E3 reproduction.
