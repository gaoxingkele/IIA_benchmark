# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm states, complete-run 60/20/20 split for 16 abnormal-situation classes.
- Result: balanced accuracy 0.993750, macro-F1 0.993734, all classes predicted, minimum template stability 0.641034.
- Evidence: `configs/experiments/fcc_hdam_state_validation.json` and `experiments/reports/fcc_hdam_state_validation.json`.
- Interpretation/boundary: this reverses the PRONTO mismatch result (balanced accuracy 0.101010) and shows task-matched alarm episodes activate the model. It remains a single transfer split rather than the source-paper protocol or E3 reproduction.

## TEP five-class payload validation (E2/P2)

- The PRONTO 12-bin parent was stopped at 482.6 wall seconds before producing predictions because it implies about 307 million template-placement comparisons on 60-bin TEP episodes.
- A full 60-bin template, selected from episode length/runtime before any test prediction, completed in 9.67 s with balanced accuracy 0.975000, macro-F1 0.974902, all five classes predicted, and minimum stability 0.196923.
- Evidence: `experiments/paper_harness/tep_alarm_wave1/run_3_initial_runtime_failure.json` and `experiments/reports/tep_alarm_hdam_state_validation.json`.
- Boundary: runtime repair is admitted; exact paper alignment rules, template width, split, and scores remain unverified.
