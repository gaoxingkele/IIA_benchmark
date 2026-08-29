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

## NPP alpha-0.50 transfer validation (E2/P1)

- Registered full 16-bin episode template on the fixed 11-class unique-trajectory split.
- Result: balanced accuracy 0.763636, macro-F1 0.752807, all 11 classes predicted, minimum template stability 0.802876, wall time 6.00 s.
- Main confusions are SGATR/SGBTR, LOCA/LOCAC, and SLBIC/SLBOC; the lower score than NPP CTFH/CASIM is retained as negative comparative evidence.
- Evidence: `experiments/reports/npp_alarm_hdam_state_validation.json`.
- Boundary: one transfer seed and an engineering template width; no paper-score reproduction is asserted.

## SOTA Wave 2 grouped classification and robustness (E2/P1)

- Full binned templates are fixed before test prediction for seeds 1103/2207/3301 on grouped TEP/NPP/FCC rising-edge trajectories.
- Mean balanced accuracy is 0.9967/0.6932/0.9375; classification paired credit passes 2/9 because NPP matches the parent and FCC gains are not consistently significant.
- Full-progress robustness AUC is 1.0000/0.6621/0.7359 with paired robustness credit in 6/9 dataset-seeds.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: exact template/convolution equations, paper width/split, and selected tables remain gated.
