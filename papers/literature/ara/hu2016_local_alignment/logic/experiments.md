# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm archive, 16 abnormal situations; complete runs 1-60 train, 61-80 calibration, 81-100 test.
- Method: priority/time-aware accelerated local alignment, 1-nearest-neighbour classification on activation sequences.
- Result: balanced accuracy 0.887500, macro-F1 0.890699, all 16 classes predicted; 8,415,297 dynamic-programming cells evaluated.
- Evidence: `configs/experiments/fcc_accelerated_alignment_validation.json` and `experiments/reports/fcc_accelerated_alignment_validation.json`.
- Boundary: task-matched transfer only; the paper flood corpus and paper score are unavailable, and this single dirty-worktree split receives no E3 or significance credit.
