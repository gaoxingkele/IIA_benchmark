# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm archive, 16 abnormal situations; complete runs 1-60 train, 61-80 calibration, 81-100 test.
- Method: priority/time-aware accelerated local alignment, 1-nearest-neighbour classification on activation sequences.
- Result: balanced accuracy 0.887500, macro-F1 0.890699, all 16 classes predicted; 8,415,297 dynamic-programming cells evaluated.
- Evidence: `configs/experiments/fcc_accelerated_alignment_validation.json` and `experiments/reports/fcc_accelerated_alignment_validation.json`.
- Boundary: task-matched transfer only; the paper flood corpus and paper score are unavailable, and this single dirty-worktree split receives no E3 or significance credit.

## Named items and three-dataset grouped validation (E3/P1)

- Table 5.5 scores 6/4.5/3 and the five matched pairs in Eq. 5.16 pass all three seeds.
- Mean balanced accuracy on TEP/NPP/FCC is 0.7300/0.1485/0.7736, versus 0.9500/0.6061/0.8806 for the frozen set-Jaccard control.
- Competitive credit is denied in all nine dataset-seed units. FCC grouped-unique accuracy is 0.1139 below the older fixed-split score, exposing duplicate-trajectory optimism.
- Evidence: `experiments/reports/book_ch5_multidataset_validation.json`.
- Boundary: alarm priorities are absent from public data; Table 5.12's 389-flood corpus and timing comparison remain unavailable.
