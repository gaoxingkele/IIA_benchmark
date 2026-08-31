# Experiments

The paper's public synthetic dataset contains 18,750 binary alarm subsequences: 3,750 per class, five classes, ten alarm variables, and a 1/min sampling rate. Evaluation uses training/calibration/test partitions inside 10 repetitions of stratified 5-fold cross-validation (50 tests), expanding windows starting at 10 minutes with 1-minute steps, alpha in {0.1, 0.05, 0.01}, and class-balanced calibration sizes 22, 102, or 2491. Metrics are accuracy, coverage, and average prediction-set size. The complete Code Ocean v2 capsule is acquired and hash-verified.

## Author-code paper-grid reproduction (P2)

- The complete 250-task `split x model` grid was executed on the official 18,750-sequence payload and assembled into all 50 folds.
- All 95 named means from Tables I-II are within the frozen absolute tolerance 0.02; the largest absolute mean delta is 0.002271.
- The five reproduced original-accuracy means for WDI/MBW/EAC/ACM/CASIM are 0.312874/0.665364/0.684437/0.628891/0.706494.
- Evidence: `experiments/paper_harness/p0_paper_exact/run_2/paper_grid/summary.json` and `docs/reports/p0_cone_paper_grid_complete_2026-08-31.md`.
- Independent same-fold conformal-layer validation uses the author MBW-LR base scores but the repository `ConEAFCCalibrator` for thresholds and prediction sets. Across all 50 folds, 3 alpha values, 3 calibration sizes, and 4 set metrics, 1,800/1,800 paired rows match exactly with maximum absolute delta zero.
- Boundary: P3 remains open until archived-Docker execution and independent base-classifier/end-to-end wrapper implementations are evaluated on the identical 50 folds; the closed conformal-layer lane alone is not full P3.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC rising-edge alarms; 60 train, 20 calibration, and 20 test runs per each of 16 classes, with no window crossing a run boundary.
- Base classifier: local CTFH; explicit complete-run calibration; alpha 0.05.
- Result: coverage 0.940625, mean set size 8.009375/16, singleton rate 0.0. Sets are non-vacuous at every tested prefix but not decision-efficient.
- Evidence: `configs/experiments/fcc_cone_ctfh_uncertainty_validation.json` and `experiments/reports/fcc_cone_ctfh_uncertainty_validation.json`.
- Boundary: this is a transfer diagnostic, not the paper's 50-test synthetic-data reproduction.

## TEP five-class payload validation (E2/P2)

- Seeded complete-sample 120/40/40 split per class; CTFH rising-edge base; alpha 0.10 from the registered model.
- Full-prefix coverage 0.890000, mean set size 1.080000/5, empty rate 0.095000, singleton rate 0.730000, singleton accuracy 0.979452.
- The sets are efficient but coverage misses the nominal 0.90 target by 0.01; the deficit is retained rather than rounded into a pass.
- Evidence: `experiments/reports/tep_alarm_cone_ctfh_validation.json`.
- Boundary: this transfer result is not interchangeable with the now-complete official 18,750-subsequence, 50-test paper grid.

## NPP alpha-0.50 transfer validation (E2/P1)

- Unique-trajectory grouped 28/10/10 split per 11 classes; rising-edge CTFH base; explicit calibration; alpha 0.10.
- Full-prefix coverage 0.900000, mean set size 1.300000/11, empty rate 0.063636, singleton rate 0.572727, singleton accuracy 0.936508.
- Evidence: `experiments/reports/npp_alarm_cone_ctfh_validation.json`.
- Boundary: efficiency is non-degenerate and nominal coverage is met on this split, but it remains a cross-domain transfer result rather than the now-complete synthetic paper grid.

## SOTA Wave 2 grouped prefix validation (E2/P1)

- Eight expanding prefixes are calibrated independently on grouped TEP/NPP/FCC trajectories for seeds 1103/2207/3301.
- Full-prefix coverage/set size is 0.8167/1.0050, 0.7689/1.0871, and 0.9010/6.6745.
- Efficiency passes 9/9, but nominal 0.90 coverage passes only 2/9; the TEP/NPP undercoverage is retained.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: local CTFH base, not the paper's 18,750-subsequence 50-test grid or official tables.
