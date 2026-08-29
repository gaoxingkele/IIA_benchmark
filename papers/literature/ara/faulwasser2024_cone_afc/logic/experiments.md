# Experiments

The paper's public synthetic dataset contains 18,750 binary alarm subsequences: 3,750 per class, five classes, ten alarm variables, and a 1/min sampling rate. Evaluation uses training/calibration/test partitions inside 10 repetitions of stratified 5-fold cross-validation (50 tests), expanding windows starting at 10 minutes with 1-minute steps, alpha in {0.1, 0.05, 0.01}, and class-balanced calibration sizes 22, 102, or 2491. Metrics are accuracy, coverage, and average prediction-set size. The Code Ocean capsule is currently HTTP 403 from this environment, so its exact data and table reproduction remain gated.

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
- Boundary: this is not the paper's 18,750-subsequence, 50-test calibration-size grid; Code Ocean access remains gated.

## NPP alpha-0.50 transfer validation (E2/P1)

- Unique-trajectory grouped 28/10/10 split per 11 classes; rising-edge CTFH base; explicit calibration; alpha 0.10.
- Full-prefix coverage 0.900000, mean set size 1.300000/11, empty rate 0.063636, singleton rate 0.572727, singleton accuracy 0.936508.
- Evidence: `experiments/reports/npp_alarm_cone_ctfh_validation.json`.
- Boundary: efficiency is non-degenerate and nominal coverage is met on this split, but the paper's synthetic payload, 50-test grid, and calibration-size ablation remain blocked.
