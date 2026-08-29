# Experiments

The paper's public synthetic dataset contains 18,750 binary alarm subsequences: 3,750 per class, five classes, ten alarm variables, and a 1/min sampling rate. Evaluation uses training/calibration/test partitions inside 10 repetitions of stratified 5-fold cross-validation (50 tests), expanding windows starting at 10 minutes with 1-minute steps, alpha in {0.1, 0.05, 0.01}, and class-balanced calibration sizes 22, 102, or 2491. Metrics are accuracy, coverage, and average prediction-set size. The Code Ocean capsule is currently HTTP 403 from this environment, so its exact data and table reproduction remain gated.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC rising-edge alarms; 60 train, 20 calibration, and 20 test runs per each of 16 classes, with no window crossing a run boundary.
- Base classifier: local CTFH; explicit complete-run calibration; alpha 0.05.
- Result: coverage 0.940625, mean set size 8.009375/16, singleton rate 0.0. Sets are non-vacuous at every tested prefix but not decision-efficient.
- Evidence: `configs/experiments/fcc_cone_ctfh_uncertainty_validation.json` and `experiments/reports/fcc_cone_ctfh_uncertainty_validation.json`.
- Boundary: this is a transfer diagnostic, not the paper's 50-test synthetic-data reproduction.
