# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC rising-edge alarms with complete-run grouping; training and calibration runs form five folds, test runs remain untouched.
- Base classifier: local CTFH; alpha 0.05.
- Result: coverage 0.965625, mean set size 7.671875/16, singleton rate 0.0. Prefix sets are smaller than the label space but do not yield singleton early decisions.
- Evidence: `configs/experiments/fcc_cross_conformal_ctfh_uncertainty_validation.json` and `experiments/reports/fcc_cross_conformal_ctfh_uncertainty_validation.json`.
- Boundary: one transfer split, not a source-paper score reproduction or clean multi-seed E3 result.

## TEP five-class payload validation (E2/P2)

- Five stratified train/calibration folds with untouched complete-sample test runs; CTFH rising-edge base and alpha 0.10.
- Coverage 0.965000, mean set size 1.260000/5, empty rate 0, singleton rate 0.740000, singleton accuracy 0.952703.
- Relative to split ConE on the same registered split, coverage and empty-set behavior improve at a 0.18-label mean set-size cost; no paired superiority claim is made from one seed.
- Evidence: `experiments/reports/tep_alarm_cross_conformal_ctfh_validation.json`.
- Boundary: the full paper equations/repetition protocol and exact reference table remain unclosed.

## NPP alpha-0.50 transfer validation (E2/P1)

- Five stratified train/calibration folds over unique trajectory components; untouched 11-class test set; rising-edge CTFH base and alpha 0.10.
- Coverage 0.963636, mean set size 1.536364/11, empty rate 0, singleton rate 0.463636, singleton accuracy 0.921569.
- Relative to split ConE, coverage improves by 0.063636 and empty sets disappear at a 0.236364-label mean set-size cost; no superiority claim is made from one seed.
- Evidence: `experiments/reports/npp_alarm_cross_conformal_ctfh_validation.json`.
- Boundary: the source paper protocol and exact target table remain unclosed.

## SOTA Wave 2 grouped cross-conformal validation (E2/P1)

- Three stratified folds and eight prefixes are fitted independently on grouped TEP/NPP/FCC pools for seeds 1103/2207/3301.
- Full-prefix coverage is 0.9617/0.9735/0.9661 with mean set size 1.2467/1.5606/8.0703 for 5/11/16 classes.
- Coverage and efficiency pass all 9 dataset-seeds; the larger FCC set cost is retained.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: exact ICPS equations, small-calibration grid, data, and postprocessing table remain gated.
