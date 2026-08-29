# Experiments

The equation-defined numerical entry in Book Section 4.4.4.1 is executable at
`experiments/paper_harness/chapter4_numeric_exact`. Runs 1-3 use seeds 41, 42,
and 43 with 3,200 samples, the published 32 segments, unit Gaussian noise, and
the published delays 10 and 8. Every run recovers both delays and obtains 1.0
dominant-driver accuracy across all 32 segments. This is E1/P2 numerical
reproduction; the gated two-month thermal-plant payload remains necessary for
the industrial experiment and strict paper-score closure.

Machine-readable evidence:
`experiments/reports/book_ch4_plr_numeric_validation.json`.

## PIADE grouped process-transition transfer

The gap-closure harness applies PLR, lag correlation, non-negative regression,
and contribution normalization to real PIADE pressure/speed transitions. Three
disjoint chronological folds each contain 20 nonoverlapping windows for each
of five machines. All 300 windows have finite nonnegative factors; 129 activate
a nonzero contribution (`0.43 ± 0.01` across folds). Equipment activation rates
range from `0.2333` for `s_3` to `0.6167` for `s_5`.

iMAKS is retained as a negative diagnostic: its sustained target offset has
zero trend during the anomalous interval and only the recovery segment
activates, so recovered lags `[61, 0]` do not match the documented 180-sample
delay. PIADE supports E2 mechanism transfer, but neither dataset closes causal
Top-k/MRR or the unavailable paper plant score. Evidence:
`experiments/reports/book_ch4_gap_closure_validation.json`.
