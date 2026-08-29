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
