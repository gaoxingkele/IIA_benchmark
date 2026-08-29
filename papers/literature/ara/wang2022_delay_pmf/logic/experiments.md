# Experiments

The duration-PMF delay design executes on TEP, PRONTO, and SKAB for three
bootstrap seeds with disjoint calibration/evaluation samples. Its mean F1 is
0.7231/0.3332/0.1093. The defining duration mechanism is only partially
activated: 19 of 27 episode-seed units contain no observed normal-alarm or
abnormal-no-alarm run and therefore use a declared zero-event posterior for
diagnostic output. Those units receive no method-activation credit.

The gated paper payload and exact industrial tables remain unavailable, so this
is E2 negative engineering evidence rather than a paper-score reproduction.
Machine-readable evidence:
`experiments/reports/book_ch2_multidataset_validation.json`.
