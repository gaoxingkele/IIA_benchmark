# Experiments

The controlled coal-feeder recreation in
`experiments/paper_harness/chapter4_numeric_exact/run_4` uses zero initial
probabilities, response time `m=20`, known single and co-existing causes, the
unknown-cause interval, and all false/missing-alarm intervals explicitly listed
in Book Section 4.3.4.1. Stable-region accuracy is 1.0. On the 51 nuisance
samples, recursive Bayesian accuracy is 0.8627 versus 0.0588 for instantaneous
state lookup. Its whole-stream accuracy is lower (0.9491 versus 0.9850) because
the intended state-change filter delays transitions. The unavailable plant
waveforms deny exact Figure 4.37/paper-score credit, so evidence remains E1/P1.

Machine-readable evidence:
`experiments/reports/book_ch4_recursive_bn_controlled_validation.json`.

## EnAS event-log transfer

The three-seed gap-closure harness evaluates all 219,893 EnAS exception-log
rows, including 103 ME, 52 HE, and 5 UE markers. The raw one-row error impulses
produce no nonempty recursive-BN decision, which is retained as a
representation mismatch. A preregistered five-row forward-persistence adapter
then activates all marked events. Known-candidate rates are `0.6796`, `0.1346`,
and `0.6000` for ME/HE/UE; unknown-cause rates are `0.3204`, `0.8654`, and
`0.4000`.

This is E2 real-data mechanism evidence, not root-tag accuracy: EnAS has manual
error categories but no per-event physical root truth. Exact thermal-plant
scores remain blocked. Evidence:
`experiments/reports/book_ch4_gap_closure_validation.json`.
