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
