# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC alarm archive, complete-run 60/20/20 split for each of 16 classes.
- Method: CHARM-equivalent vertical TID closure enumeration followed by representative-pattern clustering and Jaccard classification.
- Result: 130 closed patterns, 16 representatives, balanced accuracy 0.931250, macro-F1 0.929589.
- Runtime falsification: the parent frequent-subset-then-filter implementation exceeded four minutes and 753,811,456 B observed working set. Direct closure enumeration completed the unchanged run in 1.5357 s and matched a brute-force oracle on unit tests.
- Evidence: `experiments/reports/fcc_charm_patterns_validation.json` and `experiments/paper_harness/fcc_wave1/run_8_initial_runtime_failure.json`.
- Boundary: runtime-gene credit only; the original paper data/score and clean multi-seed E3 reproduction remain unavailable.
