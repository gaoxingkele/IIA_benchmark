TranAD / SMAP / corrected constant-fit preprocessing / full float64 fit and score / seed1103.

Full original training exited0 without resource abort. Both independent saved-score review and full native checkpoint inference exited0 without abort. All153score arrays and51label arrays across51entities (101565train,26684validation,408160test points) match bitwise, including the compressed archive SHA256. Original float64,58073parameters,CPU2,batch128 and5configured epochs retained.

Validation1% threshold, noPA primary metrics: {"precision": 0.39069999616019657, "recall": 0.1984630088357487, "f1": 0.2632191639072847, "true_positives": 10175.0, "false_positives": 15868.0, "false_negatives": 41094.0, "auroc": 0.6206773093859507, "average_precision": 0.19900292956298427}
Conditional entity-cluster F1 sensitivity interval: {"low": 0.015498270829764658, "high": 0.4891938198636954, "resamples": 200, "seed": 11103, "unit": "entity_cluster", "block_points": null, "boundary": "Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty."}
This is one fitted model with200entity-bootstrap resamples, not five-training-seed uncertainty. Original test-calibrated/test-label-selected diagnostics remain separate.

Training geometry implies10179original nonoverlapping/end-aligned windows,80configured steps per epoch,400configured steps across5epochs. These are derived from frozen executed code and original data, not an independently measured optimizer-update trace. No per-epoch trace was saved.

The model restores the pinned author float64 precision, but local window construction/preprocessing/thresholds still differ from the full historical author padding/POT/environment. No author-pipeline or fullTAB-training-harness certification. All25original precision controls,35baseline slots,5corrected SMAP precision seeds,291corrected range-evaluation jobs and45papers/681methods/273data obligations remain required. 33relevant checks passed.

Paper: https://doi.org/10.14778/3514061.3514067
Data/evaluation citation: https://github.com/decisionintelligence/TAB
