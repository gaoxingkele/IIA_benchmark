SB objective adapter / SWaT / full frozen seed1103 experiment.

The original20-epoch CPU fit completed successfully. The complete saved checkpoint was restored without fitting or coupling solves. Every train, validation and test timestamp score was independently reinferred bitwise equal; all original core/event/bootstrap metrics recomputed exactly in the original environment.

| Metric | Value |
|---|---:|
| Precision | 0.174657417 |
| Recall | 0.841214917 |
| F1 | 0.289257655 |
| AUROC | 0.821176077 |
| Average precision | 0.723579747 |
| VUS_ROC (entity macro) | 0.67729425 |
| VUS_PR (entity macro) | 0.487158825 |

Primary metrics use the original frozen validation-only1% threshold without point adjustment. The conditional one-model temporal-block F1 interval is {"low": 0.23013879667070217, "high": 0.3440158320846526, "resamples": 200, "seed": 11103, "unit": "circular_temporal_block", "block_points": 1024, "boundary": "Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty."}.

Test normal timestamp false-alarm fraction at this unchanged threshold: 0.549274218. Full score distributions are retained in the diagnostic JSON. These test-label diagnostics were calculated after freezing calibration and did not select the threshold.

Test-score-calibrated and test-label-selected PA grid metrics are retained separately in the complete metrics JSON. They must not be substituted for the primary metric or interpreted as detector improvement.

Original numerical repair: 12-stage epsilon continuation and 20000 final iterations; unchanged final regularization, path, backbone, RNG, epochs and tolerance. Original failure and source remain preserved. This audit did not run a new solver or alter training settings.

This is a local MLP/backbone and velocity-discrepancy TSAD score adapter of the cited SB objective, not a reproduction of an original-author TSAD experiment or the complete TAB training harness. Full original range buffers, source identity and macros were audited; VUS/Affiliation integrals were not independently rerun. This is one full seed, not a completed five-seed comparison. The full45paper/681method-ablation/273data-task scope and all105 entropic repair jobs remain retained.

Paper: https://arxiv.org/abs/2302.00482
