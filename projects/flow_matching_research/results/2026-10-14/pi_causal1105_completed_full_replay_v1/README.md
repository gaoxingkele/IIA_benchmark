Current update: full checkpoint re-inference and the final independent review completed on 2026-10-14. See [complete model closure](../pi_causal1105_actual_complete_model_replay_closure_v1/README.md). The registration snapshot below records the earlier pending state; its frozen configurations and evidence remain unchanged.

Original causal-axis Pi-Transformer PSM seed1105 completed all 3 epochs / 1554 minibatches with 1,930,289 parameters. The guarded full saved-score audit independently checked all 15 streams and 15 posthoc controls, 13,238,200 overlapping training score points and 87,800 test points; the original 41-point dropped tail is retained.

Saved-score PA F1: 0.9788686340. Pointwise no-PA F1 at the same source threshold: 0.0191087853 (AUROC 0.4868982793; AP 0.2741472625). The source uses test data for early stopping and pooled train/test score calibration. These are protocol diagnostics, not strict TAB leaderboard results. The original training has no persisted resource-usage receipt; its guard history is unproven.

Full checkpoint re-inference and the final independent replay review are still pending. The live native admission controller retains the original 24 GiB physical / 32 GiB commit admission, 16 GiB emergency headroom, CPU2 and unchanged score tolerances. The successful saved-score audit does not certify checkpoint re-inference. 57 relevant validation tests passed.

All 1140 original jobs, six seeds and 45-paper / 681-method / 273-data obligations remain. No retraining, extra seed, historical journal equivalence or between-seed uncertainty claim. The pinned paper, historical mirror, exact original configuration and dataset citations are carried in the runtime and child source receipts.
