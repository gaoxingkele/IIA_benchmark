MOMENT-1-base / released512 context / SKAB / full frozen zero-shot evaluation.

All 33 registered test runs and 36,656 test timestamps were scored. One complete anomaly-free run is used for training/reference and other/1.csv is reserved for validation before outcomes. The validation run retains its native anomalies; the nominal1% quantile is not a normal-only false-alarm guarantee. Original 109,641,608 parameters were strictly loaded; no random head, fine-tuning or parameter fitting. Full CPU score time: 62.312s.

| Metric | Value | Protocol / aggregation |
|---|---:|---|
| F1 | 0.247579199 | Validation-only1% threshold, noPA, micro |
| Precision | 0.537782394 | Same fixed threshold |
| Recall | 0.16080441 | Same fixed threshold |
| AUROC | 0.580650541 | Pooled timestamp ranking |
| AP | 0.425255826 | Pooled timestamp ranking |
| VUS_ROC | 0.772020024 | Macro,33defined/0undefined |
| VUS_PR | 0.695475542 | Macro,33defined/0undefined |
| Affiliation F | 0.78603461 | Macro,12defined/21undefined |

Primary F1 entity-cluster95% interval: {"low": 0.09717899155700205, "high": 0.37988971049519965, "resamples": 200, "seed": 10042, "unit": "entity_cluster", "block_points": null, "boundary": "Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty."}. Conditional on this fixed released checkpoint and threshold, not independent training seeds.

| Same frozen scores | F1 |
|---|---:|
| Validation threshold,noPA | 0.247579199 |
| Same validation threshold,ground-truth PA | 0.378787879 |
| Test-pool1% threshold,noPA | 0.0184407796 |
| Same test-pool threshold,ground-truth PA | 0.258629776 |
| Best test-label-selected plain grid | 0.391071045 |
| Best test-label-selected PA grid | 0.847469895 |

PA at the same validation threshold adds 1,354 counted true-positive timestamps; false positives stay 1,780. This is a metric-protocol effect on identical scores, not an improved detector. Test calibration and test-label oracle controls remain separate from the primary result.

The separate industrial layout was independently checked, including every native train/validation/test score, label, timestamp, source row and fault-label array present. All primary/core/event/bootstrap values recomputed exactly in Python3.10.21/NumPy1.25.2/Torch2.2.1+cu121 CPU. 38 relevant validation tests passed. All original VUS integer buffers and250thresholds per entity, source identities and macro/undefined counts were verified. Range integrals and model inference were not numerically rerun in this saved-score review. Offline512-point context can use future points; delay is not causal deployment latency. Historical-paper weights and complete TAB training-harness equivalence are not certified. All72 MOMENT jobs and the full45/681/273 obligations remain active.

Paper: https://arxiv.org/abs/2402.03885
Checkpoint: https://huggingface.co/AutonLab/MOMENT-1-base
Evaluation reference: https://github.com/decisionintelligence/TAB/tree/5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2
