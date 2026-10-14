MOMENT-1-base / released512 context / CICIDS / full frozen zero-shot evaluation.

Full6 original published entities,1,343,920 native test points and55,747 validation points. All109,641,608 fixed pretrained parameters were recorded as strictly loaded, with no new head, training or fitting. Original full scoring lasted16,447.89seconds on CPU. Current source/asset/data/result/model-reference/score hashes and guarded terminal success were verified.

| Metric | Value | Aggregation/protocol |
|---|---:|---|
|F1|0.0269767525|Validation-only1% threshold;noPA;micro|
|Precision|0.361425476|Same fixed threshold;micro|
|Recall|0.0140112762|Same fixed threshold;micro|
|AUROC|0.420365585|Pooled timestamp ranking|
|Average precision|0.292728099|Pooled timestamp ranking|
|VUS_ROC|0.607773935|Entity macro;6defined/0undefined|
|VUS_PR|0.306040991|Entity macro;6defined/0undefined|
|Affiliation F|0.340859184|Entity macro;validation-only1% threshold|

The primary F1 entity-cluster95% interval is {"low": 0.003410614433867328, "high": 0.061435771067431116, "resamples": 200, "seed": 10042, "unit": "entity_cluster", "block_points": null, "boundary": "Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty."}. It is conditional on this fixed checkpoint/threshold, not five independent training seeds or unseen-entity generalization.

| Same frozen scores | F1 |
|---|---:|
|Validation-only threshold, noPA|0.0269767525|
|Same validation-only threshold, with ground-truth PA|0.660394511|
|Test-pool1% calibration, noPA|0.0322168746|
|Same test-pool1% threshold, withPA|0.704287019|
|Best test-label-selected plain grid|0.352989471|
|Best test-label-selected PA grid|0.773421393|

With unchanged raw detector scores and the same validation threshold, PA adds 157,854 counted true-positive timestamps while false positives remain 7,956. This is an observed metric-protocol effect, not an improvement in the detector. It does not establish that any paper used the same conditions.

The full range evaluation retains250thresholds and every integer buffer per entity under the original2xmedian-event-length rule; the largest registered buffer is12,610 (12,611buffers). All VUS/Affiliation entities are defined. Source/hash/scope/macros of those already completed integrals were rechecked; the integrals were not numerically rerun in this review. All primary/core/event/grouped uncertainty values were independently recomputed exactly in the actual Python3.10/NumPy1.25.2/Torch2.2.1 producer environment.

Processed mTSBench entities preserve full registered channels and original held-out tests; no exact historical-author subset or unseen-entity claim. Offline512-point windows use future context, so reported delay is not causal deployment latency. This case reproduces saved-score evaluation for the locally frozen official release; model scores were not reinferred and historical-paper checkpoint/fullTAB-training-harness equivalence is not certified. All72 registered MOMENT contexts/checkpoint/dataset evaluations and the full45/681/273scope remain active.

Paper: https://arxiv.org/abs/2402.03885
Checkpoint: https://huggingface.co/AutonLab/MOMENT-1-base
Evaluation reference: https://github.com/decisionintelligence/TAB/tree/5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2
