MOMENT-1-base / release512 context / SMAP corrected constant-fit / full zero-shot evaluation.

All51 published entities and408160test points. All109641608 fixed parameters were strictly loaded; no fitting, head reinitialization or new training seed. Original producer environment:{"python": "3.10.21", "numpy": "1.25.2", "torch": "2.2.1+cu121", "device": "cpu", "sklearn": "1.5.2"}

Primary validation-only1% threshold, noPA:{"precision": 0.44005582693649686, "recall": 0.06149915153406542, "f1": 0.10791662388335557, "true_positives": 3153.0, "false_positives": 4012.0, "false_negatives": 48116.0, "auroc": 0.4801827655794931, "average_precision": 0.1529069113529905}
Conditional entity-cluster uncertainty:{"low": 0.012659862090339013, "high": 0.20722023156971875, "resamples": 200, "seed": 10042, "unit": "entity_cluster", "block_points": null, "boundary": "Conditional on this fitted model and fixed validation threshold; not paired algorithm or seed uncertainty."}
These200 resamples describe one fixed model and threshold; they are not independent training seeds.

Same frozen score protocol controls:{"validation_threshold": 7.676575622558569, "validation_only_no_PA": {"precision": 0.44005582693649686, "recall": 0.06149915153406542, "f1": 0.10791662388335557, "true_positives": 3153.0, "false_positives": 4012.0, "false_negatives": 48116.0}, "validation_only_same_threshold_with_PA": {"precision": 0.890874473004216, "recall": 0.6388460863289708, "f1": 0.7440988708907923, "true_positives": 32753.0, "false_positives": 4012.0, "false_negatives": 18516.0}, "test_pool_calibration_ratio1": {"ratio_percent": 1, "threshold": 14.366156997680793, "f_score": 0.0645392742470717, "adjust_f_score": 0.6657928578494411}, "best_test_label_selected_plain_grid_F1": 0.17901761295263102, "best_test_label_selected_PA_grid_F1": 0.7388912494129749, "all_controls_use_identical_frozen_score_arrays": true, "primary_threshold_selected_using_test_features_or_labels": false, "PA_control_uses_test_ground_truth_events": true}
At identical scores and the same validation threshold, ground-truth PA adds29600counted true-positive timestamps; false positives stay4012. This is a metric-protocol effect, not an improved detector or a claim about how a paper evaluated.

Full entity-macro range metrics:{"VUS_ROC": {"defined_entities": 51, "undefined_entities": 0, "mean_over_defined_entities": 0.6728547405683409}, "VUS_PR": {"defined_entities": 51, "undefined_entities": 0, "mean_over_defined_entities": 0.24280551364232608}}
Strict validation-only affiliation macros, retaining every undefined entity:{"affiliation_f": {"defined_entities": 50, "undefined_entities": 1, "mean_over_defined_entities": 0.5330779141833802}, "affiliation_precision": {"defined_entities": 50, "undefined_entities": 1, "mean_over_defined_entities": 0.4432531180745913}, "affiliation_recall": {"defined_entities": 51, "undefined_entities": 0, "mean_over_defined_entities": 0.7085391492590611}}

Original full VUS/Affiliation source hashes, array identity, every original integer buffer,250threshold budget and macro arithmetic were verified. No range-integral numerical rerun and no model score re-inference in this saved-score review. All3calibrations,10TAB-core grid controls,event metrics and grouped bootstrap values were independently recomputed exactly in the producer environment.

Original constant-fit preprocessing, entity-isolated native data and offline512-point context retained. This is the current official checkpoint zero-shot evaluation, not historical-paper weight, fine-tuning, author-pipeline or fullTAB-training-harness equivalence. All6corrected MOMENT slots,72original MOMENT jobs,291corrected data-control range jobs and45papers/681methods/273data obligations remain required. 15relevant checks passed.

Paper:https://arxiv.org/abs/2402.03885
Checkpoint:https://huggingface.co/AutonLab/MOMENT-1-base
Evaluation reference:https://github.com/decisionintelligence/TAB/tree/5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2
