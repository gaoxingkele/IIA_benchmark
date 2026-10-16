# CrossAD GECCO: first complete five-seed cohort and fixed-score protocol diagnostics

The five prespecified training seeds 1103..1107 completed the frozen original CUDA/FP32 recipe and full fresh checkpoint replay.
The newest seed1107 completed all20 epochs and8640 training forwards. Seed1104 legitimately stopped at19 epochs under the original patience; other four completed20.
Each model covers all55408 train,13852 validation and69261 test points, including the13-point tail. Source and fresh replay score payload bytes match exactly for every split.
All five source file hashes, labels and original row identities match. The independent standard-library audit recalculated full thresholds, confusion counts, F1, ROC-AUC, AP and event diagnostics.
Source and replay resource guards passed with exit0 and no memory/GPU abort. GPU results never replace the separately required original CPU experiments.

The new diagnostic registry retains all139 conditions and22 complete primary cohorts. Each completed model is analyzed at all three prespecified alarm budgets0.1/1/5%, crossing validation-only versus pooled train-plus-test score calibration with raw versus point-adjusted predictions.
All four cells use identical complete model scores and identical float64 linear quantiles/strict greater-than. Pooled calibration uses test scores without test labels; PA uses test event labels to fill hit contiguous anomaly events. No threshold is chosen using test labels.
This calculation adds zero model training updates and zero neural forwards. It is an evaluation protocol diagnostic, not a model architecture ablation or a new training seed.
The source concatenates the dataset timeline; original entity boundaries are not reconstructed. Therefore these rows are not certified as full TAB results and are not a numerical reproduction claim for the historical paper table.

Primary1% alarm budget, F1 scale0..1:

| Calibration / PA | Five-seed mean F1 | Seed t95 interval |
|---|---:|---|
| validation_raw | 0.402890 | [0.394008, 0.411772] |
| validation_PA | 0.464080 | [0.460594, 0.467567] |
| pooled_train_test_raw | 0.483827 | [0.480036, 0.487617] |
| pooled_train_test_PA | 0.597642 | [0.587164, 0.608121] |

The paired combined change is 0.194753 (five-seed descriptive t95 [0.18358193570860076, 0.20592312854060713]).
The raw strict ROC-AUC mean is 0.986118; AP mean is 0.545966. High ranking AUC alone does not imply high F1 at the frozen alert budget.
Intervals describe variation across these five fitted training seeds on this single fixed file. They do not quantify unseen-entity uncertainty, establish algorithm superiority, or imply significance across the full experimental family.
Release checks and seed2025 controls are excluded from these five-seed means. No partial cohort receives a mean or interval.

CSV artifacts contain60 individual protocol rows,15 strict metric rows and27 protocol/paired-difference summaries across all three budgets.
All five complete raw proof reports and guard receipts are retained.13 relevant mathematical, provenance and partial-cohort checks passed.

Current CrossAD CUDA closure is5/139 conditions and1/22 primary cohorts;134 conditions and21 cohorts remain. The live original source controller has advanced to the seed2025 control.
All45 papers,681 method/baseline/ablation records,273 original data tasks,702 scalar targets, and the original/TAB/new-data tracks remain in scope. The overall goal is unfinished.
