# Algorithm matrix

| Family | Book/source idea | Callable implementation | Maturity | Main limitations |
|---|---|---|---|---|
| Fixed alarm | threshold + on-delay + deadband | `ThresholdDelayDeadband` | A | sample-based delay only |
| Alarm design | FAR/MAR/AAD weighted search | `design_alarm` | A | plug-in estimates; Bayesian CI pending |
| MSPC | covariance normal zone | `MahalanobisAlarm` | A | unimodal elliptical zone |
| Multivariate NOZ | convex operating polytope + conditional bound | `ConvexHullNOZAlarm` | A | robust center trimming approximates full book search |
| Binary RCA | lagged TE + surrogate threshold | `TransferEntropyRanker` | A | first-order discrete TE; indirect pruning pending |
| Flood detection | newly activated unique tags in window | `detect_alarm_floods` | A | activation-only logs cannot recover standing alarms |
| Flood similarity | local sequence alignment | `smith_waterman_similarity` | A | priority/time-aware BLAST acceleration pending |
| Next alarm | all-current-tag context + distance decay | `EmpiricalNextAlarmPredictor` | A | transparent empirical approximation, not exact max-entropy solver |
| Robustness | missing/spurious/timing perturbations | `perturb_alarm_episode` | A | detector-delay perturbation remains protocol-level |
| CASIM | MultiRocket + ridge ensemble + LoOP | official Code Ocean reproduction slot | B | environment/data not yet locked |
| ConE-AFC | conformal early classification | metrics ready; model slot | B | calibration split and artifact pending |
| AFC-RobustBench | four realistic alarm corruptions | protocol partially implemented | C | 2026 preprint; official release audit pending |

Maturity A means callable and unit-tested, not that it reproduces the authors' reported score. Each reported experiment still needs a dataset adapter, grouped split and frozen config.
