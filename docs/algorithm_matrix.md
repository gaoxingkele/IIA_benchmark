# Callable baseline matrix

This table describes currently callable baselines. It is not the book-completeness
ledger. The authoritative 20-item closure ledger is
`configs/algorithms/book_algorithms.json`; none of those entries may be called fully
reproduced merely because a related baseline below is callable.

| Family | Book/source idea | Callable implementation | Maturity | Main limitations |
|---|---|---|---|---|
| Fixed alarm | threshold + on-delay + deadband | `ThresholdDelayDeadband` | A | sample-based delay only |
| Alarm design | FAR/MAR/AAD weighted search | `design_alarm` | A | plug-in estimates; Bayesian CI pending |
| IID delay design | Markov FAR/MAR/AAD equations 2.8/2.15/2.16 | `iid_delay_timer_performance` | A | paper industrial score pending |
| Non-IID delay design | duration tails + Bayesian interval + grid search | `design_non_iid_delay_timer` | A | paper industrial score pending |
| Deadband design | angle suitability index + Bayesian width | `deadband_index`, `design_deadband_width` | A | paper industrial score pending |
| Alarm probability plot | equal-count Markov states + four statistics | `build_alarm_probability_plot` | A | paper industrial score pending |
| MSPC | covariance normal zone | `MahalanobisAlarm` | A | unimodal elliptical zone |
| Multivariate NOZ | convex operating polytope + conditional bound | `ConvexHullNOZAlarm` | A | robust center trimming approximates full book search |
| Non-convex NOZ | angular search cones + radial boundaries | `SearchConeNOZAlarm` | A | exact paper cone-facet tuning pending |
| Direction alarm | adaptive weighted gradients + normal rule matrix | `VariationDirectionAlarm` | A | industrial rule matrix pending |
| Time-varying regression | Bayesian/ridge update + binomial freeze | `BayesianWindowRegressionAlarm` | A | pump paper score pending |
| Condenser monitor | physical pressure model + search-cone NOZ | `CondenserNOZAlarm` | A | Bayesian worst-case bounds and industrial score pending |
| Binary RCA | lagged TE + surrogate threshold | `TransferEntropyRanker` | A | first-order discrete TE; indirect pruning pending |
| Alarm NTE/NDTE | OR histories + normalization + Bernoulli surrogates | `NormalizedTransferEntropyGraph` | A | industrial paper score pending |
| IGTE/IGDTE | information granules + OPTICS + conditional TE | `information_granulation_transfer_entropy` | A | paper two-tank/TEP score pending |
| Online BN RCA | recursive probabilities + unknown-cause pattern | `RecursiveBayesianAlarmRCA` | A | thermal-plant score pending |
| Trend contribution RCA | PLR + lag correlation + non-negative MLR | `PLRContributionRCA` | A | industrial paper score pending |
| Flood detection | newly activated unique tags in window | `detect_alarm_floods` | A | activation-only logs cannot recover standing alarms |
| Flood similarity | local sequence alignment | `smith_waterman_similarity` | A | priority/time-aware BLAST acceleration pending |
| Next alarm | all-current-tag context + distance decay | `EmpiricalNextAlarmPredictor` | A | transparent empirical approximation, not exact max-entropy solver |
| Criterion C | I1/I2/I3 inheritance + delayed flood state | `criterion_c_alarm_flood_detection` | A | industrial year-scale validation pending |
| Accelerated alignment | priority/time-aware seeds + banded local alignment | `accelerated_alarm_alignment` | A | paper runtime/accuracy table pending |
| Closed patterns | vertical TID closedness + delta representative cover | `charm_closed_alarm_patterns` | A | industrial 921-to-207 result pending |
| Maximum entropy | learned log-linear f1/f2 next-alarm distribution | `MaximumEntropyNextAlarmPredictor` | A | paper Monte Carlo score pending |
| AFC-RobustBench | missing/spurious/timing/detector-delay/mixed perturbations, progress/severity profiles, Monte-Carlo intervals and robustness AUC | `run_afc_robustness_benchmark` | B | full protocol is callable; paper-score comparison pending preprint PDF/data |
| CASIM | MultiRocket + ridge ensemble + LoOP | independent callable reproduction + official Code Ocean comparison slot | B | mechanics/invariants tested; official TEP score reproduction pending capsule/data |
| ConE-AFC | class- and stepwise inductive conformal prediction | independent callable wrapper + official Code Ocean comparison slot | B | Eqs. 1-5/Algorithms 1-2 tested; official synthetic-data tables pending capsule/data |
| Chapter 6 visual analytics | bubble/treemap/analytics/bad-actor/radar/HDAP/3D-bar/correlation/workflow/event-flow/burst/similarity/spiral | `build_alarm_visual_analytics`, standalone HTML+JSON export | A | synthetic invariant validation passed; original industrial visual case pending |

Maturity A means callable and unit-tested, not equation-complete or paper-score
reproduced. Each reported experiment still needs a dataset adapter, grouped split,
frozen config, and an ARA validation record.
