# Callable baseline matrix

This table describes currently callable baselines. It is not the book-completeness
ledger. The authoritative 20-item closure ledger is
`configs/algorithms/book_algorithms.json`; none of those entries may be called fully
reproduced merely because a related baseline below is callable.

| Family | Book/source idea | Callable implementation | Maturity | Main limitations |
|---|---|---|---|---|
| Fixed alarm | threshold + on-delay + deadband | `ThresholdDelayDeadband` | A | sample-based delay only |
| Alarm design | FAR/MAR/AAD weighted search | `design_alarm` | A | plug-in estimates; Bayesian CI pending |
| IID delay design | symmetric on/off Markov chain + FAR/MAR/AAD + joint search | `AlarmOnOffDelay`, `iid_delay_timer_performance`, `design_iid_delay_timer` | A | Xu 2012 Examples 1-2 and Table VII reproduced; source industrial payload pending |
| Non-IID delay design | duration tails + Bayesian interval + grid search | `design_non_iid_delay_timer` | A | three-dataset negative/partial activation; exact paper payload pending |
| Deadband design | angle suitability index + Bayesian width | `deadband_index`, `design_deadband_width` | A | suitability passes 7/27 units; exact paper payload pending |
| Alarm probability plot | equal-count Markov states + four statistics | `build_alarm_probability_plot`, `select_alarm_probability_threshold` | A | three-dataset negative transfer; exact paper payload pending |
| MSPC | covariance normal zone | `MahalanobisAlarm` | A | unimodal elliptical zone |
| Multivariate NOZ | convex operating polytope + conditional bound | `ConvexHullNOZAlarm`, `convex_hull_fitness_index` | A | Figure 3.2 eta and Eq. 3.15 projection pass; three-dataset negative transfer; full robust subset loop and paper data pending |
| Non-convex NOZ | equation-3.18 angular search cones + radial boundaries | `SearchConeNOZAlarm` | A | three-dataset negative transfer; exact cone facets, alpha knee and paper CSTR score pending |
| Direction alarm | adaptive weighted gradients + normal rule matrix | `VariationDirectionAlarm` | A | rule matrices activate but held-out MAR is 0.9093/0.9848/0.9953 on TEP/PRONTO/SKAB; exact ATG parameter fit pending |
| Time-varying regression | Bayesian/ridge update + binomial freeze | `BayesianWindowRegressionAlarm` | A | 4/27 statistical gates pass and 0/27 pump-domain gates pass; original pump payload/score pending |
| Condenser monitor | kPa physical pressure model + search-cone NOZ | `CondenserNOZAlarm`, `condenser_alarm_rate_bounds` | A | Table 3.5 synthetic fit passes but 99% FAR upper is 0.3500; V1/V2 worst-case bounds and industrial Tables 3.6-3.7 pending |
| Binary RCA | lagged TE + surrogate threshold | `TransferEntropyRanker` | A | first-order discrete TE; indirect pruning pending |
| Alarm NTE/NDTE | OR histories + normalization + Bernoulli surrogates | `NormalizedTransferEntropyGraph` | A | industrial paper score pending |
| IGTE/IGDTE | information granules + OPTICS + conditional TE | `information_granulation_transfer_entropy` | A | paper two-tank/TEP score pending |
| Online BN RCA | recursive probabilities + unknown-cause pattern | `RecursiveBayesianAlarmRCA` | A | thermal-plant score pending |
| Trend contribution RCA | PLR + lag correlation + non-negative MLR | `PLRContributionRCA` | A | industrial paper score pending |
| Flood detection | newly activated unique tags in window | `detect_alarm_floods` | A | activation-only logs cannot recover standing alarms |
| Flood similarity | local sequence alignment | `smith_waterman_similarity` | A | priority/time-aware BLAST acceleration pending |
| Next alarm | all-current-tag context + distance decay | `EmpiricalNextAlarmPredictor` | A | transparent empirical approximation, not exact max-entropy solver |
| Criterion C | I1/I2/I3 inheritance + delayed flood state | `criterion_c_alarm_flood_detection` | A | TEP/NPP/FCC mechanism activates, but no expert flood intervals permit FAR/MAR/delay scoring; original plant data pending |
| Accelerated alignment | priority/time-aware lookup seeds + banded local alignment | `accelerated_alarm_alignment` | A | Table 5.5/Eq. 5.16 pass; multi-data BA 0.7300/0.1485/0.7736 and 0/9 competitive wins versus set Jaccard; paper Table 5.12 pending |
| Closed patterns | vertical TID closedness + bit-mask delta representative cover | `charm_closed_alarm_patterns`, `representative_alarm_patterns` | A | multi-data BA 0.9667/0.7182/0.9347 but equals class-core Jaccard; original industrial 921-to-207 result pending |
| Maximum entropy | learned log-linear f1/f2 next-alarm distribution + exact single-constraint solver | `MaximumEntropyNextAlarmPredictor`, `maximum_entropy_single_constraint` | A | Table 5.15 passes; all three transfer eta surrogates fail 0.8, original Tables 5.18-5.22/Monte Carlo pending |
| AFC-RobustBench | missing/spurious/timing/detector-delay/mixed perturbations, progress/severity profiles, Monte-Carlo intervals and robustness AUC | `run_afc_robustness_benchmark` | B | TEP/NPP/FCC × six classifiers × three outer seeds executed at E2; exact preprint datasets/protocol and scores remain blocked |
| CASIM | MultiRocket + ridge ensemble + LoOP | independent callable reproduction + official Code Ocean comparison slot | B | grouped TEP/NPP/FCC BA `1.0000/0.8182/0.9922`; robustness AUC `0.9733/0.7379/0.8318`; official capsule/paper scores pending |
| ConE-AFC | class- and stepwise inductive conformal prediction | independent callable wrapper + official Code Ocean comparison slot | B | full-prefix coverage/set size `0.8167/1.0050`, `0.7689/1.0871`, `0.9010/6.6745` on TEP/NPP/FCC; official tables pending |
| Cross-conformal AFC | pooled fold-wise class-conditional p-values + explicit empty-set repair | `CrossConformalAlarmFloodClassifier` | B | three-seed full-prefix coverage `0.9617/0.9735/0.9661` with set size `1.2467/1.5606/8.0703`; exact ICPS data/postprocessing gated |
| CTFH fingerprinting | AEM activation rates + local peaks + combinatorial temporal hashes + consensus profiles/variability | `CTFHAlarmFloodClassifier` | B | grouped BA `0.7350/0.8371/0.3828`; only NPP beats class-core consistently; paper-exact peak/hash parameters and scores gated |
| Modified TF-IDF AFC | position-weighted n-grams + spectral clustering + kernel PCA isolation + LSTM prefix classification/alert threshold | `ModifiedTFIDFVectorizer`, `KernelPCAFaultIsolator`, `TFIDFLSTMAlarmFloodClassifier` | B | three-data BA `0.8450/0.6250/0.9896`, high TEP/NPP seed variance, 100-epoch TEP fit about 419 s; KPCA is inapplicable without a normal class and VAM/paper scores remain gated |
| Structured HDAM | binned HDAM + variable-duration 2-D alignment + category medoid/consensus template + dynamic prefix matching | `HDAMTemplateMatcher` | B | grouped BA `0.9967/0.6932/0.9375`; robust AUC `1.0000/0.6621/0.7359`; NPP does not beat Jaccard, exact paper equations/scores gated |
| Time-encoded histogram hybrid | learnable exponential histogram + separately pretrained autoencoder/Transformer + joint attenuation fine-tuning | `OptimalTimeEncodedHistogramClassifier` | B | all three training phases activate, but BA is only `0.7017/0.2652/0.3698`; exact paper equations/hyperparameters/scores gated |
| Uncertainty-reduction forecasting | random forest + leave-one-out jackknife+ + stable delay timer | `UncertaintyReductionForecaster` | B | next-reduction MAE versus median baseline is `12.5743/16.0377`, `3.7065/4.0506`, `0.8197/2.3580` minutes; exact Tables 1-4 await official capsule/data |
| Chapter 6 visual analytics | bubble/treemap/analytics/bad-actor/radar/HDAP/3D-bar/correlation/workflow/event-flow/burst/similarity/spiral | `build_alarm_visual_analytics`, standalone HTML+JSON export | A | synthetic invariant validation passed; original industrial visual case pending |

Maturity A means callable and unit-tested, not equation-complete or paper-score
reproduced. Each reported experiment still needs a dataset adapter, grouped split,
frozen config, and an ARA validation record.
