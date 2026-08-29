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
| Criterion C | I1/I2/I3 inheritance + delayed flood state | `criterion_c_alarm_flood_detection` | A | SMD10TOWFGR six-month event-log candidates executed; expert flood truth and original year-scale score pending |
| Accelerated alignment | priority/time-aware seeds + banded local alignment | `accelerated_alarm_alignment` | A | paper runtime/accuracy table pending |
| Closed patterns | vertical TID closedness + delta representative cover | `charm_closed_alarm_patterns` | A | industrial 921-to-207 result pending |
| Maximum entropy | learned log-linear f1/f2 next-alarm distribution | `MaximumEntropyNextAlarmPredictor` | A | paper Monte Carlo score pending |
| AFC-RobustBench | missing/spurious/timing/detector-delay/mixed perturbations, progress/severity profiles, Monte-Carlo intervals and robustness AUC | `run_afc_robustness_benchmark` | B | full protocol is callable; paper-score comparison pending preprint PDF/data |
| CASIM | MultiRocket + ridge ensemble + LoOP | independent callable reproduction + official Code Ocean comparison slot | B | mechanics/invariants tested; official TEP score reproduction pending capsule/data |
| ConE-AFC | class- and stepwise inductive conformal prediction | independent callable wrapper + official Code Ocean comparison slot | B | Eqs. 1-5/Algorithms 1-2 tested; official synthetic-data tables pending capsule/data |
| Cross-conformal AFC | pooled fold-wise class-conditional p-values + explicit empty-set repair | `CrossConformalAlarmFloodClassifier` | B | standard mechanism/invariants tested; ICPS full text, exact postprocessing and synthetic paper scores gated |
| CTFH fingerprinting | AEM activation rates + local peaks + combinatorial temporal hashes + consensus profiles/variability | `CTFHAlarmFloodClassifier` | B | publisher-confirmed pipeline callable; paper-exact peak/hash parameters and TEP scores gated |
| Modified TF-IDF AFC | position-weighted n-grams + spectral clustering + kernel PCA isolation + LSTM prefix classification/alert threshold | `ModifiedTFIDFVectorizer`, `KernelPCAFaultIsolator`, `TFIDFLSTMAlarmFloodClassifier` | B | publisher-confirmed modules callable; paper formulas/hyperparameters, VAM data and scores gated |
| Structured HDAM | binned HDAM + variable-duration 2-D alignment + category medoid/consensus template + dynamic prefix matching | `HDAMTemplateMatcher` | B | publisher-confirmed pipeline callable; paper-exact convolution/template equations and TEP scores gated |
| Time-encoded histogram hybrid | learnable exponential histogram + separately pretrained autoencoder/Transformer + joint attenuation fine-tuning | `OptimalTimeEncodedHistogramClassifier` | B | publisher-confirmed trainable pipeline callable; exact paper equations/hyperparameters, TEP split and selected-paper scores gated |
| Uncertainty-reduction forecasting | random forest + leave-one-out jackknife+ + stable delay timer | `UncertaintyReductionForecaster` | B | full text acquired and statistical mechanics tested; Tables 1-4 await the official capsule and paper datasets |
| Chapter 6 visual analytics | bubble/treemap/analytics/bad-actor/radar/HDAP/3D-bar/correlation/workflow/event-flow/burst/similarity/spiral | `build_alarm_visual_analytics`, standalone HTML+JSON export | A | synthetic invariant validation passed; original industrial visual case pending |

Maturity A means callable and unit-tested, not equation-complete or paper-score
reproduced. Each reported experiment still needs a dataset adapter, grouped split,
frozen config, and an ARA validation record.
