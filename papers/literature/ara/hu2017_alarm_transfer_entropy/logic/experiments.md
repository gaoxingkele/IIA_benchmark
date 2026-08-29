# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## FCC, TEP Alarm, and NPP structural transfer (E2/P1-P2)

- Frozen protocol: two complete test episodes per class; six binary-entropy alarm variables after three-occurrence/three-clearance guards; lags 0-3; nine Bernoulli surrogates; significance 0.10.
- FCC: 30/32 runs activated, 394 significant NTE edges, 182 NDTE-pruned indirect edges; within/cross-class direct-edge Jaccard 0.170915/0.016544.
- TEP Alarm: 10/10 activated, 193 edges, 87 pruned; within/cross Jaccard 0.454998/0.037838.
- NPP alpha=0.50: 22/22 activated, 525 edges, 201 pruned; within/cross Jaccard 0.443203/0.031278.
- Evidence: `experiments/reports/fcc_alarm_nte_ndte_structural_validation.json`, `experiments/reports/tep_alarm_nte_ndte_structural_validation.json`, and `experiments/reports/npp_alarm_nte_ndte_structural_validation.json`.
- Boundary: the three payloads provide fault-family labels but not root-alarm tags. Graph activation, pruning, lag, and stability are credited; root-tag top-k/MRR and the unavailable industrial paper table are not claimed.
