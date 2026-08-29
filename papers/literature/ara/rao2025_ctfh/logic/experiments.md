# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC representation-pair validation (E2/P1)

- Dataset/split: official FCC alarms, complete runs 1-60 train, 61-80 calibration, 81-100 test, 16 classes.
- State representation: 62 consensus hashes, 12 predicted classes, balanced accuracy 0.512500, macro-F1 0.464254.
- Rising-edge representation: 45 hashes, 10 predicted classes, balanced accuracy 0.365625, macro-F1 0.324116.
- Evidence: `experiments/reports/fcc_ctfh_state_validation.json` and `experiments/reports/fcc_ctfh_rising_edge_validation.json`.
- Interpretation/boundary: both activate, unlike PRONTO's zero hashes, but edge conversion loses useful duration information. These transfer results do not reproduce the paper's private protocol or score.

## TEP five-class payload validation (E2/P2)

- Exact public payload: 1,000 complete 300-minute samples, 200 per IDV1/IDV2/IDV6/IDV14/IDV1+IDV5; seeded 120/40/40 split per class.
- State: 996 consensus hashes, balanced accuracy 0.725000, macro-F1 0.655143. Rising edge: 995 hashes, balanced accuracy 0.750000, macro-F1 0.683298.
- Both representations give IDV14 recall 0; the algorithm activates but fails this class distinction.
- Evidence: `experiments/reports/tep_alarm_ctfh_state_validation.json` and `experiments/reports/tep_alarm_ctfh_rising_edge_validation.json`.
- Boundary: the payload is exact and task-matched, but the paper split, parameters, repetitions, and target table are not yet closed.

## NPP alpha-0.50 transfer validation (E2/P1)

- G0 split: 11 fault families, 48 unique trajectory-component representatives per class, partitioned 28/10/10 train/calibration/test; no exact state or rising-edge component crosses partitions.
- State: 3,317 consensus hashes, balanced accuracy 0.836364, macro-F1 0.827591. Rising edge: 3,164 hashes, balanced accuracy 0.818182, macro-F1 0.812328.
- Both representations predict all 11 classes; state is slightly stronger, confirming that the FCC/TEP representation ordering is not universal.
- Evidence: `experiments/reports/npp_alarm_ctfh_state_validation.json`, `experiments/reports/npp_alarm_ctfh_rising_edge_validation.json`, and `experiments/reports/npp_alarm_alpha050_prior_validation.json`.
- Boundary: this is a grouped NPP transfer experiment, not the source paper's TEP score or exact repetition protocol.

## SOTA Wave 2 grouped classification and robustness (E2/P1)

- Three grouped seeds produce mean balanced accuracy 0.7350/0.8371/0.3828 on TEP/NPP/FCC; mechanism and chance-performance gates pass 9/9.
- Relative to class-core Jaccard, classification paired credit passes only the three NPP seeds; robustness credit passes 1/9.
- Full-progress five-regime robustness AUC is 0.7967/0.6864/0.2922.
- Evidence: `experiments/reports/sota_wave2_multidataset_validation.json`.
- Boundary: source equations, hash/peak parameters, paper split/repetitions, and target tables remain gated.
