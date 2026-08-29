# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC Criterion-C prior validation (E2/P1)

- Dataset: all 1,600 FCC alarm runs, each with 57 binary alarm variables and 60 one-minute samples.
- Check: Criterion-C candidate intervals were computed as a descriptive prior; 1,464 candidate intervals occurred across 1,352 runs.
- Evidence: `configs/models/criterion_c_fcc.json` and `experiments/reports/fcc_alarm_prior_validation.json`.
- Boundary: FCC has no expert flood-interval labels, so candidate intervals are not treated as detection truth and no precision/recall or paper-score claim is made.

## TEP five-class Criterion-C prior (E2/P2)

- Registered parameters (10-minute attention, 30-minute long-standing suppression, threshold 10, delay 2) produce 4,692 descriptive candidate intervals in 807/1,000 samples.
- Evidence: `configs/models/criterion_c_tep_five_class.json` and `experiments/reports/tep_alarm_five_class_prior_validation.json`.
- Boundary: class labels identify disturbances, not expert interval boundaries; no flood-detection precision/recall or paper-table claim is permitted.

## Three-dataset grouped validation (E2/P1)

- Seeds: 1103, 2207, and 3301 on TEP Alarm, NPP Alarm, and grouped-unique FCC Alarm.
- Result: the mean fractions of held-out runs containing Criterion-C candidates are 0.8000, 0.9364, and 0.8653. Mechanism activation passes all nine dataset-seed units.
- Gate: performance activation and competitive credit are undefined because none of the payloads supplies expert flood start/end intervals.
- Evidence: `experiments/reports/book_ch5_multidataset_validation.json`.
- Boundary: the 2,226-variable plant history, year-scale comparison, and paper FAR/MAR/delay scores remain unavailable.
