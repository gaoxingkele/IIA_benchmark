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
