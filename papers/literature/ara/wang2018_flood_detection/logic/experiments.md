# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC Criterion-C prior validation (E2/P1)

- Dataset: all 1,600 FCC alarm runs, each with 57 binary alarm variables and 60 one-minute samples.
- Check: Criterion-C candidate intervals were computed as a descriptive prior; 1,464 candidate intervals occurred across 1,352 runs.
- Evidence: `configs/models/criterion_c_fcc.json` and `experiments/reports/fcc_alarm_prior_validation.json`.
- Boundary: FCC has no expert flood-interval labels, so candidate intervals are not treated as detection truth and no precision/recall or paper-score claim is made.
