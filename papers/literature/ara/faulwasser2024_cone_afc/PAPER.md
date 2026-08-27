# Addressing Uncertainty in Online Alarm Flood Classification Using Conformal Prediction

## Metadata

- Year: 2024
- Venue: IEEE Access
- DOI: [10.1109/ACCESS.2024.3492348](https://doi.org/10.1109/ACCESS.2024.3492348)
- Benchmark role: SOTA uncertainty-aware online alarm flood classification
- ARA status: `engineering_validated_source_acquired`

## Executive summary

ConE-AFC wraps online alarm-flood classifiers with conformal calibration and returns prediction sets at controlled error levels.

## Claims and evidence

- `C1` [explicit] Conformal prediction exposes uncertainty as label sets for expanding alarm-sequence prefixes.  
  Support locator: `paper_pdf_algorithms_1_2_and_equations_1_5`
- `C2` [explicit] The authors provide a versioned Code Ocean reproduction capsule.  
  Support locator: `paper_pdf_code_and_data_availability_statement`
- `C3` [explicit] The evaluation uses 18,750 synthetic alarm subsequences and 50 repeated stratified folds.
  Support locator: `paper_pdf_sections_IV_A_IV_C`

## Reproduction status

Source acquisition is `downloaded`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
