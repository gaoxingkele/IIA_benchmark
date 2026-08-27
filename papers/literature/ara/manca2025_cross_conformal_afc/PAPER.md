# Data-Efficient Handling of Temporary Uncertainties in Online Alarm Flood Classification

## Metadata

- Year: 2025
- Venue: IEEE ICPS
- DOI: [10.1109/ICPS65515.2025.11087828](https://doi.org/10.1109/ICPS65515.2025.11087828)
- Benchmark role: SOTA data-efficient uncertainty-aware online alarm flood classification
- ARA status: `engineering_validated_source_gated`

## Executive summary

Combines fold-wise cross-conformal prediction with expanding alarm-flood classifiers and postprocesses erroneously empty prediction sets.

## Claims and evidence

- `C1` [explicit] The method combines cross-conformal prediction with early alarm-flood classification to reduce calibration-data demand.  
  Support locator: `publisher_and_author_abstract`
- `C2` [explicit] The paper introduces postprocessing for erroneously empty prediction sets.  
  Support locator: `publisher_abstract`
- `C3` [inferred] The current local top-p-value repair is an explicit independent choice and not asserted to be the paper-exact rule.  
  Support locator: `local_evidence_boundary`

## Reproduction status

Source acquisition is `not_openly_downloadable`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
