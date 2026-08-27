# Real-Time Classification and Early Warning of Industrial Alarm Floods Using Modified TF-IDF Methods

## Metadata

- Year: 2025
- Venue: Control Engineering Practice
- DOI: [10.1016/j.conengprac.2025.106485](https://doi.org/10.1016/j.conengprac.2025.106485)
- Benchmark role: SOTA n-gram clustering, fault isolation and early alarm-flood warning
- ARA status: `engineering_validated_source_gated`

## Executive summary

Uses position-weighted n-gram TF-IDF and spectral clustering for historical floods, kernel PCA for fault isolation, and a five-stage LSTM classifier with a calibrated risk threshold for online early warning.

## Claims and evidence

- `C1` [explicit] The method combines position-weighted modified TF-IDF n-grams with spectral clustering.  
  Support locator: `publisher_method_and_contribution_snippets`
- `C2` [explicit] The online pipeline uses kernel PCA for fault isolation and LSTM for early classification.  
  Support locator: `publisher_contribution_snippet`
- `C3` [explicit] The case study uses alarm logs from a VAM production simulator.  
  Support locator: `publisher_case_study_snippet`
- `C4` [inferred] Local position decay, kernels, optimizer and network sizes remain independent choices until full-text acquisition.  
  Support locator: `local_evidence_boundary`

## Reproduction status

Source acquisition is `not_openly_downloadable`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
