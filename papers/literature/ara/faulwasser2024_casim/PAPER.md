# Convolutional Kernel-Based Classification of Industrial Alarm Floods

## Metadata

- Year: 2024
- Venue: Data-Centric Engineering
- DOI: [10.1017/dce.2024.22](https://doi.org/10.1017/dce.2024.22)
- Benchmark role: SOTA early/open-set alarm flood classification
- ARA status: `source_acquired_code_pending`

## Executive summary

CASIM transforms alarm sequences with randomized convolutional kernels, pools the responses, and uses a ridge ensemble plus local outlier probability for open-set decisions.

## Claims and evidence

- `C1` [explicit] CASIM uses convolutional-kernel features for early alarm-flood classification.  
  Support locator: `paper_pdf_method`
- `C2` [explicit] Open-set evaluation includes nuisance alarms, order changes, and unseen flood classes.  
  Support locator: `paper_pdf_experiments`
- `C3` [explicit] The authors provide a Code Ocean reproduction capsule.  
  Support locator: `publisher_reproducibility_statement`

## Reproduction status

Source acquisition is `downloaded`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
