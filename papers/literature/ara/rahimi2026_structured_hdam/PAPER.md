# Online Alarm Flood Classification via Interpretable Template Extraction and Structured Convolutional Matching

## Metadata

- Year: 2026
- Venue: Computers & Chemical Engineering
- DOI: [10.1016/j.compchemeng.2026.109570](https://doi.org/10.1016/j.compchemeng.2026.109570)
- Benchmark role: SOTA interpretable visual-template online alarm flood classification
- ARA status: `engineering_validated_source_gated`

## Executive summary

Extends HDAP to a structured HDAM, aligns variable-duration historical floods through two-dimensional convolution, extracts category templates, and matches a dynamic online HDAM for early classification.

## Claims and evidence

- `C1` [explicit] The method extends HDAP into a structured matrix representation for offline and online use.  
  Support locator: `publisher_abstract_and_method_snippets`
- `C2` [explicit] Structured two-dimensional convolution aligns floods of different durations and extracts category templates.  
  Support locator: `publisher_contribution_snippet`
- `C3` [explicit] A dynamic HDAM is matched against learned templates for online early classification on TEP.  
  Support locator: `publisher_abstract_and_case_study`
- `C4` [inferred] Local normalized correlation and medoid-consensus construction are auditable independent choices until full-text acquisition.  
  Support locator: `local_evidence_boundary`

## Reproduction status

Source acquisition is `not_openly_downloadable`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
