# Online Alarm Flood Classification via Deterministic Fingerprinting with Combinatorial Hashing: A Robust and Scalable Framework

## Metadata

- Year: 2025
- Venue: Chemical Engineering Research and Design
- DOI: [10.1016/j.cherd.2025.11.026](https://doi.org/10.1016/j.cherd.2025.11.026)
- Benchmark role: SOTA deterministic and scalable online alarm flood classification
- ARA status: `engineering_validated_source_gated`

## Executive summary

Builds Alarm Evolution Matrices from sliding activation rates, extracts local peaks, forms deterministic combinatorial temporal hashes, aggregates class-level consensus fingerprint profiles, and classifies streaming floods with variability-aware evidence.

## Claims and evidence

- `C1` [explicit] The offline pipeline uses AEM, localized peaks, CTFH and category-wise consensus fingerprint profiles.  
  Support locator: `publisher_abstract_and_section_snippets`
- `C2` [explicit] The evaluation uses TEP and missing, false, delayed and reordered alarm perturbations.  
  Support locator: `publisher_abstract`
- `C3` [inferred] The local peak, SHA-256 truncation, time quantization and Jaccard rules are auditable independent choices until full-text equations are acquired.  
  Support locator: `local_evidence_boundary`

## Reproduction status

Source acquisition is `not_openly_downloadable`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
