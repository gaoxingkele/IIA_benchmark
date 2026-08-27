# Early Classification of Industrial Alarm Floods Using a Hybrid Neural Network and Optimal Time-Encoded Histograms

## Metadata

- Year: 2026
- Venue: Engineering Applications of Artificial Intelligence
- DOI: [10.1016/j.engappai.2025.113705](https://doi.org/10.1016/j.engappai.2025.113705)
- Benchmark role: SOTA early alarm-flood classification without chattering preprocessing
- ARA status: `engineering_validated_source_gated`

## Executive summary

Encodes variable-length raw alarm floods as exponentially attenuated fixed histograms, learns the attenuation by gradient optimization, compresses correlated alarms in an autoencoder latent space, and classifies with a modified Transformer using separate pretraining followed by joint fine-tuning.

## Claims and evidence

- `C1` [explicit] The histogram combines alarm frequency with exponential temporal attenuation and optimizes the attenuation by gradients.  
  Support locator: `publisher_contribution_snippet`
- `C2` [explicit] The hybrid independently trains an autoencoder and modified Transformer before joint fine-tuning.  
  Support locator: `publisher_contribution_snippet`
- `C3` [explicit] The method operates without delay-timer chattering removal and is evaluated on TEP.  
  Support locator: `publisher_abstract_and_experiment_snippet`
- `C4` [inferred] Local tokenization, Transformer modification, layer widths and loss weights remain independent choices until full-text acquisition.  
  Support locator: `local_evidence_boundary`

## Reproduction status

Source acquisition is `not_openly_downloadable`. This ARA package does not claim a
score reproduction until `evidence/runs/local_validation.md` records a command,
environment, dataset split, expected result, observed result, and pass/fail decision.
