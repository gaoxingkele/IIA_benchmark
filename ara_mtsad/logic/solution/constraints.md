# Constraints

## In scope

- Pinning the five benchmark payloads by bytes and checksums.
- Transcribing published result tables with their object references.
- One harness that varies layout, threshold rule and point adjustment explicitly.
- Re-running five classical detectors and two deep detectors on the pinned payloads.

## Out of scope

- Re-implementing every registered SOTA detector. TimesNet, iTransformer, CATCH,
  DCdetector, GCAD and SensitiveHUE are registered with their reported numbers but
  were not re-run in this iteration; no claim here is based on a re-run of them.
- Reproducing the original data owners' preprocessing scripts. The redistributed
  benchmark payload is used as shipped, and the substitution is recorded.
- Any claim about the datasets' underlying physical systems.
- SWaT deep-detector runs: the pinned split has roughly 495000 training windows at
  stride 1, which did not fit this iteration's compute budget. Recorded as not run
  rather than approximated.

## Known limitations

- USAD is reconstructed from the published algorithm; its exact batching was not
  re-read from the authors' script, so its numbers are a re-run of the algorithm,
  not of the released artefact.
- The Anomaly Transformer follows the reference implementation but re-derives the
  second minimax term from a fresh forward pass (modern autograd rejects the
  original single-graph ordering); the objective is unchanged, the gradient path
  is not bit-identical.
- Oracle thresholds (`best_f1`) use test labels and are therefore upper bounds,
  reported to expose protocol sensitivity rather than as deployable performance.
- The classical detectors are instantaneous; their scores do not use temporal
  context, which is a deliberate contrast to the windowed detectors and not a
  claim that temporal context is unnecessary.

## Ethical / licensing

SWaT payloads are redistributed under the benchmark mirror's terms while the
canonical release is gated; the config records this and the payload must be
re-checked before any redistribution. Papers whose publishers block automated
download are recorded as unavailable rather than fetched around access controls.
