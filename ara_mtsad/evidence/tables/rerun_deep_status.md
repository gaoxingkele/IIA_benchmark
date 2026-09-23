# Deep detector status and interpretation

The generated tables in `rerun_deep_summary.md` carry the numbers; this file
records the budget, the deviations and the reading.

## Coverage of this iteration

| Detector | Datasets re-run | Budget | Status |
|---|---|---|---|
| USAD (non-adversarial variant) | PSM, SMD, MSL, SMAP | 20 epochs, batch 256, window 100 | completed |
| Anomaly Transformer | SMD, MSL | 10 epochs, batch 128, window 100 (the paper's own budget) | completed |

Not run and therefore not claimed anywhere in this artifact: the Anomaly
Transformer on PSM, SMAP and SWaT, and every other registered SOTA detector
(DCdetector, TimesNet, iTransformer, CATCH, GCAD, SensitiveHUE). One Anomaly
Transformer epoch on PSM costs on the order of ten minutes on the RTX 3090 used
here because the (batch, heads, window, window) prior tensor is materialised per
layer; MSL alone took 671.9 s wall for ten epochs, and SWaT has roughly 495000
stride-1 training windows per epoch.

## USAD

The paper's adversarial second loss term is unbounded and diverged on the pinned
payloads (dead-end node X4 in `trace/exploration_tree.yaml`), so the run uses the
reconstruction-only variant declared in `configs/models/mtsad_usad.json`. It is an
ablation, not a reproduction of the released artefact.

| Dataset | re-run adjusted F1 | published USAD F1 (TranAD Table 2) |
|---|---|---|
| PSM | 0.7215 | not reported |
| SMD | 0.1277 | 0.9495 |
| MSL | 0.0942 | 0.8822 |
| SMAP | 0.4083 | 0.8419 |

The gap is reported, not tuned away. Candidate explanations, unresolved here: the
disabled adversarial phase, and the fact that the published numbers were produced
by the authors' own harness with its own preprocessing and threshold rule.

## Anomaly Transformer

| Dataset | re-run adjusted F1 | re-run point-wise F1 | published F1 (paper Table 1) | published F1 (DCdetector's re-run) |
|---|---|---|---|---|
| SMD | 0.9034 | 0.0249 | 0.9233 | 0.9033 |
| MSL | 0.8490 | 0.0114 | 0.9359 | 0.9393 |

SMD reproduces the third-party re-run to within 0.0001 and sits 0.0199 below the
method's own paper, while 2.5 percent of timestamps are classified correctly.
MSL does not reproduce: the re-run is 0.087-0.090 below both published values.
The asymmetry is itself informative - the same detector, the same harness and the
same protocol reproduce one dataset and not another, so a per-dataset claim is
required and a single averaged number would hide the failure.
