# Experiments

## E01 - Re-run the classical detector family under four declared protocols

**Verifies**: C01, C02, C05.

**Setup**: The five benchmark payloads pinned in `configs/datasets/mtsad_*.json`;
one harness (`scripts/mtsad/run_reproduction.py`) with the protocol matrix from
`configs/experiments/mtsad_reproduction.json`; three seeds for stochastic
detectors.

**Procedure**: Standardise with training-split statistics only, build the
dataset's reference window grid, fit each detector, score the test split,
threshold under each declared rule, and record the point-wise, point-adjusted and
range-based metric families side by side.

**Expected outcome**: Adjusted scores exceed point-wise scores for every detector
whose predictions are sparse relative to the labelled segments, and the ordering
of detectors is expected to be sensitive to the protocol axis being varied.

**Evidence**: `evidence/tables/rerun_classical_summary.md`.

## E02 - Quantify the point-adjustment effect directly

**Verifies**: C02.

**Setup**: The per-run records written by E01.

**Procedure**: For each dataset/detector pair, compare adjusted and point-wise
precision, recall and F1 at the identical threshold, and relate the gap to the
number of predicted ranges and the mean labelled segment length.

**Expected outcome**: The gap grows with labelled segment length and shrinks as
the detector's own false-positive count grows.

**Evidence**: `evidence/tables/protocol_sensitivity.md`.

## E03 - Re-run deep detectors through the same harness

**Verifies**: C01.

**Setup**: USAD (reconstructed from the published algorithm) and the Anomaly
Transformer (transcribed from the reference implementation), trained on the same
window grids.

**Procedure**: Train with the declared per-dataset hyperparameters, score both
splits, and evaluate through the identical protocol matrix used for the classical
detectors.

**Expected outcome**: Deep detectors are expected to beat the instantaneous
detectors under the strict point-wise protocol by a margin smaller than the
protocol-induced spread observed in E01.

**Evidence**: `evidence/tables/rerun_deep_summary.md`.

## E04 - Cross-dataset transfer of a single operating point

**Verifies**: C03, C05.

**Setup**: The per-run records of E01/E03 plus the per-dataset anomaly ratio.

**Procedure**: Apply one percentile rule across all datasets and check whether
the resulting recall tracks the datasets' anomaly rates rather than the
detectors' behaviour.

**Expected outcome**: Recall is expected to be driven by anomaly rate and segment
structure, not by detector identity.

**Evidence**: `evidence/tables/protocol_sensitivity.md`.

## E05 - Split-version audit of the SWaT payload

**Verifies**: C04.

**Setup**: The registered papers' dataset tables and the pinned payload files.

**Procedure**: Compare the training-row counts stated in the registered papers
with the row counts of the pinned payloads, and record the disagreement instead
of normalising it away.

**Expected outcome**: The stated counts are expected to disagree; the artifact
must then carry that disagreement into every SWaT comparison.

**Evidence**: `evidence/tables/swat_split_audit.md`.

## E06 - Re-run the strongest registered detector with its own per-dataset budget

**Verifies**: C01, C05.

**Setup**: DCdetector transcribed from its reference implementation, with the
per-dataset window, patch sizes and anomaly ratio taken from the repository's own
scripts (SMD 105/[5,7]/0.6, MSL 90/[3,5]/1.0, SMAP 105/[3,5,7]/0.85,
PSM 60/[1,3,5]/1.0, SWaT 105/[3,5,7]/1.0).

**Procedure**: Train with the reference budget (three epochs), score both splits,
and evaluate through the same protocol matrix used for every other detector, so
the only difference from the other rows is the detector and its declared budget.

**Expected outcome**: The re-run is expected to land near the published DCdetector
row under the reference protocol while its point-wise score stays far lower, which
would confirm that the published ranking is a property of the shared protocol
rather than of the individual method.

**Evidence**: `evidence/tables/rerun_dcdetector_summary.md`.
