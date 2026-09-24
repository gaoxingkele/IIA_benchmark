# AUC-family check (GCAD)

## Why a third metric family appears

The artifact's headline comparisons are point-adjusted F1, and the cross-family
check with CATCH uses the affiliation family. GCAD's Table 2 reports neither: it
prints **AUROC and AUPRC** on the five datasets. Those metrics are threshold-free,
so for GCAD the comparison contains no threshold rule at all - it compares the
ranking a model produces. `scripts/mtsad/score_auc.py` computes them from the
persisted score series.

## Result

| Dataset | Re-run AUROC | Published AUROC | Delta | Re-run AUPRC | Published AUPRC | Delta |
|---|---|---|---|---|---|---|
| SMD | 0.7035 | 0.9533 | 0.2498 | 0.0930 | 0.7502 | 0.6572 |
| PSM | 0.6549 | 0.7618 | 0.1069 | 0.4340 | 0.6136 | 0.1796 |
| MSL | 0.5634 | 0.7658 | 0.2024 | 0.1481 | 0.3679 | 0.2198 |
| SMAP | 0.4948 | 0.7273 | 0.2325 | 0.1368 | 0.4555 | 0.3187 |
| SWaT | pending | 0.8690 | - | pending | 0.7758 | - |

## What was ruled out, and what was not

The first transcription had the wrong forecast target: it fed the model a window
shortened by `pred_len` and asked it to reproduce that window's own tail, where
the release feeds the whole window and predicts the steps that follow it. That was
corrected (`target_mode: next_steps`, recoverable from overlapping windows for
every dataset whose stride is 1; SMD keeps `window_tail` and carries the
deviation), and the corrected runs are the ones in the table above.

Correcting it did **not** close the gap: MSL moved from 0.5790 to 0.5634 AUROC and
PSM from 0.6437 to 0.6549, so the target formulation was a real defect but not the
explanation. What remains, recorded rather than tuned away:

1. **Training budget.** The release trains with `train_epochs 100` and early
   stopping (patience 2) against its own validation split; this harness trains a
   fixed 10 epochs on the training split.
2. **Test stride.** The release samples test windows with stride 5 (SWaT) and 10
   (PSM); the harness uses one stride for training and testing.
3. **Causality-matrix sampling.** The release samples a fraction `sample_p` of the
   training batches when building the reference matrix; this harness samples
   stochastically with the same rate, but the batch order differs.
4. **Preprocessing.** The release's dataloader also selects a `target_slice`;
   this harness assumes all channels.

AUC is the strictest of the three families for this claim: it needs no threshold,
so "the threshold rule was wrong" cannot be the explanation, and the ranking
itself is what falls short. GCAD therefore joins TranAD in the set of registrations
whose published recipe could not be matched end to end, and its verdict is
withheld for the same reason - the remaining deviations are documented and the
gap is reported rather than closed by tuning.

One row sharpens the reading: SMAP lands at 0.4948 AUROC, i.e. a coin flip, on a
dataset where the same harness reproduces TimesNet, the Anomaly Transformer and
DCdetector within a few points. A chance-level ranking on one dataset with a
documented target deviation is more consistent with the causal score not being
comparable at that window length than with a general harness fault, but the
artifact does not claim which, and the row is left as an open discrepancy.
