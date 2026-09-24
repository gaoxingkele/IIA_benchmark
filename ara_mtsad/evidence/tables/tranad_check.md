# TranAD check

## What was done

The TranAD module was transcribed from the release (two-phase transformer with
focus-score self-conditioning, `main.py:backprop`'s window construction and
`(1/n)` / `(1 - 1/n)` phase weighting, AdamW with weight decay and `StepLR(5, 0.9)`),
and re-run with score caching on the four datasets the paper reports: SMAP, MSL,
SWaT and SMD. PSM is excluded because TranAD does not evaluate it.

## Reference-protocol result versus the published F1

| Dataset | Re-run F1-PA | Published F1 | Delta | Verdict |
|---|---|---|---|---|
| SWaT | 0.7918 | 0.8151 | 0.0233 | near |
| MSL | 0.6947 | 0.9494 | 0.2547 | off |
| SMAP | 0.5898 | 0.8915 | 0.3017 | off |
| SMD | 0.1470 | 0.9605 | 0.8135 | off |

SMD's gap is the largest in the artifact and is not a small calibration issue:
under this harness TranAD barely detects SMD at all while the paper reports 0.96.

## The POT route, and why it was not used for the verdict

TranAD does not threshold with a percentile: `src/pot.py` fits a `SPOT` object on
the training scores and takes `mean(thresholds) * lm_d[1]`, with per-dataset
`lm_d` from `src/params.json`. `scripts/mtsad/score_pot.py` implements the
documented static POT rule on the persisted scores and produced:

| Dataset | POT threshold | training max | test max | POT F1-PA |
|---|---|---|---|---|
| MSL | 2297357 | 1060 | 1470 | 0.0000 |
| SMAP | 102665 | 5497 | 5497 | 0.0000 |
| SWaT | 291 | 80 | 95 | 0.0000 |
| SMD | 14.3 | 72 | 14194 | 0.1832 |

On three datasets the threshold lands above the maximum score, i.e. nothing is
detected. The cause is visible in the same table: the initial POT threshold is a
99.9th (MSL) or 2nd (SMAP) percentile of the training scores that sits at 0.005
and 0.017 while the maxima are 1060 and 5497, so the GPD tail extrapolation is
unstable at that scale. The release does not threshold raw errors - it normalises
the score array before calling `pot_eval`, and the `lm_d` values were tuned for
that normalised scale. Reproducing their operating point therefore requires their
normalisation as well, which this iteration did not reimplement; the numbers
above are reported rather than tuned until they look plausible.

## Two structural caveats on this comparison

1. **Representative series.** The release's `load_dataset` evaluates a single
   series per dataset (`machine-1-1_` for SMD, `P-1_` for SMAP, `C-1_` for MSL),
   while its own Table 2 dataset statistics list whole-dataset sizes. This
   artifact evaluates the whole pinned payload. The two are therefore not
   guaranteed to be the same data span, which is a stronger caveat than the
   threshold rule for the SMD row in particular.
2. **Per-file parameters.** `params.json` keys its `lm_d` and `lr_d` by file
   prefix, and SMD lists four machines with different values, so "the TranAD
   operating point" is per series rather than per dataset.

## Reading

TranAD is the one registered SOTA whose published recipe could not be matched
end-to-end in this iteration: the model was transcribed, but its threshold
mechanism depends on a score normalisation the paper does not state and the
release applies implicitly, and its published numbers may be computed on a
different data span than the pinned payload. Its reference-protocol row is
reported as a diagnostic with those two caveats attached, and its verdict is
withheld rather than issued against a protocol the paper does not use.
