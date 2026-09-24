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
unstable at that scale.

### Correction: the release's own SPOT was then run instead

The first reading of that failure was that the release normalises its scores
before thresholding. Inspecting `main.py` shows it does **not**: it passes the raw
per-timestep loss to `pot_eval` and averages over features for the dataset-level
result. The failure was therefore in this artifact's implementation of the
algorithm, not in a missing normalisation step.

The release vendors SPOT as `src/spot.py`, so the honest fix is to call *their*
code. `tmp/mtsad_recon/pot_reference_spots.py` (a scratch script, since the file
is third-party) imports their `SPOT` and reproduces `pot.py` line for line -
`SPOT(q)`, `fit(train, test)`, `initialize(level=lm_d[0], min_extrema=False)`,
`run(dynamic=False)`, `mean(thresholds) * lm_d[1]` - on this artifact's persisted
scores:

| Dataset | SPOT threshold | test max | SPOT-thresholded F1-PA | reference-protocol F1-PA | published F1 |
|---|---|---|---|---|---|
| MSL | 41.03 | 1469.72 | 0.3929 | 0.6947 | 0.9494 |
| SMAP | 5.00 | 5497.21 | 0.6885 | 0.5898 | 0.8915 |
| SWaT | 7.84 | 94.82 | 0.5971 | 0.7918 | 0.8151 |
| SMD | SPOT raised on an empty exceedance set at level 0.99995 | 14193.95 | run failed | 0.1470 | 0.9605 |

This is the decisive comparison: with the release's own threshold code, the gap
to the published numbers does not close - it is 0.557 on MSL, 0.203 on SMAP and
0.218 on SWaT, and SMD cannot even be thresholded at the released `level`. Two
threshold rules were tried and neither reproduces the paper, so the threshold
mechanism is ruled out as the explanation.

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
end-to-end. The model was transcribed, both plausible threshold rules were tried
(this artifact's percentile rule and the release's own SPOT), and neither
reproduces the paper. What remains is the data span and the fit: the release's
`load_dataset` evaluates a single representative series per dataset while its own
Table 2 lists whole-dataset sizes, and `params.json` keys the threshold parameters
by file prefix, so the released pipeline that produced the 0.95/0.89/0.96 rows is
not the pipeline these re-runs reproduce. The verdict is withheld and the row is
reported as a diagnostic, with both threshold rules published next to it so the
reader can see that the choice of rule was not what decided the outcome.
