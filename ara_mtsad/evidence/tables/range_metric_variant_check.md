# Range-metric variant check

## What was compared

Two definitions travel under the same "range-based" label:

| Variant | Definition used | Implementation |
|---|---|---|
| flat range-based (`alpha = 0`) | precision of a predicted range is its best overlap fraction with a real range; recall of a real range is the fraction covered by predicted ranges | this repository: `range_based_metrics(bias="flat")` |
| affiliation | every point inside a predicted range keeps partial positional credit towards its affiliated real range | bundled with the DCdetector release: `metrics/affiliation/pr_from_events` |

## Measurement

`scripts/mtsad/verify_range_metrics.py` fetches the reference implementation from
`DAMO-DI-ML/KDD2023-DCdetector@main` (never vendored into this repository), draws
40 random binary series of length 200-3000 with about 5 percent labelled and 4
percent predicted density, and scores both variants on the same series.

| Metric | local flat range-based | reference affiliation | gap |
|---|---|---|---|
| mean precision | 0.0438 | 0.4792 | 0.4354 |
| mean recall | 0.0347 | 0.3263 | 0.2916 |
| worst precision gap over the 40 trials | - | - | 0.5268 |
| worst recall gap over the 40 trials | - | - | 0.4900 |

Full output: `range_metric_variant_check.json` in this directory.

## Consequence

The two variants differ by an order of magnitude on sparse predictions, so a
"range-based F1" column is not comparable across papers unless the variant is
named. This artifact reports the strict flat variant, which is the conservative
choice: its numbers cannot be explained away as the affiliation variant's partial
credit. Reading the affiliation variant through this repository's tables would
understate the published scores, and reading the flat variant through a paper that
used affiliation would overstate the gap - both errors are avoided by naming the
variant, which is why the harness records the metric family alongside the
threshold rule and the point-adjustment flag.

## Published corroboration

The split is visible in the literature itself, not only in this controlled check.
CATCH's Table 2 reports the affiliation family for the same five datasets: for
SMAP it prints Anomaly Transformer at 0.703 Aff-F, while this artifact's flat
range-based column gives Anomaly Transformer 0.0291 on SMAP at the reference
threshold. The two numbers describe the same dataset and the same model family
and differ by 0.67. Because the preprocessing and threshold also differ between
the two setups, that gap is corroboration rather than a controlled measurement -
the controlled measurement is the 40-trial table above. The transcription lives
in `knowledge_base/literature/mtsad_reported_affiliation.json`, deliberately
separate from the point-adjusted file.
