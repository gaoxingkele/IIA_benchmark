# Related Work

Typed dependency graph for this artifact. Every entry resolves to
`papers/literature/mtsad_registry.json`, which carries the access status and the
local PDF when one exists.

## RW1 - xu2022_anomaly_transformer (ICLR 2022) - *baseline* and *bounds*

Defines the five-dataset protocol used as the comparison anchor here, and the
association-discrepancy energy that the transcribed detector implements. Its
Table 1 supplies the reported rows transcribed into the evidence layer. Bounds
this study: the numbers this artifact compares against are those of the anchor
paper, not a neutral third-party measurement.

## RW2 - yang2023_dcdetector (KDD 2023) - *refutes* (the stability of the anchor ranking)

Re-runs the anchor method on the same dataset names and reports a different score
for it, which is direct evidence that the anchor ranking is protocol-relative.

## RW3 - tuli2022_tranad (VLDB 2022) - *extends* (the comparison table)

Adds a transformer detector with adversarial self-conditioning and, importantly,
its own dataset table - the source of the SWaT row-count disagreement audited in
E05. Its table also supplies USAD, OmniAnomaly and DAGMM baselines used here.

## RW4 - audibert2020_usad (KDD 2020) - *baseline*

The two-decoder adversarial autoencoder whose algorithm is reconstructed in
`USADDetector`; its published numbers enter the comparison only through RW3's
reproduction of them.

## RW5 - su2019_omnianomaly (KDD 2019) - *imports* (the SMD payload)

Origin of the Server Machine Dataset. Reconstructed quantities in this artifact
are not inherited from this paper; it is cited for the data.

## RW6 - hundman2018_telemanom (KDD 2018) - *imports* (the SMAP/MSL payload)

Origin of the labelled spacecraft channel subset; its dynamic-thresholding rule
is a different threshold family from the percentile rule used here and is
recorded as out of scope.

## RW7 - abdulaal2021_psm (KDD 2021) - *imports* (the PSM payload)

Origin of the eBay server-metric payload, whose published anomaly rate is
reproduced by the pinned file in this artifact.

## RW8 - goh2016_swat_dataset (CRITIS 2016) - *imports* (the SWaT payload)

Origin of the water-treatment testbed data. Gated; the payload used here is the
redistribution, and that substitution is recorded in the dataset config.

## RW9 - zhang2019_mscred (AAAI 2019), wu2023_timesnet (ICLR 2023), liu2024_itransformer (ICLR 2024), xie2024_catch (ICLR 2025), zhang2025_gcad (AAAI 2025) - *baseline*

Registered SOTA detectors on this family. Not re-implemented in this iteration;
their reported numbers are available in the local PDFs and are the natural next
re-run targets.

## RW10 - liu2024_tsb_ad (NeurIPS 2024 D&B), kim2025_tab (VLDB 2025) - *bounds*

Independent benchmark efforts that document the protocol sensitivity this
artifact measures directly, and provide third-party re-evaluations of the same
detectors. They bound any conclusion drawn from a single paper's table.

## RW11 - ritter2024_overestimation, qi2026_revisiting_omnianomaly - *bounds* (domain-specific re-evaluations)

Spacecraft-telemetry and SMD-specific re-evaluations showing that reported
advantages shrink under stricter protocols; they corroborate C02 from outside
this artifact's own harness.
