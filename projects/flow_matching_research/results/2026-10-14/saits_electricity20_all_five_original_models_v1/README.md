SAITS / Electricity / nominal20% missing: all five original seeds26..30.

All five 11511000-parameter models completed the original max10000/patience30 protocol and passed full native checkpoint reinference plus independent review of all290x100x370 cells,1944331 held-out coordinates and10 months. Current full input/artifact/proof hashes were rechecked. No new fitting or inference was performed for this aggregation.

|Metric|Five-model mean|SampleSTD|Student-t95% interval (df4)|
|---|---|---|---|
|mae|0.834772289|0.044851954|[0.779081232, 0.890463345]|
|rmse|1.27350981|0.0515975038|[1.20944305, 1.33757658]|
|mre|0.446672326|0.0239994958|[0.416873016, 0.476471636]|

Training next_epoch by seed26..30: 418, 429, 388, 407, 456.

These intervals summarize five training models on one frozen split. The ten-month block intervals remain separate within-model temporal sensitivity, recorded per seed in full_five_seed_cohort_summary.json. Actual replacement-mask coverage is18.1205126% of all grid cells. Original prediction arrays were not retained, so metric agreement is certified at the original2e-6 tolerance rather than bitwise source-prediction equivalence. Historical per-rate tuning/preprocessing conditions are unknown; author equivalence and strictTAB performance are not certified.

The full210 SAITS/BRITS queue and45-paper/681-method-ablation/273-data-task objective remain active. Electricity nominal30% seed30 is independently tracked.

Paper: https://arxiv.org/abs/2202.08516
Data: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
