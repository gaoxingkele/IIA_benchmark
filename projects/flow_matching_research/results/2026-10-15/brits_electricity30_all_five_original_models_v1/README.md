BRITS / Electricity / nominal30% missing: all five original seeds26..30.

All five 6993328-parameter models completed the original max10000/patience30 protocol and passed full native checkpoint reinference plus independent review of all290x100x370 cells,2781282 held-out coordinates and10 months. Current full input/artifact/proof hashes were rechecked. No new fitting or inference was performed for this aggregation.

|Metric|Five-model mean|SampleSTD|Student-t95% interval (df4)|
|---|---|---|---|
|mae|0.987137127|0.00992222951|[0.974817052, 0.999457202]|
|rmse|1.46698151|0.00908261773|[1.45570395, 1.47825907]|
|mre|0.528334582|0.00531059051|[0.521740613, 0.534928551]|

Training next_epoch by seed26..30: 732, 940, 964, 995, 957.

These intervals summarize five training models on one frozen split. The ten-month block intervals remain separate within-model temporal sensitivity, recorded per seed in full_five_seed_cohort_summary.json. Actual replacement-mask coverage is25.9206151% of all grid cells. Original prediction arrays were not retained, so metric agreement is certified at the original2e-6 tolerance rather than bitwise source-prediction equivalence. Historical per-rate tuning/preprocessing conditions are unknown; author equivalence and strictTAB performance are not certified.

The full210 SAITS/BRITS queue and45-paper/681-method-ablation/273-data-task objective remain active. All five Electricity nominal30% seeds26..30 are now closed; the remaining original slots stay pending.

Paper: https://arxiv.org/abs/2202.08516
Data: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

49 relevant checks passed. All five actual metric vectors were additionally checked using math.fsum/scalar sampleSTD and SciPy Student-t df4 intervals at1e-12. Total fully reviewed SAITS/BRITS models:39/210.
