# Registered SOTA detectors: re-run status

This artifact re-runs the anchor method (Anomaly Transformer), the strongest
registered SOTA on the shared table (DCdetector), a classic deep baseline (USAD)
and five classical detectors. The remaining registered SOTA papers are catalogued
here with what is already pinned and what a re-run would still need.

| Paper | Reported numbers | Metric family | Re-run status | What a re-run needs |
|---|---|---|---|---|
| Anomaly Transformer (ICLR 2022) | transcribed, Table 1 | point-adjusted F1 | **re-run** on all five datasets | - |
| DCdetector (KDD 2023) | transcribed, Table 1 | point-adjusted F1 | **re-run** on all five datasets | - |
| TranAD (VLDB 2022) | transcribed, Table 2 | F1 with the authors' threshold rule | not re-run | a two-phase adversarial transformer transcription; its per-dataset budgets are in the release |
| USAD (KDD 2020) | transcribed via TranAD, Table 2 | F1 | **re-run** as a labelled non-adversarial ablation | the adversarial term diverged; the published recipe itself is still unreproduced |
| CATCH (ICLR 2025) | transcribed, Table 2, 16 methods x 5 datasets | affiliation (Aff-F, A-R) | not re-run | transcription of `ts_benchmark/baselines/catch/CATCH.py` plus the CFM module; per-dataset budgets are pinned from the release's scripts (MSL: cf_dim 32, d_model 256, lr 5e-4, 5 epochs, ratio 5.0; SMD: cf_dim 32, d_model 128, 2 layers, 1 epoch, ratio 5.0), and the data comes from a OneDrive/Baidu mirror rather than the Time-Series-Library payload |
| TimesNet (ICLR 2023) | available in the local PDF | point-adjusted F1 | not re-run | TSLib-style reconstruction harness; the Time-Series-Library scripts give the budgets |
| iTransformer (ICLR 2024) | no point-adjusted number in the corpus; CATCH Table 2 gives the affiliation column | affiliation | **re-run** as a diagnostic on all five datasets and cross-checked in the affiliation family (two columns agree, three do not) | a published point-adjusted reference, if one exists outside this corpus, before it can carry a verdict row |
| GCAD (AAAI 2025) | available in the local PDF | point-adjusted F1 | not re-run | causal-graph predictor transcription |
| SensitiveHUE (KDD 2024) | blocked PDF; numbers not transcribed | unknown | not re-run | the ACM host rejects automated clients, so even the table is unavailable |
| ModernTCN (ICLR 2024) | available via CATCH Table 2 | affiliation | not re-run | modern pure-conv reconstruction harness |

## Why the remaining re-runs are not free

Each row above is a faithful-transcription task, not a configuration change:

* the Anomaly Transformer and DCdetector were transcribed from their released
  code, which is why the harness can attribute their results to a specific set of
  equations; the remaining models have the same requirement;
* CATCH ships inside its own benchmark framework (`ts_benchmark`) with a
  frequency-loss module and a channel-fusion module, and its data distribution is
  a separate mirror, so a re-run has to reconcile the payload with the one pinned
  here before any comparison is meaningful;
* its reported metric is the affiliation family, so even a successful re-run
  cannot be compared with the point-adjusted table without converting one of the
  two families, which this artifact has already shown differ by up to 0.53.

## What is already decided for those re-runs

The harness side needs no new machinery: `run_reproduction.py` takes a model
config, `--save-scores` persists the score series, `compare_with_papers.py` turns
records into a verdict table, and `sweep_thresholds.py` separates threshold
placement from score quality. Adding a detector is therefore a transcription task
plus a config, and the verdict table updates itself.
