# Registered SOTA detectors: transcription and re-run status

Every registered SOTA paper now has either a transcribed, runnable detector or a
recorded reason why its numbers cannot be compared. The table below is the
current state, not a plan.

| Paper | Reported numbers | Metric family | Transcribed | Re-run coverage | Verdict |
|---|---|---|---|---|---|
| Anomaly Transformer (ICLR 2022) | Table 1, all five datasets | point-adjusted F1 | yes | all five | SMD matches (delta 0.0001), PSM matches, SMAP near; SWaT and MSL off with the budget audit recorded |
| DCdetector (KDD 2023) | Table 1, all five | point-adjusted F1 | yes | all five | SWaT/SMD/MSL near, PSM near, MSL off; also converted to affiliation for the cross-check |
| TimesNet (ICLR 2023) | Tables 5/9/10, all five | point-adjusted F1 | yes | all five | **all five within 0.068**, two within 0.01 - the cleanest reproduction in the artifact |
| iTransformer (ICLR 2024) | no point-adjusted number in the corpus; CATCH reports it in the affiliation family | affiliation | yes | all five, diagnostic only | two columns agree, three do not; the release publishes a budget for MSL only |
| CATCH (ICLR 2025) | Table 2, 16 methods x 5 datasets | affiliation (Aff-F, A-R) | yes | four of five; SWaT in flight | SMD 0.019, SMAP 0.043, MSL 0.061, PSM 0.106 |
| GCAD (AAAI 2025) | paper table | point-adjusted F1 | yes | queued behind CATCH/SWaT | not yet issued |
| USAD (KDD 2020) | via TranAD Table 2 | F1 | yes | four datasets, labelled ablation | the adversarial term diverged; the reconstruction-only variant is reported and does not match |
| TranAD (VLDB 2022) | Table 2, four datasets | F1 with the release's POT threshold | yes | four datasets | **verdict withheld**: both threshold rules tried, including the release's own SPOT, and neither reproduces the paper; its loader also evaluates one representative series per dataset where the paper reports whole-dataset sizes |

## What the twelve transcribable detectors cost

Each row above is a faithful transcription of a released implementation, not a
configuration change:

| Detector | Source size | Notable transcription work |
|---|---|---|
| Anomaly Transformer | ~2 files | association attention with the Gaussian prior and the minimax objective |
| DCdetector | ~4 files | dual attention, RevIN, multi-scale patching, per-dataset windows |
| TimesNet | ~3 files | FFT period detection, 2D reshaping, inception blocks |
| iTransformer | ~4 files | inverted embedding and the time-axis projection |
| TranAD | ~3 files | two-phase training with focus-score self-conditioning, plus the POT/SPOT threshold |
| CATCH | 10 files, ~43 KB | frequency patching, gumbel-softmax channel mask, cross-channel transformer, dynamical contrastive loss, frequency reconstruction loss, two optimisers |
| GCAD | ~4 files | TSMixer forecaster plus the gradient causality matrix and its relative-deviation score |

## Not transcribed, and why no number is claimed

| Paper | Reason |
|---|---|
| SensitiveHUE (KDD 2024) | the publisher blocks automated download, so even its result table is unavailable; no number of its is transcribed anywhere in this artifact |
| ModernTCN (ICLR 2024) | the OpenReview PDF is blocked; its numbers appear only indirectly, inside CATCH's affiliation table, and are transcribed there rather than claimed as its own |
