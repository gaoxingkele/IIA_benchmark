# Deep detectors re-run on the pinned MTSAD payloads

Generated 2026-09-24 from 6 run records.
Every number below is produced by `scripts/mtsad/run_reproduction.py`; the
protocol named in each row is the one declared in
`configs/experiments/mtsad_reproduction.json`.

**Point-adjusted F1 (reference protocol: percentile threshold + point adjustment)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| USAD | 0.1277 | 0.0942 | 0.4083 | - | 0.7215 |
| AnomalyTransformer | 0.9034 | 0.8490 | - | - | - |

**Point-wise F1 at the same threshold (no point adjustment)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| USAD | 0.0702 | 0.0815 | 0.0415 | - | 0.0513 |
| AnomalyTransformer | 0.0249 | 0.0114 | - | - | - |

**Range-based F1 (alpha = 0, flat bias)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| USAD | 0.0939 | 0.1065 | 0.0985 | - | 0.0789 |
| AnomalyTransformer | 0.0988 | 0.0136 | - | - | - |

**Same detector, four protocols (headline F1 per protocol)**

| Dataset | Method | best_f1_adjusted | best_f1_pointwise | per_point_adjusted | tslib_reference |
|---|---|---|---|---|---|
| MSL | AnomalyTransformer | 0.8368 | 0.0419 | 0.5899 | 0.8490 |
| MSL | USAD | 0.2485 | 0.2381 | 0.1223 | 0.0942 |
| PSM | USAD | 0.6437 | 0.6066 | 0.7289 | 0.7215 |
| SMAP | USAD | 0.2294 | 0.2294 | 0.4305 | 0.4083 |
| SMD | AnomalyTransformer | 0.7871 | 0.0510 | 0.1000 | 0.9034 |
| SMD | USAD | 0.4421 | 0.3013 | 0.1035 | 0.1277 |

**Re-run (reference protocol) versus published numbers**

| Method | Dataset | Re-run adjusted F1 | Published F1 | Source |
|---|---|---|---|---|
| AnomalyTransformer | MSL | 0.8490 | 0.9393 | xu2022_anomaly_transformer |
| AnomalyTransformer | PSM | not re-run | 0.9737 | xu2022_anomaly_transformer |
| AnomalyTransformer | SMAP | not re-run | 0.9641 | xu2022_anomaly_transformer |
| AnomalyTransformer | SMD | 0.9034 | 0.9033 | xu2022_anomaly_transformer |
| AnomalyTransformer | SWaT | not re-run | 0.9422 | xu2022_anomaly_transformer |
| dagmm | MSL | not re-run | 0.8482 | registered paper table |
| dagmm | SMAP | not re-run | 0.8888 | registered paper table |
| dagmm | SMD | not re-run | 0.9491 | registered paper table |
| dagmm | SWaT | not re-run | 0.8128 | registered paper table |
| dcdetector | MSL | not re-run | 0.9660 | registered paper table |
| dcdetector | PSM | not re-run | 0.9794 | registered paper table |
| dcdetector | SMAP | not re-run | 0.9702 | registered paper table |
| dcdetector | SMD | not re-run | 0.8718 | registered paper table |
| dcdetector | SWaT | not re-run | 0.9633 | registered paper table |
| IsolationForest | MSL | not re-run | 0.6645 | registered paper table |
| IsolationForest | PSM | not re-run | 0.8348 | registered paper table |
| IsolationForest | SMAP | not re-run | 0.5553 | registered paper table |
| IsolationForest | SMD | not re-run | 0.5364 | registered paper table |
| IsolationForest | SWaT | not re-run | 0.4702 | registered paper table |
| lstm_vae | MSL | not re-run | 0.8262 | registered paper table |
| lstm_vae | PSM | not re-run | 0.8096 | registered paper table |
| lstm_vae | SMAP | not re-run | 0.7810 | registered paper table |
| lstm_vae | SMD | not re-run | 0.8230 | registered paper table |
| lstm_vae | SWaT | not re-run | 0.8220 | registered paper table |
| OCSVM | MSL | not re-run | 0.7082 | registered paper table |
| OCSVM | PSM | not re-run | 0.7067 | registered paper table |
| OCSVM | SMAP | not re-run | 0.5634 | registered paper table |
| OCSVM | SMD | not re-run | 0.5619 | registered paper table |
| OCSVM | SWaT | not re-run | 0.4723 | registered paper table |
| omnianomaly | MSL | not re-run | 0.8765 | registered paper table |
| omnianomaly | PSM | not re-run | 0.8083 | registered paper table |
| omnianomaly | SMAP | not re-run | 0.8728 | registered paper table |
| omnianomaly | SMD | not re-run | 0.9401 | registered paper table |
| omnianomaly | SWaT | not re-run | 0.8131 | registered paper table |
| tranad | MSL | not re-run | 0.9494 | registered paper table |
| tranad | SMAP | not re-run | 0.8915 | registered paper table |
| tranad | SMD | not re-run | 0.9605 | registered paper table |
| tranad | SWaT | not re-run | 0.8151 | registered paper table |
| USAD | MSL | 0.0942 | 0.8822 | registered paper table |
| USAD | SMAP | 0.4083 | 0.8419 | registered paper table |
| USAD | SMD | 0.1277 | 0.9495 | registered paper table |
| USAD | SWaT | not re-run | 0.8143 | registered paper table |
