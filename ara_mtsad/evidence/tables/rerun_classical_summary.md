# Classical detectors re-run on the pinned MTSAD payloads

Generated 2026-09-24 from 75 run records.
Every number below is produced by `scripts/mtsad/run_reproduction.py`; the
protocol named in each row is the one declared in
`configs/experiments/mtsad_reproduction.json`.

**Point-adjusted F1 (reference protocol: percentile threshold + point adjustment)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| PCA | 0.7339 ± 0.0000 | 0.7004 ± 0.0000 | 0.6973 ± 0.0000 | 0.8226 ± 0.0000 | 0.8791 ± 0.0000 |
| Mahalanobis | 0.7850 ± 0.0000 | 0.7000 ± 0.0000 | 0.7059 ± 0.0000 | 0.8204 ± 0.0000 | 0.8853 ± 0.0000 |
| kNN | 0.7937 ± 0.0030 | 0.3704 ± 0.0910 | 0.6960 ± 0.0010 | 0.8221 ± 0.0000 | 0.9052 ± 0.0146 |
| IsolationForest | 0.6049 ± 0.0177 | 0.8613 ± 0.0215 | 0.6896 ± 0.0025 | 0.7932 ± 0.0000 | 0.7429 ± 0.0053 |
| OCSVM | 0.7509 ± 0.0315 | 0.7059 ± 0.0071 | 0.6981 ± 0.0083 | 0.8221 ± 0.0000 | 0.9147 ± 0.0001 |

**Point-wise F1 at the same threshold (no point adjustment)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| PCA | 0.0740 ± 0.0000 | 0.0567 ± 0.0000 | 0.0060 ± 0.0000 | 0.2896 ± 0.0000 | 0.0357 ± 0.0000 |
| Mahalanobis | 0.0787 ± 0.0000 | 0.0585 ± 0.0000 | 0.0072 ± 0.0000 | 0.2865 ± 0.0000 | 0.0436 ± 0.0000 |
| kNN | 0.1415 ± 0.0113 | 0.0748 ± 0.0005 | 0.0099 ± 0.0003 | 0.2894 ± 0.0002 | 0.0843 ± 0.0002 |
| IsolationForest | 0.1721 ± 0.0107 | 0.0349 ± 0.0047 | 0.0074 ± 0.0001 | 0.2352 ± 0.0070 | 0.0280 ± 0.0003 |
| OCSVM | 0.0856 ± 0.0122 | 0.0569 ± 0.0005 | 0.0073 ± 0.0003 | 0.2893 ± 0.0004 | 0.0765 ± 0.0007 |

**Range-based F1 (alpha = 0, flat bias)**

| Method | SMD | MSL | SMAP | SWaT | PSM |
|---|---|---|---|---|---|
| PCA | 0.1940 ± 0.0000 | 0.0523 ± 0.0000 | 0.0086 ± 0.0000 | 0.1810 ± 0.0000 | 0.0578 ± 0.0000 |
| Mahalanobis | 0.1879 ± 0.0000 | 0.0524 ± 0.0000 | 0.0080 ± 0.0000 | 0.2407 ± 0.0000 | 0.2008 ± 0.0000 |
| kNN | 0.1595 ± 0.0023 | 0.0694 ± 0.0007 | 0.0082 ± 0.0004 | 0.2314 ± 0.0000 | 0.1890 ± 0.0485 |
| IsolationForest | 0.0995 ± 0.0032 | 0.0117 ± 0.0014 | 0.0098 ± 0.0009 | 0.0016 ± 0.0008 | 0.0515 ± 0.0013 |
| OCSVM | 0.1713 ± 0.0012 | 0.0527 ± 0.0004 | 0.0101 ± 0.0010 | 0.2314 ± 0.0000 | 0.3076 ± 0.0002 |

**Same detector, four protocols (headline F1 per protocol)**

| Dataset | Method | best_f1_adjusted | best_f1_pointwise | per_point_adjusted | tslib_reference |
|---|---|---|---|---|---|
| MSL | IsolationForest | 0.3782 | 0.2290 | 0.8779 | 0.8779 |
| MSL | kNN | 0.6144 | 0.2416 | 0.3722 | 0.3722 |
| MSL | Mahalanobis | 0.4662 | 0.2432 | 0.7000 | 0.7000 |
| MSL | OCSVM | 0.2696 | 0.1952 | 0.7015 | 0.7015 |
| MSL | PCA | 0.4778 | 0.2072 | 0.7004 | 0.7004 |
| PSM | IsolationForest | 0.7649 | 0.5200 | 0.7392 | 0.7392 |
| PSM | kNN | 0.5840 | 0.5426 | 0.9155 | 0.9155 |
| PSM | Mahalanobis | 0.6750 | 0.6163 | 0.8853 | 0.8853 |
| PSM | OCSVM | 0.5486 | 0.5169 | 0.9146 | 0.9146 |
| PSM | PCA | 0.6621 | 0.6150 | 0.8791 | 0.8791 |
| SMAP | IsolationForest | 0.3468 | 0.2960 | 0.6905 | 0.6905 |
| SMAP | kNN | 0.3908 | 0.2931 | 0.6946 | 0.6946 |
| SMAP | Mahalanobis | 0.2561 | 0.2323 | 0.7059 | 0.7059 |
| SMAP | OCSVM | 0.2268 | 0.2268 | 0.6919 | 0.6919 |
| SMAP | PCA | 0.2522 | 0.2318 | 0.6973 | 0.6973 |
| SMD | IsolationForest | 0.6453 | 0.2250 | 0.6298 | 0.6298 |
| SMD | kNN | 0.6550 | 0.2637 | 0.7979 | 0.7979 |
| SMD | Mahalanobis | 0.7305 | 0.2586 | 0.7850 | 0.7850 |
| SMD | OCSVM | 0.7042 | 0.2479 | 0.7730 | 0.7730 |
| SMD | PCA | 0.6525 | 0.2080 | 0.7339 | 0.7339 |
| SWAT | IsolationForest | 0.8524 | 0.7515 | 0.7932 | 0.7932 |
| SWAT | kNN | 0.8468 | 0.7803 | 0.8221 | 0.8221 |
| SWAT | Mahalanobis | 0.8299 | 0.7729 | 0.8204 | 0.8204 |
| SWAT | OCSVM | 0.8532 | 0.7753 | 0.8221 | 0.8221 |
| SWAT | PCA | 0.8361 | 0.7692 | 0.8226 | 0.8226 |

**Re-run (reference protocol) versus published numbers**

| Method | Dataset | Re-run adjusted F1 | Published F1 | Source |
|---|---|---|---|---|
| AnomalyTransformer | MSL | not re-run | 0.9393 | xu2022_anomaly_transformer |
| AnomalyTransformer | PSM | not re-run | 0.9737 | xu2022_anomaly_transformer |
| AnomalyTransformer | SMAP | not re-run | 0.9641 | xu2022_anomaly_transformer |
| AnomalyTransformer | SMD | not re-run | 0.9033 | xu2022_anomaly_transformer |
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
| IsolationForest | MSL | 0.8613 ± 0.0215 | 0.6645 | registered paper table |
| IsolationForest | PSM | 0.7429 ± 0.0053 | 0.8348 | registered paper table |
| IsolationForest | SMAP | 0.6896 ± 0.0025 | 0.5553 | registered paper table |
| IsolationForest | SMD | 0.6049 ± 0.0177 | 0.5364 | registered paper table |
| IsolationForest | SWaT | 0.7932 ± 0.0000 | 0.4702 | registered paper table |
| lstm_vae | MSL | not re-run | 0.8262 | registered paper table |
| lstm_vae | PSM | not re-run | 0.8096 | registered paper table |
| lstm_vae | SMAP | not re-run | 0.7810 | registered paper table |
| lstm_vae | SMD | not re-run | 0.8230 | registered paper table |
| lstm_vae | SWaT | not re-run | 0.8220 | registered paper table |
| OCSVM | MSL | 0.7059 ± 0.0071 | 0.7082 | registered paper table |
| OCSVM | PSM | 0.9147 ± 0.0001 | 0.7067 | registered paper table |
| OCSVM | SMAP | 0.6981 ± 0.0083 | 0.5634 | registered paper table |
| OCSVM | SMD | 0.7509 ± 0.0315 | 0.5619 | registered paper table |
| OCSVM | SWaT | 0.8221 ± 0.0000 | 0.4723 | registered paper table |
| omnianomaly | MSL | not re-run | 0.8765 | registered paper table |
| omnianomaly | PSM | not re-run | 0.8083 | registered paper table |
| omnianomaly | SMAP | not re-run | 0.8728 | registered paper table |
| omnianomaly | SMD | not re-run | 0.9401 | registered paper table |
| omnianomaly | SWaT | not re-run | 0.8131 | registered paper table |
| tranad | MSL | not re-run | 0.9494 | registered paper table |
| tranad | SMAP | not re-run | 0.8915 | registered paper table |
| tranad | SMD | not re-run | 0.9605 | registered paper table |
| tranad | SWaT | not re-run | 0.8151 | registered paper table |
| USAD | MSL | not re-run | 0.8822 | registered paper table |
| USAD | SMAP | not re-run | 0.8419 | registered paper table |
| USAD | SMD | not re-run | 0.9495 | registered paper table |
| USAD | SWaT | not re-run | 0.8143 | registered paper table |
