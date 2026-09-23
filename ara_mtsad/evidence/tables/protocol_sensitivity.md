# Protocol sensitivity

Derived on 2026-09-24 by `scripts/mtsad/derive_protocol_evidence.py`
from 79 run records in `experiments/runs/mtsad_reproduction/`.
Each detector is scored once per run; the rows below differ only in the
declared protocol, so the spread is attributable to the protocol.

## Adjusted versus point-wise F1 (reference threshold rule)

| Dataset | Method | adjusted F1 | point-wise F1 | range F1 | ratio adj/point |
|---|---|---|---|---|---|
| MSL | IsolationForest | 0.8613 | 0.0349 | 0.0117 | 24.7 |
| MSL | kNN | 0.3704 | 0.0748 | 0.0694 | 5.0 |
| MSL | Mahalanobis | 0.7000 | 0.0585 | 0.0524 | 12.0 |
| MSL | OCSVM | 0.7059 | 0.0569 | 0.0527 | 12.4 |
| MSL | PCA | 0.7004 | 0.0567 | 0.0523 | 12.4 |
| MSL | USAD (non-adversarial) | 0.0942 | 0.0815 | 0.1065 | 1.2 |
| PSM | IsolationForest | 0.7429 | 0.0280 | 0.0515 | 26.5 |
| PSM | kNN | 0.9052 | 0.0843 | 0.1890 | 10.7 |
| PSM | Mahalanobis | 0.8853 | 0.0436 | 0.2008 | 20.3 |
| PSM | OCSVM | 0.9147 | 0.0765 | 0.3076 | 12.0 |
| PSM | PCA | 0.8791 | 0.0357 | 0.0578 | 24.6 |
| PSM | USAD (non-adversarial) | 0.7215 | 0.0513 | 0.0789 | 14.1 |
| SMAP | IsolationForest | 0.6896 | 0.0074 | 0.0098 | 92.7 |
| SMAP | kNN | 0.6960 | 0.0099 | 0.0082 | 70.6 |
| SMAP | Mahalanobis | 0.7059 | 0.0072 | 0.0080 | 98.6 |
| SMAP | OCSVM | 0.6981 | 0.0073 | 0.0101 | 96.0 |
| SMAP | PCA | 0.6973 | 0.0060 | 0.0086 | 116.0 |
| SMAP | USAD (non-adversarial) | 0.4083 | 0.0415 | 0.0985 | 9.8 |
| SMD | IsolationForest | 0.6049 | 0.1721 | 0.0995 | 3.5 |
| SMD | kNN | 0.7937 | 0.1415 | 0.1595 | 5.6 |
| SMD | Mahalanobis | 0.7850 | 0.0787 | 0.1879 | 10.0 |
| SMD | OCSVM | 0.7509 | 0.0856 | 0.1713 | 8.8 |
| SMD | PCA | 0.7339 | 0.0740 | 0.1940 | 9.9 |
| SMD | USAD (non-adversarial) | 0.1277 | 0.0702 | 0.0939 | 1.8 |
| SWAT | IsolationForest | 0.7932 | 0.2352 | 0.0016 | 3.4 |
| SWAT | kNN | 0.8221 | 0.2894 | 0.2314 | 2.8 |
| SWAT | Mahalanobis | 0.8204 | 0.2865 | 0.2407 | 2.9 |
| SWAT | OCSVM | 0.8221 | 0.2893 | 0.2314 | 2.8 |
| SWAT | PCA | 0.8226 | 0.2896 | 0.1810 | 2.8 |

## Threshold rule sensitivity (point adjustment on in both columns)

| Dataset | Method | percentile threshold adjusted F1 | oracle threshold adjusted F1 |
|---|---|---|---|
| MSL | IsolationForest | 0.8613 | 0.3773 |
| MSL | kNN | 0.3704 | 0.5661 |
| MSL | Mahalanobis | 0.7000 | 0.4662 |
| MSL | OCSVM | 0.7059 | 0.2753 |
| MSL | PCA | 0.7004 | 0.4778 |
| MSL | USAD (non-adversarial) | 0.0942 | 0.2485 |
| PSM | IsolationForest | 0.7429 | 0.7548 |
| PSM | kNN | 0.9052 | 0.5883 |
| PSM | Mahalanobis | 0.8853 | 0.6750 |
| PSM | OCSVM | 0.9147 | 0.5425 |
| PSM | PCA | 0.8791 | 0.6621 |
| PSM | USAD (non-adversarial) | 0.7215 | 0.6437 |
| SMAP | IsolationForest | 0.6896 | 0.3665 |
| SMAP | kNN | 0.6960 | 0.3870 |
| SMAP | Mahalanobis | 0.7059 | 0.2561 |
| SMAP | OCSVM | 0.6981 | 0.2269 |
| SMAP | PCA | 0.6973 | 0.2522 |
| SMAP | USAD (non-adversarial) | 0.4083 | 0.2294 |
| SMD | IsolationForest | 0.6049 | 0.6457 |
| SMD | kNN | 0.7937 | 0.6918 |
| SMD | Mahalanobis | 0.7850 | 0.7305 |
| SMD | OCSVM | 0.7509 | 0.7053 |
| SMD | PCA | 0.7339 | 0.6525 |
| SMD | USAD (non-adversarial) | 0.1277 | 0.4421 |
| SWAT | IsolationForest | 0.7932 | 0.8493 |
| SWAT | kNN | 0.8221 | 0.8479 |
| SWAT | Mahalanobis | 0.8204 | 0.8299 |
| SWAT | OCSVM | 0.8221 | 0.8521 |
| SWAT | PCA | 0.8226 | 0.8361 |

## Layout sensitivity for windowed detectors

| Dataset | Method | window_flatten adjusted F1 | last_point adjusted F1 |
|---|---|---|---|
| MSL | IsolationForest | 0.3773 | 0.8613 |
| MSL | kNN | 0.5661 | 0.3704 |
| MSL | Mahalanobis | 0.4662 | 0.7000 |
| MSL | OCSVM | 0.2753 | 0.7059 |
| MSL | PCA | 0.4778 | 0.7004 |
| MSL | USAD (non-adversarial) | 0.2485 | 0.1223 |
| PSM | IsolationForest | 0.7548 | 0.7429 |
| PSM | kNN | 0.5883 | 0.9052 |
| PSM | Mahalanobis | 0.6750 | 0.8853 |
| PSM | OCSVM | 0.5425 | 0.9147 |
| PSM | PCA | 0.6621 | 0.8791 |
| PSM | USAD (non-adversarial) | 0.6437 | 0.7289 |
| SMAP | IsolationForest | 0.3665 | 0.6896 |
| SMAP | kNN | 0.3870 | 0.6960 |
| SMAP | Mahalanobis | 0.2561 | 0.7059 |
| SMAP | OCSVM | 0.2269 | 0.6981 |
| SMAP | PCA | 0.2522 | 0.6973 |
| SMAP | USAD (non-adversarial) | 0.2294 | 0.4305 |
| SMD | IsolationForest | 0.6457 | 0.6049 |
| SMD | kNN | 0.6918 | 0.7937 |
| SMD | Mahalanobis | 0.7305 | 0.7850 |
| SMD | OCSVM | 0.7053 | 0.7509 |
| SMD | PCA | 0.6525 | 0.7339 |
| SMD | USAD (non-adversarial) | 0.4421 | 0.1035 |
| SWAT | IsolationForest | 0.8493 | 0.7932 |
| SWAT | kNN | 0.8479 | 0.8221 |
| SWAT | Mahalanobis | 0.8299 | 0.8204 |
| SWAT | OCSVM | 0.8521 | 0.8221 |
| SWAT | PCA | 0.8361 | 0.8226 |

## Cross-dataset operating point (PCA, reference protocol recall)

| Dataset | labelled anomaly rate | recall |
|---|---|---|
| MSL | 0.1053 | 0.0301 |
| PSM | 0.2776 | 0.0183 |
| SMAP | 0.1279 | 0.0031 |
| SMD | 0.0416 | 0.0450 |
| SWAT | 0.1214 | 0.1697 |
