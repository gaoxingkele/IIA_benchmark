# Protocol sensitivity

Derived on 2026-09-25 by `scripts/mtsad/derive_protocol_evidence.py`
from 133 run records in `experiments/runs/mtsad_reproduction/`.
Each detector is scored once per run; the rows below differ only in the
declared protocol, so the spread is attributable to the protocol.

## Adjusted versus point-wise F1 (reference threshold rule)

| Dataset | Method | adjusted F1 | point-wise F1 | range F1 | ratio adj/point |
|---|---|---|---|---|---|
| MSL | AnomalyTransformer | 0.8503 | 0.0166 | 0.0164 | 51.2 |
| MSL | catch | 0.7190 | 0.1236 | 0.1318 | 5.8 |
| MSL | dcdetector | 0.8511 | 0.0180 | 0.0392 | 47.2 |
| MSL | gcad | 0.1968 | 0.0572 | 0.0880 | 3.4 |
| MSL | IsolationForest | 0.8613 | 0.0349 | 0.0117 | 24.7 |
| MSL | itransformer | 0.7126 | 0.0736 | 0.0578 | 9.7 |
| MSL | kNN | 0.3704 | 0.0748 | 0.0694 | 5.0 |
| MSL | Mahalanobis | 0.7000 | 0.0585 | 0.0524 | 12.0 |
| MSL | OCSVM | 0.7059 | 0.0569 | 0.0527 | 12.4 |
| MSL | PCA | 0.7004 | 0.0567 | 0.0523 | 12.4 |
| MSL | timesnet | 0.8133 | 0.0561 | 0.0601 | 14.5 |
| MSL | tranad | 0.6947 | 0.0543 | 0.0521 | 12.8 |
| MSL | USAD (non-adversarial) | 0.0942 | 0.0815 | 0.1065 | 1.2 |
| PSM | AnomalyTransformer | 0.9675 | 0.0210 | 0.0524 | 46.0 |
| PSM | catch | 0.9345 | 0.1188 | 0.4121 | 7.9 |
| PSM | dcdetector | 0.9647 | 0.0196 | 0.0155 | 49.2 |
| PSM | gcad | 0.8996 | 0.0443 | 0.1984 | 20.3 |
| PSM | IsolationForest | 0.7429 | 0.0280 | 0.0515 | 26.5 |
| PSM | itransformer | 0.9474 | 0.0463 | 0.3186 | 20.5 |
| PSM | kNN | 0.9052 | 0.0843 | 0.1890 | 10.7 |
| PSM | Mahalanobis | 0.8853 | 0.0436 | 0.2008 | 20.3 |
| PSM | OCSVM | 0.9147 | 0.0765 | 0.3076 | 12.0 |
| PSM | PCA | 0.8791 | 0.0357 | 0.0578 | 24.6 |
| PSM | timesnet | 0.9722 | 0.0495 | 0.3491 | 19.6 |
| PSM | USAD (non-adversarial) | 0.7215 | 0.0513 | 0.0789 | 14.1 |
| SMAP | AnomalyTransformer | 0.9535 | 0.0219 | 0.0291 | 43.6 |
| SMAP | catch | 0.6739 | 0.0383 | 0.0484 | 17.6 |
| SMAP | dcdetector | 0.9414 | 0.0148 | 0.0221 | 63.5 |
| SMAP | gcad | 0.6525 | 0.0360 | 0.0569 | 18.1 |
| SMAP | IsolationForest | 0.6896 | 0.0074 | 0.0098 | 92.7 |
| SMAP | itransformer | 0.6663 | 0.0362 | 0.0432 | 18.4 |
| SMAP | kNN | 0.6960 | 0.0099 | 0.0082 | 70.6 |
| SMAP | Mahalanobis | 0.7059 | 0.0072 | 0.0080 | 98.6 |
| SMAP | OCSVM | 0.6981 | 0.0073 | 0.0101 | 96.0 |
| SMAP | PCA | 0.6973 | 0.0060 | 0.0086 | 116.0 |
| SMAP | timesnet | 0.7357 | 0.0399 | 0.0760 | 18.4 |
| SMAP | tranad | 0.5898 | 0.0075 | 0.0211 | 79.1 |
| SMAP | USAD (non-adversarial) | 0.4083 | 0.0415 | 0.0985 | 9.8 |
| SMD | AnomalyTransformer | 0.8827 | 0.0206 | 0.0656 | 42.9 |
| SMD | catch | 0.5745 | 0.2507 | 0.1312 | 2.3 |
| SMD | dcdetector | 0.8488 | 0.0123 | 0.0241 | 69.0 |
| SMD | gcad | 0.0956 | 0.0677 | 0.0764 | 1.4 |
| SMD | IsolationForest | 0.6049 | 0.1721 | 0.0995 | 3.5 |
| SMD | itransformer | 0.8272 | 0.1045 | 0.1815 | 7.9 |
| SMD | kNN | 0.7937 | 0.1415 | 0.1595 | 5.6 |
| SMD | Mahalanobis | 0.7850 | 0.0787 | 0.1879 | 10.0 |
| SMD | OCSVM | 0.7509 | 0.0856 | 0.1713 | 8.8 |
| SMD | PCA | 0.7339 | 0.0740 | 0.1940 | 9.9 |
| SMD | timesnet | 0.8542 | 0.1107 | 0.2076 | 7.7 |
| SMD | tranad | 0.1470 | 0.0311 | 0.0257 | 4.7 |
| SMD | USAD (non-adversarial) | 0.1277 | 0.0702 | 0.0939 | 1.8 |
| SWAT | AnomalyTransformer | 0.8754 | 0.0137 | 0.0071 | 64.0 |
| SWAT | catch | 0.8492 | 0.0850 | 0.1704 | 10.0 |
| SWAT | dcdetector | 0.9518 | 0.0190 | 0.0246 | 50.0 |
| SWAT | IsolationForest | 0.7932 | 0.2352 | 0.0016 | 3.4 |
| SWAT | itransformer | 0.9116 | 0.0316 | 0.0616 | 28.9 |
| SWAT | kNN | 0.8221 | 0.2894 | 0.2314 | 2.8 |
| SWAT | Mahalanobis | 0.8204 | 0.2865 | 0.2407 | 2.9 |
| SWAT | OCSVM | 0.8221 | 0.2893 | 0.2314 | 2.8 |
| SWAT | PCA | 0.8226 | 0.2896 | 0.1810 | 2.8 |
| SWAT | timesnet | 0.9293 | 0.0792 | 0.1833 | 11.7 |
| SWAT | tranad | 0.7918 | 0.2855 | 0.0047 | 2.8 |

## Threshold rule sensitivity (point adjustment on in both columns)

| Dataset | Method | percentile threshold adjusted F1 | oracle threshold adjusted F1 |
|---|---|---|---|
| MSL | AnomalyTransformer | 0.8503 | 0.7120 |
| MSL | catch | 0.7190 | 0.3447 |
| MSL | dcdetector | 0.8511 | 0.2186 |
| MSL | gcad | 0.1968 | 0.3397 |
| MSL | IsolationForest | 0.8613 | 0.3773 |
| MSL | itransformer | 0.7126 | 0.2871 |
| MSL | kNN | 0.3704 | 0.5661 |
| MSL | Mahalanobis | 0.7000 | 0.4662 |
| MSL | OCSVM | 0.7059 | 0.2753 |
| MSL | PCA | 0.7004 | 0.4778 |
| MSL | timesnet | 0.8133 | 0.2780 |
| MSL | tranad | 0.6947 | 0.3629 |
| MSL | USAD (non-adversarial) | 0.0942 | 0.2485 |
| PSM | AnomalyTransformer | 0.9675 | 0.9530 |
| PSM | catch | 0.9345 | 0.5674 |
| PSM | dcdetector | 0.9647 | 0.4350 |
| PSM | gcad | 0.8996 | 0.5770 |
| PSM | IsolationForest | 0.7429 | 0.7548 |
| PSM | itransformer | 0.9474 | 0.4343 |
| PSM | kNN | 0.9052 | 0.5883 |
| PSM | Mahalanobis | 0.8853 | 0.6750 |
| PSM | OCSVM | 0.9147 | 0.5425 |
| PSM | PCA | 0.8791 | 0.6621 |
| PSM | timesnet | 0.9722 | 0.4343 |
| PSM | USAD (non-adversarial) | 0.7215 | 0.6437 |
| SMAP | AnomalyTransformer | 0.9535 | 0.5642 |
| SMAP | catch | 0.6739 | 0.2642 |
| SMAP | dcdetector | 0.9414 | 0.3283 |
| SMAP | gcad | 0.6525 | 0.2834 |
| SMAP | IsolationForest | 0.6896 | 0.3665 |
| SMAP | itransformer | 0.6663 | 0.2306 |
| SMAP | kNN | 0.6960 | 0.3870 |
| SMAP | Mahalanobis | 0.7059 | 0.2561 |
| SMAP | OCSVM | 0.6981 | 0.2269 |
| SMAP | PCA | 0.6973 | 0.2522 |
| SMAP | timesnet | 0.7357 | 0.2751 |
| SMAP | tranad | 0.5898 | 0.2273 |
| SMAP | USAD (non-adversarial) | 0.4083 | 0.2294 |
| SMD | AnomalyTransformer | 0.8827 | 0.5782 |
| SMD | catch | 0.5745 | 0.5385 |
| SMD | dcdetector | 0.8488 | 0.1031 |
| SMD | gcad | 0.0956 | 0.1885 |
| SMD | IsolationForest | 0.6049 | 0.6457 |
| SMD | itransformer | 0.8272 | 0.5466 |
| SMD | kNN | 0.7937 | 0.6918 |
| SMD | Mahalanobis | 0.7850 | 0.7305 |
| SMD | OCSVM | 0.7509 | 0.7053 |
| SMD | PCA | 0.7339 | 0.6525 |
| SMD | timesnet | 0.8542 | 0.5821 |
| SMD | tranad | 0.1470 | 0.3092 |
| SMD | USAD (non-adversarial) | 0.1277 | 0.4421 |
| SWAT | AnomalyTransformer | 0.8754 | 0.9018 |
| SWAT | catch | 0.8492 | 0.2166 |
| SWAT | dcdetector | 0.9518 | 0.2187 |
| SWAT | IsolationForest | 0.7932 | 0.8493 |
| SWAT | itransformer | 0.9116 | 0.2166 |
| SWAT | kNN | 0.8221 | 0.8479 |
| SWAT | Mahalanobis | 0.8204 | 0.8299 |
| SWAT | OCSVM | 0.8221 | 0.8521 |
| SWAT | PCA | 0.8226 | 0.8361 |
| SWAT | timesnet | 0.9293 | 0.2166 |
| SWAT | tranad | 0.7918 | 0.8183 |

## Layout sensitivity for windowed detectors

| Dataset | Method | window_flatten adjusted F1 | last_point adjusted F1 |
|---|---|---|---|
| MSL | AnomalyTransformer | 0.7120 | 0.7852 |
| MSL | catch | 0.3447 | 0.7248 |
| MSL | dcdetector | 0.2186 | 0.9232 |
| MSL | gcad | 0.3397 | 0.2117 |
| MSL | IsolationForest | 0.3773 | 0.8613 |
| MSL | itransformer | 0.2871 | 0.4692 |
| MSL | kNN | 0.5661 | 0.3704 |
| MSL | Mahalanobis | 0.4662 | 0.7000 |
| MSL | OCSVM | 0.2753 | 0.7059 |
| MSL | PCA | 0.4778 | 0.7004 |
| MSL | timesnet | 0.2780 | 0.8768 |
| MSL | tranad | 0.3629 | 0.8845 |
| MSL | USAD (non-adversarial) | 0.2485 | 0.1223 |
| PSM | AnomalyTransformer | 0.9530 | 0.9652 |
| PSM | catch | 0.5674 | 0.9172 |
| PSM | dcdetector | 0.4350 | 0.9578 |
| PSM | gcad | 0.5770 | 0.9092 |
| PSM | IsolationForest | 0.7548 | 0.7429 |
| PSM | itransformer | 0.4343 | 0.9272 |
| PSM | kNN | 0.5883 | 0.9052 |
| PSM | Mahalanobis | 0.6750 | 0.8853 |
| PSM | OCSVM | 0.5425 | 0.9147 |
| PSM | PCA | 0.6621 | 0.8791 |
| PSM | timesnet | 0.4343 | 0.9230 |
| PSM | USAD (non-adversarial) | 0.6437 | 0.7289 |
| SMAP | AnomalyTransformer | 0.5642 | 0.9376 |
| SMAP | catch | 0.2642 | 0.7433 |
| SMAP | dcdetector | 0.3283 | 0.8791 |
| SMAP | gcad | 0.2834 | 0.6692 |
| SMAP | IsolationForest | 0.3665 | 0.6896 |
| SMAP | itransformer | 0.2306 | 0.6826 |
| SMAP | kNN | 0.3870 | 0.6960 |
| SMAP | Mahalanobis | 0.2561 | 0.7059 |
| SMAP | OCSVM | 0.2269 | 0.6981 |
| SMAP | PCA | 0.2522 | 0.6973 |
| SMAP | timesnet | 0.2751 | 0.7115 |
| SMAP | tranad | 0.2273 | 0.6916 |
| SMAP | USAD (non-adversarial) | 0.2294 | 0.4305 |
| SMD | AnomalyTransformer | 0.5782 | 0.0825 |
| SMD | catch | 0.5385 | 0.4313 |
| SMD | dcdetector | 0.1031 | 0.1698 |
| SMD | gcad | 0.1885 | 0.0958 |
| SMD | IsolationForest | 0.6457 | 0.6049 |
| SMD | itransformer | 0.5466 | 0.2156 |
| SMD | kNN | 0.6918 | 0.7937 |
| SMD | Mahalanobis | 0.7305 | 0.7850 |
| SMD | OCSVM | 0.7053 | 0.7509 |
| SMD | PCA | 0.6525 | 0.7339 |
| SMD | timesnet | 0.5821 | 0.2145 |
| SMD | tranad | 0.3092 | 0.1813 |
| SMD | USAD (non-adversarial) | 0.4421 | 0.1035 |
| SWAT | AnomalyTransformer | 0.9018 | 0.8741 |
| SWAT | catch | 0.2166 | 0.8538 |
| SWAT | dcdetector | 0.2187 | 0.9417 |
| SWAT | IsolationForest | 0.8493 | 0.7932 |
| SWAT | itransformer | 0.2166 | 0.8903 |
| SWAT | kNN | 0.8479 | 0.8221 |
| SWAT | Mahalanobis | 0.8299 | 0.8204 |
| SWAT | OCSVM | 0.8521 | 0.8221 |
| SWAT | PCA | 0.8361 | 0.8226 |
| SWAT | timesnet | 0.2166 | 0.9219 |
| SWAT | tranad | 0.8183 | 0.7930 |

## Cross-dataset operating point (PCA, reference protocol recall)

| Dataset | labelled anomaly rate | recall |
|---|---|---|
| MSL | 0.1053 | 0.0301 |
| PSM | 0.2776 | 0.0183 |
| SMAP | 0.1279 | 0.0031 |
| SMD | 0.0416 | 0.0450 |
| SWAT | 0.1214 | 0.1697 |
