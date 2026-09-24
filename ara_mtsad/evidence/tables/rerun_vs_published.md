# Re-run versus published numbers (reference protocol)

Generated 2026-09-24 from 96 run records.
Reference protocol = percentile threshold over pooled train+test energy
(the rule used by the anchor papers) plus point adjustment.

| Method | Dataset | Re-run F1-PA | Re-run point-wise F1 | Published F1 | |delta| | verdict | config | source |
|---|---|---|---|---|---|---|---|---|
| AnomalyTransformer | MSL | 0.8557 | 0.0182 | 0.9393 | 0.0836 | off | win100/step1/ratio1/ep3/bs256 | yang2023_dcdetector |
| AnomalyTransformer | PSM | 0.9688 | 0.0220 | 0.9737 | 0.0049 | matches | win100/step1/ratio1 | yang2023_dcdetector |
| AnomalyTransformer | SMAP | 0.9535 | 0.0219 | 0.9641 | 0.0106 | near | win100/step1/ratio1/ep3/bs256 | yang2023_dcdetector |
| AnomalyTransformer | SMD | 0.9034 | 0.0249 | 0.9033 | 0.0001 | matches | win100/step100/ratio0.5 | yang2023_dcdetector |
| AnomalyTransformer | SWAT | 0.8754 | 0.0137 | 0.9422 | 0.0668 | off | win100/step1/ratio0.1/ep3/bs256 | yang2023_dcdetector |
| DCdetector | MSL | 0.8512 | 0.0180 | 0.9660 | 0.1148 | off | win90/step1/ratio1 | yang2023_dcdetector |
| DCdetector | PSM | 0.9647 | 0.0196 | 0.9794 | 0.0147 | near | win60/step1/ratio1 | yang2023_dcdetector |
| DCdetector | SMAP | 0.9414 | 0.0148 | 0.9702 | 0.0288 | near | win105/step1/ratio0.85/ep3/bs256 | yang2023_dcdetector |
| DCdetector | SMD | 0.8488 | 0.0123 | 0.8718 | 0.0230 | near | win105/step100/ratio0.6 | yang2023_dcdetector |
| DCdetector | SWAT | 0.9518 | 0.0190 | 0.9633 | 0.0115 | near | win105/step1/ratio1/ep3/bs128 | yang2023_dcdetector |
| IsolationForest | MSL | 0.8309 | 0.0300 | 0.6645 | 0.1664 | off | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| IsolationForest | PSM | 0.7503 | 0.0285 | 0.8348 | 0.0845 | off | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| IsolationForest | SMAP | 0.6862 | 0.0073 | 0.5553 | 0.1309 | off | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| IsolationForest | SMD | 0.5915 | 0.1812 | 0.5364 | 0.0551 | off | win100/step100/ratio0.5 | xu2022_anomaly_transformer |
| IsolationForest | SWAT | 0.7932 | 0.2378 | 0.4702 | 0.3230 | off | win100/step1/ratio1 | xu2022_anomaly_transformer |
| OCSVM | MSL | 0.7015 | 0.0566 | 0.7082 | 0.0067 | matches | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| OCSVM | PSM | 0.9146 | 0.0765 | 0.7067 | 0.2079 | off | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| OCSVM | SMAP | 0.6919 | 0.0071 | 0.5634 | 0.1285 | off | win100/step1/ratio0.5 | xu2022_anomaly_transformer |
| OCSVM | SMD | 0.7064 | 0.0688 | 0.5619 | 0.1445 | off | win100/step100/ratio0.5 | xu2022_anomaly_transformer |
| OCSVM | SWAT | 0.8221 | 0.2888 | 0.4723 | 0.3498 | off | win100/step1/ratio1 | xu2022_anomaly_transformer |
| USAD | MSL | 0.0942 | 0.0815 | 0.8822 | 0.7880 | off | win100/step1/ratio0.5 | tuli2022_tranad |
| USAD | SMAP | 0.4083 | 0.0415 | 0.8419 | 0.4336 | off | win100/step1/ratio0.5 | tuli2022_tranad |
| USAD | SMD | 0.1277 | 0.0702 | 0.9495 | 0.8218 | off | win100/step100/ratio0.5 | tuli2022_tranad |

## Verdict tally

| verdict | pairs |
|---|---|
| matches | 3 |
| near | 5 |
| off | 15 |

## Every configuration that was run

| Method | Dataset | config | F1-PA | point-wise F1 | range F1 |
|---|---|---|---|---|---|
| AnomalyTransformer | MSL | win100/step1/ratio0.5 | 0.8490 | 0.0114 | 0.0136 |
| AnomalyTransformer | MSL | win100/step1/ratio1/ep3/bs256 | 0.8557 | 0.0182 | 0.0158 |
| AnomalyTransformer | MSL | win100/step1/ratio1/ep3/bs256 | 0.8557 | 0.0182 | 0.0158 |
| AnomalyTransformer | MSL | win100/step1/ratio1/ep3/bs256 | 0.8478 | 0.0174 | 0.0156 |
| AnomalyTransformer | MSL | win100/step1/ratio1/ep3/bs256 | 0.8540 | 0.0174 | 0.0190 |
| AnomalyTransformer | MSL | win100/step1/ratio1/ep3/bs256 | 0.8396 | 0.0171 | 0.0186 |
| AnomalyTransformer | PSM | win100/step1/ratio0.5 | 0.9662 | 0.0200 | 0.0423 |
| AnomalyTransformer | PSM | win100/step1/ratio1 | 0.9688 | 0.0220 | 0.0625 |
| AnomalyTransformer | SMAP | win100/step1/ratio1/ep3/bs256 | 0.9535 | 0.0219 | 0.0291 |
| AnomalyTransformer | SMD | win100/step100/ratio0.5 | 0.8619 | 0.0163 | 0.0324 |
| AnomalyTransformer | SMD | win100/step100/ratio0.5 | 0.9034 | 0.0249 | 0.0988 |
| AnomalyTransformer | SWAT | win100/step1/ratio0.1/ep3/bs256 | 0.8754 | 0.0137 | 0.0071 |
| DCdetector | MSL | win90/step1/ratio1 | 0.8512 | 0.0180 | 0.0395 |
| DCdetector | PSM | win60/step1/ratio1 | 0.9647 | 0.0196 | 0.0155 |
| DCdetector | SMAP | win105/step1/ratio0.85/ep3/bs256 | 0.9414 | 0.0148 | 0.0221 |
| DCdetector | SMD | win105/step100/ratio0.6 | 0.8488 | 0.0123 | 0.0241 |
| DCdetector | SWAT | win105/step1/ratio1/ep3/bs128 | 0.9518 | 0.0190 | 0.0247 |
| IsolationForest | MSL | win100/step1/ratio0.5 | 0.8779 | 0.0334 | 0.0117 |
| IsolationForest | MSL | win100/step1/ratio0.5 | 0.8309 | 0.0300 | 0.0099 |
| IsolationForest | MSL | win100/step1/ratio0.5 | 0.8751 | 0.0412 | 0.0134 |
| IsolationForest | PSM | win100/step1/ratio0.5 | 0.7392 | 0.0278 | 0.0505 |
| IsolationForest | PSM | win100/step1/ratio0.5 | 0.7503 | 0.0285 | 0.0533 |
| IsolationForest | PSM | win100/step1/ratio0.5 | 0.7392 | 0.0277 | 0.0505 |
| IsolationForest | SMAP | win100/step1/ratio0.5 | 0.6905 | 0.0075 | 0.0108 |
| IsolationForest | SMAP | win100/step1/ratio0.5 | 0.6862 | 0.0073 | 0.0100 |
| IsolationForest | SMAP | win100/step1/ratio0.5 | 0.6921 | 0.0076 | 0.0086 |
| IsolationForest | SMD | win100/step100/ratio0.5 | 0.6298 | 0.1570 | 0.0951 |
| IsolationForest | SMD | win100/step100/ratio0.5 | 0.5915 | 0.1812 | 0.1014 |
| IsolationForest | SMD | win100/step100/ratio0.5 | 0.5932 | 0.1781 | 0.1022 |
| IsolationForest | SWAT | win100/step1/ratio1 | 0.7932 | 0.2378 | 0.0027 |
| IsolationForest | SWAT | win100/step1/ratio1 | 0.7932 | 0.2256 | 0.0010 |
| IsolationForest | SWAT | win100/step1/ratio1 | 0.7932 | 0.2423 | 0.0012 |
| kNN | MSL | win100/step1/ratio0.5 | 0.3722 | 0.0744 | 0.0700 |
| kNN | MSL | win100/step1/ratio0.5 | 0.2581 | 0.0755 | 0.0698 |
| kNN | MSL | win100/step1/ratio0.5 | 0.4809 | 0.0743 | 0.0685 |
| kNN | PSM | win100/step1/ratio0.5 | 0.9155 | 0.0843 | 0.1851 |
| kNN | PSM | win100/step1/ratio0.5 | 0.9155 | 0.0841 | 0.2503 |
| kNN | PSM | win100/step1/ratio0.5 | 0.8845 | 0.0846 | 0.1316 |
| kNN | SMAP | win100/step1/ratio0.5 | 0.6946 | 0.0097 | 0.0077 |
| kNN | SMAP | win100/step1/ratio0.5 | 0.6967 | 0.0096 | 0.0086 |
| kNN | SMAP | win100/step1/ratio0.5 | 0.6967 | 0.0103 | 0.0082 |
| kNN | SMD | win100/step100/ratio0.5 | 0.7979 | 0.1545 | 0.1627 |
| kNN | SMD | win100/step100/ratio0.5 | 0.7914 | 0.1430 | 0.1572 |
| kNN | SMD | win100/step100/ratio0.5 | 0.7918 | 0.1270 | 0.1586 |
| kNN | SWAT | win100/step1/ratio1 | 0.8221 | 0.2890 | 0.2314 |
| kNN | SWAT | win100/step1/ratio1 | 0.8221 | 0.2896 | 0.2313 |
| kNN | SWAT | win100/step1/ratio1 | 0.8221 | 0.2896 | 0.2314 |
| Mahalanobis | MSL | win100/step1/ratio0.5 | 0.7000 | 0.0585 | 0.0524 |
| Mahalanobis | MSL | win100/step1/ratio0.5 | 0.7000 | 0.0585 | 0.0524 |
| Mahalanobis | MSL | win100/step1/ratio0.5 | 0.7000 | 0.0585 | 0.0524 |
| Mahalanobis | PSM | win100/step1/ratio0.5 | 0.8853 | 0.0436 | 0.2008 |
| Mahalanobis | PSM | win100/step1/ratio0.5 | 0.8853 | 0.0436 | 0.2008 |
| Mahalanobis | PSM | win100/step1/ratio0.5 | 0.8853 | 0.0436 | 0.2008 |
| Mahalanobis | SMAP | win100/step1/ratio0.5 | 0.7059 | 0.0072 | 0.0080 |
| Mahalanobis | SMAP | win100/step1/ratio0.5 | 0.7059 | 0.0072 | 0.0080 |
| Mahalanobis | SMAP | win100/step1/ratio0.5 | 0.7059 | 0.0072 | 0.0080 |
| Mahalanobis | SMD | win100/step100/ratio0.5 | 0.7850 | 0.0787 | 0.1879 |
| Mahalanobis | SMD | win100/step100/ratio0.5 | 0.7850 | 0.0787 | 0.1879 |
| Mahalanobis | SMD | win100/step100/ratio0.5 | 0.7850 | 0.0787 | 0.1879 |
| Mahalanobis | SWAT | win100/step1/ratio1 | 0.8204 | 0.2865 | 0.2407 |
| Mahalanobis | SWAT | win100/step1/ratio1 | 0.8204 | 0.2865 | 0.2407 |
| Mahalanobis | SWAT | win100/step1/ratio1 | 0.8204 | 0.2865 | 0.2407 |
| OCSVM | MSL | win100/step1/ratio0.5 | 0.7015 | 0.0566 | 0.0528 |
| OCSVM | MSL | win100/step1/ratio0.5 | 0.7159 | 0.0576 | 0.0532 |
| OCSVM | MSL | win100/step1/ratio0.5 | 0.7002 | 0.0567 | 0.0521 |
| OCSVM | PSM | win100/step1/ratio0.5 | 0.9146 | 0.0765 | 0.3075 |
| OCSVM | PSM | win100/step1/ratio0.5 | 0.9146 | 0.0757 | 0.3074 |
| OCSVM | PSM | win100/step1/ratio0.5 | 0.9148 | 0.0773 | 0.3078 |
| OCSVM | SMAP | win100/step1/ratio0.5 | 0.6919 | 0.0071 | 0.0105 |
| OCSVM | SMAP | win100/step1/ratio0.5 | 0.6925 | 0.0070 | 0.0110 |
| OCSVM | SMAP | win100/step1/ratio0.5 | 0.7099 | 0.0077 | 0.0087 |
| OCSVM | SMD | win100/step100/ratio0.5 | 0.7730 | 0.0974 | 0.1697 |
| OCSVM | SMD | win100/step100/ratio0.5 | 0.7735 | 0.0905 | 0.1718 |
| OCSVM | SMD | win100/step100/ratio0.5 | 0.7064 | 0.0688 | 0.1725 |
| OCSVM | SWAT | win100/step1/ratio1 | 0.8221 | 0.2888 | 0.2314 |
| OCSVM | SWAT | win100/step1/ratio1 | 0.8221 | 0.2896 | 0.2314 |
| OCSVM | SWAT | win100/step1/ratio1 | 0.8221 | 0.2895 | 0.2315 |
| PCA | MSL | win100/step1/ratio0.5 | 0.7004 | 0.0567 | 0.0523 |
| PCA | MSL | win100/step1/ratio0.5 | 0.7004 | 0.0567 | 0.0523 |
| PCA | MSL | win100/step1/ratio0.5 | 0.7004 | 0.0567 | 0.0523 |
| PCA | PSM | win100/step1/ratio0.5 | 0.8791 | 0.0357 | 0.0578 |
| PCA | PSM | win100/step1/ratio0.5 | 0.8791 | 0.0357 | 0.0578 |
| PCA | PSM | win100/step1/ratio0.5 | 0.8791 | 0.0357 | 0.0578 |
| PCA | SMAP | win100/step1/ratio0.5 | 0.6973 | 0.0060 | 0.0086 |
| PCA | SMAP | win100/step1/ratio0.5 | 0.6973 | 0.0060 | 0.0086 |
| PCA | SMAP | win100/step1/ratio0.5 | 0.6973 | 0.0060 | 0.0086 |
| PCA | SMD | win100/step100/ratio0.5 | 0.7339 | 0.0740 | 0.1940 |
| PCA | SMD | win100/step100/ratio0.5 | 0.7339 | 0.0740 | 0.1940 |
| PCA | SMD | win100/step100/ratio0.5 | 0.7339 | 0.0740 | 0.1940 |
| PCA | SWAT | win100/step1/ratio1 | 0.8226 | 0.2896 | 0.1810 |
| PCA | SWAT | win100/step1/ratio1 | 0.8226 | 0.2896 | 0.1810 |
| PCA | SWAT | win100/step1/ratio1 | 0.8226 | 0.2896 | 0.1810 |
| USAD | MSL | win100/step1/ratio0.5 | 0.0942 | 0.0815 | 0.1065 |
| USAD | PSM | win100/step1/ratio0.5 | 0.7215 | 0.0513 | 0.0789 |
| USAD | SMAP | win100/step1/ratio0.5 | 0.4083 | 0.0415 | 0.0985 |
| USAD | SMD | win100/step100/ratio0.5 | 0.1277 | 0.0702 | 0.0939 |
