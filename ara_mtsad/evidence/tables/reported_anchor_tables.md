# Published anchor tables

Transcribed with `pdftotext -layout` from the local PDFs; values are copied
verbatim and are percentages except where a row states fractions.

## Anomaly Transformer (xu2022_anomaly_transformer.pdf, Table 1)

| Method | SMD P/R/F1 | MSL P/R/F1 | SMAP P/R/F1 | SWaT P/R/F1 | PSM P/R/F1 |
|---|---|---|---|---|---|
| OCSVM | 44.34 / 76.72 / 56.19 | 59.78 / 86.87 / 70.82 | 53.85 / 59.07 / 56.34 | 45.39 / 49.22 / 47.23 | 62.75 / 80.89 / 70.67 |
| IsolationForest | 42.31 / 73.29 / 53.64 | 53.94 / 86.54 / 66.45 | 52.39 / 59.07 / 55.53 | 49.29 / 44.95 / 47.02 | 76.09 / 92.45 / 83.48 |
| LSTM-VAE | 75.76 / 90.08 / 82.30 | 85.49 / 79.94 / 82.62 | 92.20 / 67.75 / 78.10 | 76.00 / 89.50 / 82.20 | 73.62 / 89.92 / 80.96 |
| OmniAnomaly | 83.68 / 86.82 / 85.22 | 89.02 / 86.37 / 87.67 | 92.49 / 81.99 / 86.92 | 81.42 / 84.30 / 82.83 | 88.39 / 74.46 / 80.83 |
| Anomaly Transformer (Ours) | 89.40 / 95.45 / 92.33 | 92.09 / 95.15 / 93.59 | 94.13 / 99.40 / 96.69 | 91.55 / 96.73 / 94.07 | 96.91 / 98.90 / 97.89 |

## DCdetector (yang2023_dcdetector.pdf, Table 1)

| Method | SMD P/R/F1 | MSL P/R/F1 | SMAP P/R/F1 | SWaT P/R/F1 | PSM P/R/F1 |
|---|---|---|---|---|---|
| AnomalyTrans (re-run by DCdetector) | 88.47 / 92.28 / 90.33 | 91.92 / 96.03 / 93.93 | 93.59 / 99.41 / 96.41 | 89.10 / 99.28 / 94.22 | 96.94 / 97.81 / 97.37 |
| DCdetector | 83.59 / 91.10 / 87.18 | 93.69 / 99.69 / 96.60 | 95.63 / 98.92 / 97.02 | 93.11 / 99.77 / 96.33 | 97.14 / 98.74 / 97.94 |

Note the anchor method's SMD F1 differs between the two papers (92.33 versus
90.33) with identical dataset names - the observation behind C01.

## TranAD (tuli2022_tranad.pdf, Table 2)

| Method | SMAP F1 | MSL F1 | SWaT F1 | SMD F1 |
|---|---|---|---|---|
| USAD | 0.8419 | 0.8822 | 0.8143 | 0.9495 |
| OmniAnomaly | 0.8728 | 0.8765 | 0.8131 | 0.9401 |
| DAGMM | 0.8888 | 0.8482 | 0.8128 | 0.9491 |
| TranAD | 0.8915 | 0.9494 | 0.8151 | 0.9605 |

PSM is not evaluated in TranAD. Dataset table rows used by the SWaT audit: SMAP
135183 / 427617 / "25 (55)" / 13.13 percent anomalous; SWaT 496800 / 449919 /
"51 (1)" / 11.98 percent anomalous.
