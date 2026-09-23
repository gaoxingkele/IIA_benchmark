# SWaT split audit

## What the papers state

| Source | SWaT training rows | SWaT test rows | Dimensions | Anomaly rate |
|---|---|---|---|---|
| TranAD (tuli2022_tranad.pdf, Table 2 dataset table) | 496800 | 449919 | 51 (1) | 11.98 % |
| Pinned payload `swat_train2.csv` / `swat2.csv` (this artifact) | 495000 | 449919 | 51 (+1 label column) | 12.14 % |
| Raw workbooks (this artifact, canonical release shape) | 495000 | 449920 | 51 (+1 label column) | - |

Source quotes: the TranAD text was extracted with `pdftotext -layout` from
`papers/literature/mtsad_pdfs/tuli2022_tranad.pdf` and contains the line pair
`SMAP 135183 427617 25 (55) 13.13` and `SWaT 496800 449919 51 (1) 11.98`. The
payload row counts were measured by loading the pinned files, and the raw workbook
row counts were measured with `openpyxl` in read-only mode
(`SWaT_Dataset_Normal_v1.xlsx`: 495000 data rows; `SWaT_Dataset_Attack_v0.xlsx`:
449920 data rows including the boundary row the reference preprocessing drops).

## Why this matters

The three SWaT training sizes differ by 1800 and 19800 rows (30 minutes and 5.5
hours at one-second sampling). None of the audited sources documents the
difference, so a SWaT F1 quoted from one paper cannot be compared with a SWaT F1
computed on another paper's split without first pinning which split was used. This
artifact therefore records the discrepancy and uses the payload as shipped rather
than trimming it to match the paper.

## Consequence for the comparison

SWaT rows in the comparison tables carry this caveat: the re-run is on 495000
training rows, the TranAD reference numbers are on 496800 rows, and the Anomaly
Transformer reference numbers are on an unspecified-but-different preprocessing.
