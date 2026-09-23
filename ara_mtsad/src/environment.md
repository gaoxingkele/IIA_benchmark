# Environment

## Host

- Windows, PowerShell; workspace `D:\aicoding\IIA_benchmark`.
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB, driver 610.62.
- RAM: 68,572,536,832 bytes total.

## Interpreter and dependencies

- Python 3.12.10.
- torch 2.13.0+cu126 (installed for this study so that the deep detectors could be
  trained on the GPU; the base environment shipped a CPU-only build).
- numpy 2.4.6, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.9.0, pyyaml 6.0.3,
  pytest 9.1.1, openpyxl 3.1.5, matplotlib 3.11.0.
- Poppler `pdftotext`/`pdfinfo` used for PDF text extraction and page counts.

## Network

- HTTP(S) proxy `http://127.0.0.1:17890`, registered in the acquisition scripts.
- `aria2c` 1.37.0 from `C:\Users\10175\AppData\Local\aria2\...`;
  `aria2c` is not on `PATH`, so the downloaders search explicit candidates.

## Data

- Benchmark payloads under `data/public_datasets/mtsad/` (git-ignored), pinned by
  SHA-256 in `configs/datasets/public_sources.json` and re-checkable with
  `python scripts/data_acquisition/download_public_datasets.py --family mtsad`
  followed by `python scripts/data_acquisition/audit_public_datasets.py`.
- Literature PDFs under `papers/literature/mtsad_pdfs/` (git-ignored); the tracked
  manifest `papers/literature/mtsad_download_manifest.json` carries the SHA-256
  and page count of every file that was available.

## Determinism

- Seeds 1103/1104/1105 for stochastic components; torch and numpy seeds are set
  inside the detectors; window construction and thresholds are deterministic.
- GPU kernels are not bit-deterministic across library versions; the run records
  therefore carry the interpreter and revision, and per-seed spread is reported
  instead of a single number.
