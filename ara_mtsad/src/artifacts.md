# Code Artefact Index

Every code artefact introduced or extended by this study. Paths are relative to
the repository root.

## Data layer

- `src/iia_benchmark/data/mtsad.py` - payload loaders (`npy`, PSM CSV, SWaT CSV),
  the shape contract, train-only standardisation, window helpers and the two
  layout mappings.
- `configs/datasets/mtsad_smd.json`, `mtsad_msl.json`, `mtsad_smap.json`,
  `mtsad_psm.json`, `mtsad_swat.json` - per-dataset loader, expected shapes,
  window grid, anomaly ratio and provenance.
- `configs/datasets/public_sources.json` - round-4 acquisition entries with
  SHA-256 checksums for all fifteen payload files.

## Evaluation layer

- `src/iia_benchmark/evaluation/mtsad_metrics.py` - point-wise metrics, vectorised
  point adjustment, range-based metrics, oracle and percentile thresholds, and
  the single-protocol evaluator.

## Model layer

- `src/iia_benchmark/models/mtsad_detectors.py` - five instantaneous detectors and
  two windowed deep detectors behind one interface, with `# Grounding:` tags.
- `configs/models/mtsad_pca.json`, `mtsad_mahalanobis.json`, `mtsad_knn.json`,
  `mtsad_isolation_forest.json`, `mtsad_ocsvm.json`, `mtsad_usad.json`,
  `mtsad_anomaly_transformer.json`, `mtsad_dcdetector.json` - hyperparameters,
  entry points and reproduction status per detector. The DCdetector config
  additionally carries `dataset_overrides`, because the reference scripts use a
  different window and anomaly budget per dataset (SMD 105, MSL 90, SMAP 105,
  PSM 60, SWaT 105).

## Experiment layer

- `scripts/mtsad/run_reproduction.py` - the protocol-matrix harness; writes one
  JSON record per (dataset, detector, seed) plus `records.json`.
- `scripts/mtsad/summarize_reproduction.py` - renders run records as markdown
  result tables and compares them with the transcribed published numbers.
- `scripts/mtsad/derive_protocol_evidence.py` - derives the protocol-sensitivity
  table straight from the run records.
- `scripts/mtsad/verify_range_metrics.py` - cross-checks the local flat
  range-based metric against the reference affiliation implementation.
- `configs/experiments/mtsad_reproduction.json` - dataset list, detector list,
  seeds and the four declared protocols.

## Acquisition layer

- `scripts/data_acquisition/download_public_datasets.py` - extended with
  `--family` prefix selection for multi-file dataset families.
- `scripts/literature/download_representative_papers.py` - extended with
  absolute-path resolution and a browser user agent for publisher PDFs.

## Tests

- `tests/test_mtsad_metrics.py` - point adjustment, range metrics, oracle
  threshold (checked against brute force), threshold pooling, unscored-point
  handling.
- `tests/test_mtsad_data.py` - loader round trip, shape-contract enforcement,
  train-only standardisation, window helpers.
- `tests/test_mtsad_detectors.py` - detector factory, PCA ranking, Mahalanobis
  translation invariance, and CPU smoke runs of both deep detectors.
