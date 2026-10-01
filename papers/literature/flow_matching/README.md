# Flow matching literature and experimental data

The collection covers the 20 papers requested on 2026-10-01: GRASP, TG-MSFM,
GiFlow, CFMI, TSFlow, DFM, PrismFlow, JFI, CSDI, SSSD, BRITS, SAITS, GRIN,
Diffusion-TS, GANF, CATCH, DCdetector, Flow Matching, Rectified Flow, and OT-CFM.

`paper_metadata.json` contains bibliographic provenance and dataset bindings.
`paper_download_manifest.json` identifies locally validated PDFs. PDFs and
extracted text stay local under `pdfs/` and `extracted/`; hashes and acquisition
metadata are tracked. JFI currently has no verified open full-text source.

Data registrations live in `configs/acquisition/flow_matching_*_data_sources.json`.
They distinguish paper-specific processed data, public raw data, synthetic
generation recipes, and restricted or withdrawn artifacts. Shared dataset names
do not imply equivalent preprocessing, splits, masks, or experiment versions.
Existing data is preserved. Acquiring data does not implement a method or create
a leaderboard result.

The downloader prefers aria2c via `http://127.0.0.1:17890`, verifies registered
publisher checksums when available, records local SHA-256, checks file signatures,
and validates ZIP CRCs. A local hash alone is not publisher authentication. Large
DTU files use resumable ranges after measured aria2 transport failures; the final
assembled HDF5 must match the publisher MD5 and byte size before becoming available.

```powershell
python -m pip install -e ".[acquisition]"
python scripts/data_acquisition/download_flow_matching_bundle.py --registry configs/acquisition/flow_matching_papers.json --manifest papers/literature/flow_matching/paper_download_manifest.json
python scripts/data_acquisition/download_flow_matching_bundle.py --registry configs/acquisition/flow_matching_data.json --manifest papers/literature/flow_matching/data_download_manifest.json
python scripts/data_acquisition/download_flow_matching_bundle.py --registry configs/acquisition/flow_matching_additional_data.json --manifest papers/literature/flow_matching/additional_data_manifest.json
python scripts/data_acquisition/download_flow_matching_bundle.py --registry configs/acquisition/flow_matching_foundation_extra_data.json --manifest papers/literature/flow_matching/foundation_extra_data_manifest.json
python scripts/data_acquisition/download_flow_matching_bundle.py --registry configs/acquisition/flow_matching_fm_data_sources.json --manifest papers/literature/flow_matching/fm_large_data_manifest.json --min-bytes 1000000000
python scripts/data_acquisition/download_flow_matching_drive.py --workers 6 --reuse-inventory --retry-failed
python scripts/data_acquisition/download_mega_public_bundle.py --help
python scripts/data_acquisition/report_flow_matching_bundle.py
```

Download manifests count only completed validated artifacts. `.part`, `.chunks`,
and `.assembling` files are intermediate transfers and are never benchmark data.
The report records remaining access gates and generation/preprocessing work.

The initial acquisition is still running. These are continuation commands, not
an assertion that all experimental artifacts are present. The Windows background
runner `continue_flow_matching_downloads.py` waits for the active generic download
queues, retries failed files twice, and refreshes the report every 30 seconds.
Its machine-specific process IDs, logs, and status are kept locally in
`data/public_datasets/flow_matching/acquisition_runtime/`; after a restart, use
the commands above rather than reusing stale process IDs. Drive and MEGA have
separate downloaders and manifests. Do not run duplicate queues against the same
destination while a transfer is active.

On the alternate-transport recovery pass, native curl successfully recovered all
72 previously failed Drive files. `drive_download_manifest.json` now records all
775 requested author-folder files as available. The remaining public files are
registered separately, preserving the original publisher checksums:

```powershell
python scripts/data_acquisition/download_flow_matching_alternatives.py --registry configs/acquisition/flow_matching_alternative_sources.json --manifest papers/literature/flow_matching/alternative_download_manifest.json --workers 6
python scripts/data_acquisition/download_flow_matching_alternatives.py --registry configs/acquisition/flow_matching_alternative_sources.json --manifest papers/literature/flow_matching/alternative_tep_manifest.json --only-large --workers 3
python scripts/data_acquisition/download_flow_matching_alternatives.py --registry configs/acquisition/flow_matching_alternative_extra_sources.json --manifest papers/literature/flow_matching/alternative_extra_manifest.json --workers 4
```

The second command reuses the original `.chunks` directory and validates each
16 MiB HTTP byte range before appending it. Final TEP files still require the
publisher byte size and MD5. Other files use native curl with curl-cffi as a
fallback and retain separate transport partials. Curl documentation and
curl-cffi documentation are linked in the alternate registry. Access gates,
withdrawn MEGA shares, and private experiment datasets remain explicit gaps.
CelebA range probes returned real data, but full-download requests still report
Google Drive quota exceeded. Its image archive and metadata remain unavailable;
their publisher MD5 checksums are registered for later verification. The author EEG extension was found
to be `EEG_Eye_State.arff`, not ZIP; the registration is corrected and the old
transfer files are preserved.
