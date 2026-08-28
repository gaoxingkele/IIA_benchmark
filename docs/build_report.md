# Initial build report — 2026-08-27

> Historical snapshot. Current machine-readable state is in `docs/status_audit.json`.
> TEP/NPP archives were acquired on 2026-08-28, and the four official FCC Alarm
> attachments were acquired from ReSeeD on 2026-08-29.

## Delivered

- 433-page reference book verified and split into front matter + six page-addressable chapters; source SHA-256 begins `d002db25abf1`.
- Six chapter-to-benchmark notes, task contract, algorithm maturity matrix, evaluation protocol and three expansion rounds.
- Runnable implementations for threshold/delay/deadband design, Mahalanobis, convex-hull NOZ/dynamic bounds, transfer-entropy ranking, flood detection, Smith–Waterman similarity, next-alarm prediction and episode perturbation.
- TEP ASCII and PIADE alarm adapters; JSON experiment/config system and result writer.
- aria2/proxy-first public data registry, downloader, checksum/Git audit and structural profiler.

## Data state

Present resources total 118,820,844 bytes in this cache audit (includes Git working trees/metadata): TEP classic, PRONTO README/technical report, both PIADE tables, SKAB, and TEP/NPP DataPort landing metadata. PRONTO 1.72 GB full payload is opt-in. TEP alarm and NPP binary archives remain login/terms gated; FCC DOI is registered but its repository redirect failed during this run.

PIADE profile: 429,394 raw rows, 5 equipment IDs, 92,084 non-sentinel alarm intervals; 23,376 hourly rows and 164 columns. SKAB contains 35 experiment CSVs. TEP classic contains 44 run files and 52 features.

## Smoke effects

These values validate plumbing only and are not leaderboard claims.

| Experiment | Key result |
|---|---|
| univariate design | threshold 0.5, delay 8, deadband 0.25; F1 0.9950, FAR 0.00429, MAR 0.004, AAD 2 samples |
| convex NOZ | F1 0.9852, FAR 0.015, MAR 0, 56 hull facets |
| transfer entropy RCA | ROOT ranked first at lag 3; TE 0.2429 vs surrogate threshold 0.00452 |
| flood similarity | leave-one-episode nearest-neighbor accuracy 1.0 on 10 synthetic episodes |

## Verification

`pytest -q`: 16 passed. `validate_scaffold.py`: valid. Dataset audit: all default downloadable records valid. Tests cover stateful alarms, grid design, multivariate outliers/dynamic bounds, delayed causality, alignment/flood perturbation, PIADE/TEP adapters, metrics, all four runner paths and book page boundaries.
