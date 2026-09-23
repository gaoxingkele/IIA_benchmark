---
title: "Reproduction and protocol audit of the SWaT/MSL/SMAP/PSM/SMD anomaly-detection benchmark"
authors:
  - "IIA Benchmark maintainers"
year: 2026
venue: "Internal reproduction study (IIA_benchmark)"
doi: "Not applicable - study of published artefacts"
ara_version: 1
domain: "time-series anomaly detection"
keywords:
  - multivariate time-series anomaly detection
  - benchmark reproduction
  - point adjustment
  - SWaT
  - SMD
  - PSM
  - MSL
  - SMAP
claims_summary:
  - "Reported ranks on this family are driven by the evaluation protocol as much as by the detector."
  - "Point adjustment converts one detected point into a whole-segment credit, decoupling the reported F1 from point-wise precision."
  - "The five datasets are not mutually comparable without their per-dataset preprocessing and split versions."
abstract: >
  This artifact pins the SWaT/MSL/SMAP/PSM/SMD benchmark payloads, transcribes the
  published result tables of the anchor papers, re-runs five classical detectors and
  two deep detectors through one harness, and separates four protocol choices
  (window-to-timestamp layout, threshold rule, point adjustment, split version).
  It exists to answer a concrete question - do re-run numbers reproduce the papers -
  and to make the answer auditable rather than anecdotal.
---

# Layer Index

- `logic/problem.md` - what is being reproduced and why the numbers disagree.
- `logic/claims.md` - falsifiable claims with proof pointers to `logic/experiments.md`.
- `logic/concepts.md` - protocol vocabulary (point adjustment, range-based F1, association discrepancy).
- `logic/experiments.md` - the re-run plans, without result numbers.
- `logic/related_work.md` - typed dependency graph over the 19 registered papers.
- `logic/solution/constraints.md` - what this study can and cannot establish.
- `logic/solution/method.md` - the harness formalisation and its protocol matrix.
- `src/environment.md` - hardware, interpreter, dependency versions.
- `src/artifacts.md` - pointer index to every code/config artefact.
- `trace/exploration_tree.yaml` - the decision and dead-end graph of this iteration.
- `evidence/README.md` - filing status of every source object (tables, figures).
- `evidence/tables/*.md` - transcribed published tables and re-run result tables.
