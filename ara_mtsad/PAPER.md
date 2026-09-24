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
  - "Point adjustment converts one detected point into a whole-segment credit, so the headline F1 measures segment coverage rather than detection sharpness."
  - "The five datasets are not mutually comparable without their per-dataset preprocessing and split versions."
  - "Two metric families travel under the same benchmark name and differ by up to 0.53 on identical predictions."
  - "Residual gaps after matching payload, protocol and metric family track the model and its partly unpublished training configuration, not the dataset column."
abstract: >
  This artifact pins the SWaT/MSL/SMAP/PSM/SMD benchmark payloads, transcribes the
  published result tables of the anchor papers, transcribes and re-runs eleven
  detectors (five classical plus Anomaly Transformer, DCdetector, TimesNet,
  iTransformer, USAD and TranAD), and reports the result under explicit protocol
  choices: window-to-timestamp layout, threshold rule (percentile, oracle or the
  release's own POT), point adjustment, metric family and split version. 124 runs
  feed three comparison tables - 23 point-adjusted pairs and 16 affiliation-family
  pairs - which resolve to 5 matches, 8 near and 15 off against the transcribed
  published values, with every gap either explained or recorded as an open
  discrepancy. The question it answers is concrete - do re-run numbers reproduce
  the papers - and the answer is made auditable rather than anecdotal.
---

# Headline findings

1. **Point adjustment, not the model, sets the scale.** At one threshold, the same
   detector scores 0.0060 point-wise and 0.6973 point-adjusted on SMAP (a factor of
   116); the range across datasets is 2.8x to 116x (`evidence/tables/protocol_sensitivity.md`).
2. **A transcription can reproduce a published number exactly.** The Anomaly
   Transformer re-run reaches SMD 0.9034 against the 0.9033 a third party reports
   for the same method (delta 0.0001) while classifying 2.5 percent of timestamps
   correctly (`evidence/tables/rerun_deep_status.md`).
3. **TimesNet reproduces everywhere.** All five datasets within 0.068 in the
   point-adjusted family and within 0.068 in the affiliation family, which is the
   clearest evidence that matching payload and protocol is what makes a column
   reproducible (`evidence/tables/rerun_deep_status.md`, `affiliation_family_check.md`).
4. **Two metric families share one benchmark name.** A controlled comparison on
   identical predictions differs by up to 0.53 in precision, and CATCH publishes
   the other family for the same five datasets
   (`evidence/tables/range_metric_variant_check.md`).
5. **Most residual gaps vanish when the family is matched**, but not all: 16
   converted comparisons average 0.062 absolute difference, with DCdetector on
   SMAP (0.161) and iTransformer on MSL (0.131) still open
   (`evidence/tables/affiliation_comparison.md`).
6. **One recipe could not be matched end to end.** TranAD's model was transcribed
   and both threshold rules were tried, including the release's own vendored SPOT;
   neither reproduces the paper, and its release evaluates one representative
   series per dataset where the paper reports whole-dataset sizes, so its verdict
   is withheld (`evidence/tables/tranad_check.md`).

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
- `trace/exploration_tree.yaml` - the decision, correction and dead-end graph.
- `evidence/README.md` - filing status of every source object (tables, figures).
- `evidence/tables/rerun_vs_published.md` - the verdict table over every run record.
- `evidence/tables/rerun_deep_status.md` - deep-detector verdicts and the gap audits.
- `evidence/tables/affiliation_family_check.md` - the cross-family comparison.
- `evidence/tables/tranad_check.md` - the withheld verdict and why.
- `evidence/tables/registered_sota_status.md` - what each remaining paper still needs.
- `evidence/tables/*.md` - further transcribed tables and derived analyses.
