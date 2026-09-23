# Problem

## Observations

The five-dataset suite (SWaT, MSL, SMAP, PSM, SMD) is the de-facto benchmark for
multivariate time-series anomaly detection. It is used as the comparison table of
Anomaly Transformer, TranAD, DCdetector, TimesNet, iTransformer, CATCH, GCAD and
successor papers, and the payloads are redistributed by the Time-Series-Library.

Three observations motivate this artifact:

1. The same method receives different scores in different papers. Anomaly
   Transformer reports SMD F1 92.33 (its own Table 1) while DCdetector reproduces
   the same method at 90.33 (DCdetector Table 1), with identical dataset names.
2. The dataset statistics themselves disagree between papers. TranAD's dataset
   table lists SWaT with 496800 training rows; the Redistributed benchmark split
   loaded by the reference SWATSegLoader yields 495000 rows; the payload shipped in
   this repository is the latter. The 1800/19800-row gaps are not explained in the
   papers.
3. The reported protocol hides a large part of the difference. Point adjustment
   credits a whole labelled anomaly segment when one point inside it is detected,
   so the headline F1 and the point-wise F1 of the same detector can differ by an
   order of magnitude (see `evidence/tables/`).

## Gaps

- No single artefact in the repository pinned the payload bytes, the split
  version, the window layout, the threshold rule and the metric family together.
- The published numbers were not transcribed with their objects, so a comparison
  could silently mix the papers' protocols.
- No re-run harness existed for this dataset family, so "we reproduce the paper"
  could not be checked against anything.

## Key insight

For this benchmark family the evaluation protocol is a first-class experimental
variable, not a detail: window-to-timestamp layout, threshold rule, point
adjustment and split version each move the headline number. A reproduction claim
is therefore only meaningful when all four are declared, and the honest question
is not "does the score match" but "does the score match *under a declared
protocol*".

## Assumptions

- The Time-Series-Library redistribution is an acceptable stand-in for the
  original dataset releases when it is byte-pinned and its provenance recorded.
- The canonical original releases (SMD, PSM, SMAP/MSL, SWaT) remain the
  authoritative sources for claims about the data's semantics.
- A detector implemented from the published equations is a different artefact
  from the authors' released code, and the artifact labels which of the two was
  used for every number.
