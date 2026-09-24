# Claims

## C01 - Protocol choice dominates detector identity on this family

**Statement**: On this benchmark family, moving between declared protocol choices
changes a detector's headline score by more than the spread between detectors,
so a ranking that does not fix the protocol is not a property of the methods.

**Conditions**: Holds for reconstruction- and distance-style detectors evaluated
through the windowed layouts used here; untested for protocols that additionally
require calibrated alarm design (for example FAR/MAR-based alarm generation).

**Falsification criteria**: Falsified if, across the five datasets, re-ranking
detectors under the strict point-wise protocol reproduces the ranking obtained
under the point-adjusted protocol with no rank inversions.

**Dependencies**: C02.

**Proof**: E01, E03.

## C02 - Point adjustment decouples the reported score from point-wise precision

**Statement**: Because a single detected point credits an entire labelled anomaly
segment, a detector that fires once per segment attains a near-perfect adjusted
score while its point-wise precision stays at the level implied by the segment
length, so the adjusted score measures segment coverage rather than detection
sharpness.

**Conditions**: Established for contiguous point labels as shipped in this
family; the size of the effect scales with segment length and is therefore
dataset-dependent.

**Falsification criteria**: Falsified if, on these payloads, the ratio between a
detector's adjusted and point-wise score is independent of its false-positive
rate.

**Proof**: E01, E02.

## C03 - The five datasets are not commensurable without their preprocessing

**Statement**: Channel count, sampling rate and label granularity differ enough
across the five payloads that a detector's score on one payload carries no direct
information about its score on another, so cross-dataset averages are only
interpretable when each dataset's preprocessing and threshold rule are applied
per dataset.

**Conditions**: Applies to the shipped splits; it does not claim the datasets are
uninformative, only that their scores are not exchangeable units.

**Falsification criteria**: Falsified if a single threshold and a single
preprocessing recipe reproduce every dataset's reported operating point.

**Proof**: E04.

## C04 - SWaT comparisons require a pinned split version

**Statement**: Published SWaT numbers in this family are produced on differently
sized training splits, so two SWaT scores can be compared only after the split
version is pinned; an unpinned comparison can attribute a training-set difference
to a modelling difference.

**Conditions**: Evidence covers the splits named in the registered papers and the
redistributed payload; other SWaT derivatives were not audited.

**Falsification criteria**: Falsified if the registered papers' SWaT training
row counts can be shown to agree once the same preprocessing script is applied.

**Proof**: E05.

## C05 - Classical detectors already occupy the reported classical-baseline band

**Statement**: Under the reference protocol, instantaneous detectors that ignore
temporal context reach the same score band as the classical baselines quoted in
the anchor papers, so a temporal model's advantage on this family is only
measurable once the protocol inflation is removed.

**Conditions**: Claimed for the classical baseline rows of the anchor tables
(isolation forest, one-class SVM, distance-style detectors); it does not extend to
the deep baselines whose reported numbers were produced by their own harnesses.

**Falsification criteria**: Falsified if the re-run instantaneous detectors fall
systematically below the quoted classical baseline rows across all five payloads.

**Proof**: E01, E04.

## C06 - On this family, a column is reproducible if the payload and protocol match; residual gaps are method-specific

**Statement**: Independently transcribed detectors, evaluated on the payload their
papers used, reproduce every dataset of the family within a few F1 points, and the
residual gap tracks neither the dataset column nor the architecture family but the
training configuration - so a residual gap is evidence about the transcription and
its budget, not about the dataset's reliability.

**Conditions**: Established for the four deep detectors re-run here under the
reference protocol, three of which have published counterparts. The claim is about
this artifact's transcriptions, not about the papers' released code, and it does
not extend to datasets outside the family.

**Falsification criteria**: Falsified if a fifth transcription lands outside the
three-point band on a dataset whose payload, protocol *and* training budget are
all verified identical to the reference.

**Dependencies**: C01.

**Proof**: E03, E06.

## C07 - Independent metric families agree once the family is matched

**Statement**: Converting a re-run's predictions into the metric family a third
party published in brings the two numbers within a few points, including on the
dataset where the point-adjusted comparison was furthest off - so the apparent
cross-paper disagreement is dominated by the metric family rather than by the
model or the data.

**Conditions**: Established for TimesNet across all five datasets and for the
Anomaly Transformer on MSL, both converted at this harness's own percentile
threshold rather than at the third party's operating point; it does not claim
bit-identical protocols.

**Falsification criteria**: Falsified if a further conversion of a different model
lands far outside the band, or if the third party's numbers can be shown to use a
different payload rather than a different family.

**Dependencies**: C02, C06.

**Proof**: E07.
