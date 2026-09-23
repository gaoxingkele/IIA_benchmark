# Concepts

## Point adjustment

A post-processing rule applied to a binary prediction before scoring, defined in
the benchmark lineage (`utils.tools.adjustment`): when any timestamp inside a
contiguous ground-truth anomaly segment is predicted positive, every timestamp of
that segment is re-labelled as a true positive. It was introduced to tolerate
labelling uncertainty about the exact anomaly onset and is the reason the
headline F1 of this family is not a point-wise quantity.

## Point-wise F1

Precision and recall computed directly per timestamp with no tolerance window.
This is the strict counterpart of point-adjusted F1 and exposes false positives
that point adjustment forgives.

## Range-based F1

A scoring family that compares predicted anomaly *ranges* with real anomaly ranges
through an overlap reward plus an optional existence reward weighted by `alpha`;
`alpha = 0` removes the existence credit and is the variant reported here. It
measures how much of a real range is covered rather than how many points are
correct.

## Window-to-timestamp layout

The mapping from per-window scores to a per-timestamp score series. The reference
harnesses concatenate label windows and score windows, so a single window score
appears as many entries as the window is long (`window_flatten`), whereas an
alternative layout assigns each window score to its final timestamp and averages
the votes (`last_point`). The layout alone re-weights every timestamp in the
metric.

## Threshold rule

How the decision boundary is chosen: a percentile of the pooled train+test energy
distribution (`percentile_combined`, the reference rule, parameterised by
`anomaly_ratio`) or an oracle search maximising point-wise F1 on the test labels
(`best_f1`). The first uses the test split's score distribution; the second uses
the test labels.

## Association discrepancy (Anomaly Transformer)

The divergence between the learned *series association* (softmax attention map)
and a Gaussian *prior association* parameterised by a learnable per-head scale.
The training objective is a minimax game between reconstruction error and the
discrepancy; at test time the reconstruction error is re-weighted by
`softmax(-series_loss - prior_loss)`. Transcribed from the reference
implementation in `model/attn.py` and `solver.py`.

## Reconstruction energy

The per-timestamp anomaly score of a reconstruction model: the mean squared error
between the input window and its reconstruction. It is the score that both the
Time-Series-Library harness and the Anomaly Transformer solver threshold.

## Benchmark payload vs canonical release

The *canonical release* is the dataset as published by its originators (iTrust
SWaT request form, NASA telemetry, eBay server metrics, NetManAIOps machine
telemetry). The *benchmark payload* is the preprocessed split redistributed for
these papers. This artifact pins the payload bytes and records the canonical
release as provenance, because only the payload is what the papers' numbers were
computed on.
