"""Interpretable HDAM templates and structured convolutional matching.

Rahimi et al. (2026) extend the High-Density Alarm Plot into a matrix,
align variable-duration floods with structured 2-D convolution, extract
category templates offline, and match dynamic matrices online.  The full
scoring equations are gated, so this implementation fixes tag semantics on the
row axis and exposes every local choice and alignment result for audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class HDAMAlignment:
    """Best normalized cross-correlation placement between two HDAMs."""

    score: float
    template_start: int
    candidate_start: int
    overlap_width: int
    tag_shift: int


@dataclass(frozen=True)
class HDAMTemplate:
    """Category consensus template derived from aligned historical segments."""

    label: Hashable
    values: np.ndarray
    stability: float
    source_episode: int
    source_start: int
    sample_count: int


def high_density_alarm_matrix(
    alarm_series: np.ndarray,
    bin_size: int = 1,
    aggregation: str = "binary",
) -> np.ndarray:
    """Convert a tag-by-time alarm series into a binned HDAM."""

    values = np.asarray(alarm_series, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("alarm_series must have shape (alarm_tags, time)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("alarm_series must contain finite nonnegative values")
    if bin_size <= 0 or aggregation not in {"binary", "count", "rate"}:
        raise ValueError("bin_size must be positive and aggregation must be binary/count/rate")
    bins = int(np.ceil(values.shape[1] / bin_size))
    padded = np.pad(values, ((0, 0), (0, bins * bin_size - values.shape[1])))
    counts = np.sum(padded.reshape(values.shape[0], bins, bin_size), axis=2)
    if aggregation == "binary":
        return (counts > 0).astype(float)
    if aggregation == "rate":
        return counts / bin_size
    return counts


def _tag_overlap(
    template: np.ndarray, candidate: np.ndarray, tag_shift: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if tag_shift >= 0:
        template_start = 0
        candidate_start = tag_shift
    else:
        template_start = -tag_shift
        candidate_start = 0
    rows = min(
        template.shape[0] - template_start,
        candidate.shape[0] - candidate_start,
    )
    if rows <= 0:
        return None
    return (
        template[template_start : template_start + rows],
        candidate[candidate_start : candidate_start + rows],
    )


def structured_hdam_alignment(
    template: np.ndarray,
    candidate: np.ndarray,
    max_tag_shift: int = 0,
    minimum_overlap: int = 1,
) -> HDAMAlignment:
    """Find the highest normalized 2-D correlation under bounded shifts.

    When one matrix is shorter, it is treated as the convolution kernel.  A
    nonzero ``max_tag_shift`` is available for rank-ordered HDAM variants; the
    default keeps alarm tag identities fixed and shifts only in time.
    """

    left = np.asarray(template, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("template and candidate must be finite matrices")
    if max_tag_shift < 0 or minimum_overlap <= 0:
        raise ValueError("max_tag_shift must be nonnegative and minimum_overlap positive")
    best: HDAMAlignment | None = None
    for tag_shift in range(-max_tag_shift, max_tag_shift + 1):
        row_pair = _tag_overlap(left, right, tag_shift)
        if row_pair is None:
            continue
        left_rows, right_rows = row_pair
        if left.shape[1] <= right.shape[1]:
            overlap = left.shape[1]
            placements = (
                (0, candidate_start)
                for candidate_start in range(right.shape[1] - overlap + 1)
            )
        else:
            overlap = right.shape[1]
            placements = (
                (template_start, 0)
                for template_start in range(left.shape[1] - overlap + 1)
            )
        if overlap < minimum_overlap:
            continue
        for template_start, candidate_start in placements:
            a = left_rows[:, template_start : template_start + overlap]
            b = right_rows[:, candidate_start : candidate_start + overlap]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator == 0:
                score = 1.0 if not np.any(a) and not np.any(b) else 0.0
            else:
                score = float(np.sum(a * b) / denominator)
            alignment = HDAMAlignment(
                float(np.clip(score, 0.0, 1.0)),
                template_start,
                candidate_start,
                overlap,
                tag_shift,
            )
            if best is None or (
                alignment.score,
                alignment.overlap_width,
                -abs(alignment.tag_shift),
                -alignment.candidate_start,
            ) > (
                best.score,
                best.overlap_width,
                -abs(best.tag_shift),
                -best.candidate_start,
            ):
                best = alignment
    if best is None:
        raise ValueError("matrices have no permitted overlap")
    return best


def extract_hdam_template(
    matrices: Sequence[np.ndarray],
    label: Hashable,
    template_width: int | None = None,
    candidate_stride: int = 1,
) -> HDAMTemplate:
    """Select a representative segment and average its aligned class peers."""

    values = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not values or any(matrix.ndim != 2 for matrix in values):
        raise ValueError("matrices must contain finite two-dimensional HDAMs")
    if any(not np.all(np.isfinite(matrix)) for matrix in values):
        raise ValueError("HDAMs must be finite")
    rows = values[0].shape[0]
    if any(matrix.shape[0] != rows for matrix in values):
        raise ValueError("all HDAMs must use the same alarm-tag rows")
    width = min(matrix.shape[1] for matrix in values) if template_width is None else int(template_width)
    if width <= 0 or candidate_stride <= 0 or any(matrix.shape[1] < width for matrix in values):
        raise ValueError("template_width and candidate_stride are incompatible with HDAMs")

    best_score = -np.inf
    best_source = 0
    best_start = 0
    best_alignments: list[HDAMAlignment] = []
    for source_index, matrix in enumerate(values):
        for start in range(0, matrix.shape[1] - width + 1, candidate_stride):
            candidate = matrix[:, start : start + width]
            alignments = [structured_hdam_alignment(candidate, other) for other in values]
            score = float(np.mean([item.score for item in alignments]))
            if (score, -source_index, -start) > (best_score, -best_source, -best_start):
                best_score = score
                best_source = source_index
                best_start = start
                best_alignments = alignments

    segments = [
        matrix[:, alignment.candidate_start : alignment.candidate_start + width]
        for matrix, alignment in zip(values, best_alignments)
    ]
    consensus = np.mean(np.stack(segments), axis=0)
    return HDAMTemplate(
        label,
        consensus,
        float(best_score),
        best_source,
        best_start,
        len(values),
    )


class HDAMTemplateMatcher:
    """Offline category-template extraction and dynamic online matching."""

    def __init__(
        self,
        bin_size: int = 1,
        aggregation: str = "binary",
        template_width: int | None = None,
        candidate_stride: int = 1,
        max_tag_shift: int = 0,
    ) -> None:
        self.bin_size = int(bin_size)
        self.aggregation = aggregation
        self.template_width = template_width
        self.candidate_stride = int(candidate_stride)
        self.max_tag_shift = int(max_tag_shift)

    def _matrix(self, episode: np.ndarray) -> np.ndarray:
        return high_density_alarm_matrix(episode, self.bin_size, self.aggregation)

    def fit(self, X: np.ndarray | Sequence[np.ndarray], y: np.ndarray) -> "HDAMTemplateMatcher":
        episodes = [np.asarray(episode, dtype=float) for episode in X]
        labels = np.asarray(y)
        if not episodes or labels.shape != (len(episodes),):
            raise ValueError("X and y must contain matching alarm-flood episodes")
        matrices = [self._matrix(episode) for episode in episodes]
        self.classes_ = np.asarray(sorted(set(labels.tolist()), key=repr))
        if self.classes_.size < 2:
            raise ValueError("at least two alarm-flood categories are required")
        self.templates_ = tuple(
            extract_hdam_template(
                [matrix for matrix, item_label in zip(matrices, labels) if item_label == label],
                label,
                self.template_width,
                self.candidate_stride,
            )
            for label in self.classes_
        )
        return self

    def similarity_scores(self, X: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        if not hasattr(self, "templates_"):
            raise RuntimeError("fit must be called before prediction")
        matrices = [self._matrix(np.asarray(episode, dtype=float)) for episode in X]
        return np.asarray(
            [
                [
                    structured_hdam_alignment(
                        template.values, matrix, self.max_tag_shift
                    ).score
                    for template in self.templates_
                ]
                for matrix in matrices
            ],
            dtype=float,
        )

    def predict_proba(self, X: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        scores = self.similarity_scores(X)
        totals = np.sum(scores, axis=1, keepdims=True)
        uniform = np.full_like(scores, 1.0 / scores.shape[1])
        return np.divide(scores, totals, out=uniform, where=totals > 0)

    def predict(self, X: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        return self.classes_[np.argmax(self.similarity_scores(X), axis=1)]

    def predict_evolution(
        self,
        X: np.ndarray,
        prefix_lengths: Sequence[int],
    ) -> dict[int, np.ndarray]:
        episodes = np.asarray(X, dtype=float)
        if episodes.ndim != 3:
            raise ValueError("X must have shape (episodes, alarm_tags, time)")
        lengths = tuple(sorted(set(int(item) for item in prefix_lengths)))
        if not lengths or lengths[0] <= 0 or lengths[-1] > episodes.shape[2]:
            raise ValueError("prefix lengths must lie within the episode duration")
        return {
            length: self.predict(episodes[:, :, :length])
            for length in lengths
        }
