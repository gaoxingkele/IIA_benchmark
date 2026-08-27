"""Deterministic alarm-flood fingerprints with combinatorial temporal hashing.

Rao, Chen, and Shah (2025) describe an offline/online pipeline consisting of
an Alarm Evolution Matrix (AEM), localized peaks, Combinatorial Temporal
Fingerprint Hashing (CTFH), category-wise Consensus Fingerprint Profiles, and
an intra-category variability index.  The publisher exposes this architecture
but not the equation-level full text in the current environment.  This module
therefore provides a deterministic, auditable implementation of that published
pipeline while keeping paper-exact parameters and score reproduction gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Hashable

import numpy as np


@dataclass(frozen=True)
class AlarmEvolutionMatrix:
    """Short-horizon activation rates for alarm tags over sliding windows."""

    values: np.ndarray
    window_starts: np.ndarray
    window_size: int
    stride: int


@dataclass(frozen=True, order=True)
class AEMPeak:
    """One deterministic local maximum in an Alarm Evolution Matrix."""

    window_index: int
    tag_index: int
    activity: float


@dataclass(frozen=True, order=True)
class CTFHFingerprint:
    """Compact code for an anchor/target peak pair."""

    code: int
    anchor_window: int
    anchor_tag: int
    target_tag: int
    delta_window: int


@dataclass(frozen=True)
class ConsensusFingerprintProfile:
    """Majority fingerprint template and its within-class variability."""

    label: Hashable
    hashes: frozenset[int]
    variability_index: float
    sample_count: int


def alarm_evolution_matrix(
    alarm_series: np.ndarray, window_size: int = 5, stride: int = 1
) -> AlarmEvolutionMatrix:
    """Encode alarm activations as sliding-window per-tag activation rates."""

    values = np.asarray(alarm_series, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("alarm_series must have shape (alarm_tags, time)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("alarm_series must contain finite nonnegative activations")
    if window_size <= 0 or stride <= 0 or window_size > values.shape[1]:
        raise ValueError("window_size and stride must define at least one window")
    starts = np.arange(0, values.shape[1] - window_size + 1, stride, dtype=int)
    rates = np.column_stack(
        [np.mean(values[:, start : start + window_size], axis=1) for start in starts]
    )
    return AlarmEvolutionMatrix(rates, starts, int(window_size), int(stride))


def extract_aem_peaks(
    aem: AlarmEvolutionMatrix,
    threshold: float = 0.2,
    temporal_radius: int = 1,
) -> tuple[AEMPeak, ...]:
    """Extract per-tag temporal peaks with deterministic plateau handling."""

    if threshold < 0 or temporal_radius < 0:
        raise ValueError("threshold and temporal_radius must be nonnegative")
    matrix = np.asarray(aem.values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("aem.values must be a finite matrix")
    peaks: list[AEMPeak] = []
    for tag_index, row in enumerate(matrix):
        for window_index, activity in enumerate(row):
            if activity < threshold:
                continue
            left = max(0, window_index - temporal_radius)
            right = min(row.size, window_index + temporal_radius + 1)
            neighborhood = row[left:right]
            if activity < np.max(neighborhood):
                continue
            # A plateau produces one peak at its first index, keeping hashes
            # stable across platforms and iteration order.
            equal_positions = np.flatnonzero(neighborhood == activity) + left
            if equal_positions.size and window_index != int(equal_positions[0]):
                continue
            peaks.append(AEMPeak(window_index, tag_index, float(activity)))
    return tuple(sorted(peaks))


def _fingerprint_code(anchor_tag: int, target_tag: int, quantized_delta: int) -> int:
    payload = f"{anchor_tag}:{target_tag}:{quantized_delta}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)


def combinatorial_temporal_fingerprints(
    peaks: tuple[AEMPeak, ...] | list[AEMPeak],
    min_delta: int = 1,
    max_delta: int = 10,
    fanout: int = 5,
    delta_quantization: int = 1,
) -> tuple[CTFHFingerprint, ...]:
    """Pair anchor peaks with a bounded target zone and hash the combination."""

    if min_delta < 0 or max_delta < min_delta:
        raise ValueError("delta bounds are invalid")
    if fanout <= 0 or delta_quantization <= 0:
        raise ValueError("fanout and delta_quantization must be positive")
    ordered = tuple(sorted(peaks))
    fingerprints: list[CTFHFingerprint] = []
    for anchor_index, anchor in enumerate(ordered):
        candidates = [
            target
            for target in ordered[anchor_index + 1 :]
            if min_delta <= target.window_index - anchor.window_index <= max_delta
        ]
        candidates.sort(
            key=lambda item: (
                item.window_index - anchor.window_index,
                -item.activity,
                item.tag_index,
            )
        )
        for target in candidates[:fanout]:
            delta = target.window_index - anchor.window_index
            quantized = int(np.floor((delta + delta_quantization / 2) / delta_quantization))
            code = _fingerprint_code(anchor.tag_index, target.tag_index, quantized)
            fingerprints.append(
                CTFHFingerprint(
                    code,
                    anchor.window_index,
                    anchor.tag_index,
                    target.tag_index,
                    delta,
                )
            )
    return tuple(fingerprints)


def fingerprint_similarity(left: frozenset[int], right: frozenset[int]) -> float:
    """Return Jaccard similarity between two compact fingerprint sets."""

    union = left | right
    return len(left & right) / len(union) if union else 1.0


class CTFHAlarmFloodClassifier:
    """Offline consensus-profile training and online prefix classification."""

    def __init__(
        self,
        window_size: int = 5,
        stride: int = 1,
        peak_threshold: float = 0.2,
        temporal_radius: int = 1,
        min_delta: int = 1,
        max_delta: int = 10,
        fanout: int = 5,
        delta_quantization: int = 1,
        consensus_fraction: float = 0.5,
    ) -> None:
        if not 0 < consensus_fraction <= 1:
            raise ValueError("consensus_fraction must be in (0, 1]")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.peak_threshold = float(peak_threshold)
        self.temporal_radius = int(temporal_radius)
        self.min_delta = int(min_delta)
        self.max_delta = int(max_delta)
        self.fanout = int(fanout)
        self.delta_quantization = int(delta_quantization)
        self.consensus_fraction = float(consensus_fraction)

    def _validate_tensor(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must have shape (episodes, alarm_tags, time)")
        if values.shape[2] < self.window_size:
            raise ValueError("episodes are shorter than window_size")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("X must contain finite nonnegative alarm activations")
        return values

    def transform_one(self, alarm_series: np.ndarray) -> frozenset[int]:
        aem = alarm_evolution_matrix(alarm_series, self.window_size, self.stride)
        peaks = extract_aem_peaks(aem, self.peak_threshold, self.temporal_radius)
        fingerprints = combinatorial_temporal_fingerprints(
            peaks,
            self.min_delta,
            self.max_delta,
            self.fanout,
            self.delta_quantization,
        )
        return frozenset(item.code for item in fingerprints)

    def transform(self, X: np.ndarray) -> list[frozenset[int]]:
        return [self.transform_one(episode) for episode in self._validate_tensor(X)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CTFHAlarmFloodClassifier":
        fingerprints = self.transform(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != len(fingerprints):
            raise ValueError("y must contain one label per episode")
        self.classes_ = np.asarray(sorted(set(labels.tolist()), key=repr))
        if self.classes_.size < 2:
            raise ValueError("at least two alarm-flood classes are required")
        profiles: list[ConsensusFingerprintProfile] = []
        for label in self.classes_:
            samples = [item for item, item_label in zip(fingerprints, labels) if item_label == label]
            universe = set().union(*samples)
            threshold = self.consensus_fraction * len(samples)
            consensus = frozenset(
                code for code in universe if sum(code in sample for sample in samples) >= threshold
            )
            similarities = [fingerprint_similarity(sample, consensus) for sample in samples]
            variability = float(1.0 - np.mean(similarities))
            profiles.append(
                ConsensusFingerprintProfile(label, consensus, variability, len(samples))
            )
        self.profiles_ = tuple(profiles)
        return self

    def similarity_scores(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "profiles_"):
            raise RuntimeError("fit must be called before prediction")
        samples = self.transform(X)
        return np.asarray(
            [
                [fingerprint_similarity(sample, profile.hashes) for profile in self.profiles_]
                for sample in samples
            ],
            dtype=float,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.similarity_scores(X)
        totals = np.sum(scores, axis=1, keepdims=True)
        uniform = np.full_like(scores, 1.0 / scores.shape[1])
        return np.divide(scores, totals, out=uniform, where=totals > 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.similarity_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_evolution(
        self, X: np.ndarray, prefix_lengths: tuple[int, ...] | list[int]
    ) -> dict[int, np.ndarray]:
        values = self._validate_tensor(X)
        lengths = tuple(sorted(set(int(item) for item in prefix_lengths)))
        if not lengths or lengths[0] < self.window_size or lengths[-1] > values.shape[2]:
            raise ValueError("prefix lengths must lie between window_size and episode length")
        return {length: self.predict(values[:, :, :length]) for length in lengths}
