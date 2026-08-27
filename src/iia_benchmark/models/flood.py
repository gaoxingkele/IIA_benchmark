from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import exp
from typing import Iterable, Sequence

import numpy as np

from iia_benchmark.data.schema import AlarmEpisode, AlarmEvent


def smith_waterman_similarity(
    first: Sequence[str],
    second: Sequence[str],
    *,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap: float = -1.0,
) -> float:
    """Normalized local sequence-alignment similarity in the interval [0, 1]."""
    if not first or not second:
        return 0.0
    table = np.zeros((len(first) + 1, len(second) + 1), dtype=float)
    best = 0.0
    for i, left in enumerate(first, start=1):
        for j, right in enumerate(second, start=1):
            diagonal = table[i - 1, j - 1] + (match if left == right else mismatch)
            table[i, j] = max(0.0, diagonal, table[i - 1, j] + gap, table[i, j - 1] + gap)
            best = max(best, table[i, j])
    return float(best / (match * min(len(first), len(second))))


def detect_alarm_floods(
    events: Iterable[AlarmEvent],
    *,
    window_seconds: float = 600.0,
    threshold: int = 10,
) -> tuple[AlarmEpisode, ...]:
    """Detect floods by unique newly activated tags in rolling windows.

    Counting unique activations makes this baseline less sensitive to chattering
    than a raw event-count criterion.  Stateful long-standing-alarm handling is a
    separate preprocessing concern because activation-only logs cannot reveal it.
    """
    activations = sorted((event for event in events if event.state == 1), key=lambda event: event.timestamp)
    if threshold < 1 or window_seconds <= 0:
        raise ValueError("threshold and window_seconds must be positive")
    windows: list[tuple[int, int]] = []
    left = 0
    for right, event in enumerate(activations):
        while event.timestamp - activations[left].timestamp > window_seconds:
            left += 1
        tags = {item.tag for item in activations[left : right + 1]}
        if len(tags) >= threshold:
            windows.append((left, right))
    if not windows:
        return ()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (min(merged[-1][0], start), max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(
        AlarmEpisode(
            episode_id=f"flood_{index:04d}",
            events=tuple(activations[start : end + 1]),
        )
        for index, (start, end) in enumerate(merged, start=1)
    )


class EmpiricalNextAlarmPredictor:
    """Book-inspired all-context next-alarm predictor with time-distance decay."""

    def __init__(self, *, distance_scale: float = 3.0) -> None:
        if distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        self.distance_scale = distance_scale

    def fit(self, sequences: Iterable[Sequence[str]]) -> "EmpiricalNextAlarmPredictor":
        self.sequences_ = tuple(tuple(sequence) for sequence in sequences if sequence)
        if not self.sequences_:
            raise ValueError("at least one non-empty sequence is required")
        self.vocabulary_ = tuple(sorted({tag for sequence in self.sequences_ for tag in sequence}))
        return self

    def predict_proba(self, current: Sequence[str]) -> dict[str, float]:
        if not hasattr(self, "sequences_"):
            raise RuntimeError("model is not fitted")
        remaining = [tag for tag in self.vocabulary_ if tag not in set(current)]
        if not remaining:
            return {}
        scores: Counter[str] = Counter()
        for historical in self.sequences_:
            positions = {tag: index for index, tag in enumerate(historical)}
            for candidate in remaining:
                if candidate not in positions:
                    continue
                candidate_position = positions[candidate]
                for context_tag in dict.fromkeys(current):
                    if context_tag in positions:
                        distance = candidate_position - positions[context_tag]
                        weight = 1.0 if distance > 0 else exp(-((abs(distance) / self.distance_scale) ** 2) / 2)
                        scores[candidate] += weight
        if not scores:
            probability = 1.0 / len(remaining)
            return {tag: probability for tag in remaining}
        total = sum(scores.values())
        return {tag: scores[tag] / total for tag in remaining}

    def predict(self, current: Sequence[str]) -> str | None:
        probabilities = self.predict_proba(current)
        return max(probabilities, key=probabilities.get) if probabilities else None


def perturb_alarm_episode(
    episode: AlarmEpisode,
    *,
    missing_probability: float = 0.0,
    spurious_count: int = 0,
    timing_jitter: float = 0.0,
    detector_delay: float = 0.0,
    spurious_tags: Sequence[str] | None = None,
    seed: int = 0,
) -> AlarmEpisode:
    """Apply reproducible AFC-RobustBench-style event-stream perturbations.

    ``detector_delay`` models a pipeline that starts extracting the episode
    after its true onset; preceding events are unavailable to the classifier.
    The remaining timestamps stay on the original clock so prefix evaluation
    can use a common clean observation horizon.
    """
    if (
        not 0 <= missing_probability <= 1
        or spurious_count < 0
        or timing_jitter < 0
        or detector_delay < 0
    ):
        raise ValueError("invalid perturbation parameters")
    rng = np.random.default_rng(seed)
    kept: list[AlarmEvent] = []
    onset = min((event.timestamp for event in episode.events), default=0.0)
    detection_time = onset + detector_delay
    for event in episode.events:
        if event.timestamp >= detection_time and rng.random() >= missing_probability:
            jitter = float(rng.normal(0.0, timing_jitter)) if timing_jitter else 0.0
            kept.append(replace(event, timestamp=max(0.0, event.timestamp + jitter)))
    if episode.events:
        low = min(max(detection_time, event.timestamp) for event in episode.events)
        high = max(event.timestamp for event in episode.events) + 1.0
        tag_pool = tuple(spurious_tags) if spurious_tags else ()
        for index in range(spurious_count):
            kept.append(
                AlarmEvent(
                    timestamp=float(rng.uniform(low, high)),
                    tag=(
                        str(rng.choice(tag_pool))
                        if tag_pool
                        else f"SPURIOUS_{index:03d}"
                    ),
                )
            )
    kept.sort(key=lambda event: event.timestamp)
    return replace(episode, episode_id=f"{episode.episode_id}_perturbed", events=tuple(kept))
