"""Alarm-flood analysis algorithms from Book Chapter 5."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CriterionCResult:
    sample_indices: np.ndarray
    attention_sets: tuple[frozenset[str], ...]
    cardinality: np.ndarray
    raw_detection: np.ndarray
    delayed_detection: np.ndarray


def criterion_c_alarm_flood_detection(
    alarm_states: Iterable[Iterable[int]],
    *,
    tag_names: Sequence[str] | None = None,
    attention_window: int = 600,
    long_standing_window: int = 1800,
    update_step: int = 1,
    threshold: int = 10,
    delay_samples: int = 1,
) -> CriterionCResult:
    """Book Criterion C and delay state machine, equations (5.5)-(5.7)."""

    states = np.asarray(list(alarm_states), dtype=np.int8)
    if states.ndim != 2 or not np.isin(states, [0, 1]).all():
        raise ValueError("alarm_states must be a binary time-by-tag matrix")
    if min(attention_window, long_standing_window, update_step, threshold, delay_samples) < 1:
        raise ValueError("window, step, threshold, and delay parameters must be positive")
    names = tuple(tag_names or [f"tag_{index}" for index in range(states.shape[1])])
    if len(names) != states.shape[1]:
        raise ValueError("tag_names length differs from the number of columns")
    activations = np.maximum(states - np.vstack([np.zeros((1, states.shape[1]), dtype=np.int8), states[:-1]]), 0)
    sample_indices = np.arange(attention_window - 1, len(states), update_step)
    previous: set[int] = set()
    attention_sets, cardinality, raw = [], [], []
    for end in sample_indices:
        start = end - attention_window + 1
        recent_states = states[start : end + 1]
        i1 = set(np.flatnonzero(np.any(activations[start : end + 1], axis=0)))
        i2 = set(np.flatnonzero(np.all(recent_states == 1, axis=0)))
        if end + 1 >= long_standing_window:
            i3 = set(np.flatnonzero(np.all(states[end - long_standing_window + 1 : end + 1] == 1, axis=0)))
        else:
            i3 = set()
        current = i1 | ((previous & i2) - i3)
        previous = current
        attention_sets.append(frozenset(names[index] for index in current))
        cardinality.append(len(current))
        raw.append(int(len(current) >= threshold))
    delayed = np.zeros(len(raw), dtype=np.int8)
    active, run = False, 0
    for index, value in enumerate(raw):
        if value == int(active):
            run = 0
        else:
            run += 1
            if run >= delay_samples:
                active = not active
                run = 0
        delayed[index] = int(active)
    return CriterionCResult(
        sample_indices,
        tuple(attention_sets),
        np.asarray(cardinality),
        np.asarray(raw, dtype=np.int8),
        delayed,
    )


@dataclass(frozen=True)
class AlarmToken:
    tag: str
    timestamp: float
    priority: int = 1


@dataclass(frozen=True)
class AlignmentSeed:
    first_start: int
    second_start: int
    length: int
    score: float


@dataclass(frozen=True)
class AlarmAlignmentResult:
    score: float
    similarity: float
    pre_match_similarity: float
    seeds: tuple[AlignmentSeed, ...]
    aligned_pairs: tuple[tuple[int, int], ...]
    cells_evaluated: int


def _jaccard(first: set[Hashable] | frozenset[Hashable], second: set[Hashable] | frozenset[Hashable]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def priority_match_score(priority: int, priority_levels: int, *, base: float = 3.0, increment: float = 1.5) -> float:
    if not 1 <= priority <= priority_levels:
        raise ValueError("priority must lie in [1, priority_levels]")
    return float(base + increment * (priority_levels - priority))


def _alignment_pair_score(
    left: AlarmToken,
    right: AlarmToken,
    first_start: float,
    second_start: float,
    *,
    priority_levels: int,
    mismatch: float,
    time_tolerance: float,
) -> float:
    if left.tag != right.tag:
        return mismatch
    priority = min(left.priority, right.priority)
    score = priority_match_score(priority, priority_levels)
    if time_tolerance > 0:
        difference = (left.timestamp - first_start) - (right.timestamp - second_start)
        score *= float(np.exp(-(difference**2) / (2.0 * time_tolerance**2)))
    return score


def accelerated_alarm_alignment(
    first: Sequence[AlarmToken],
    second: Sequence[AlarmToken],
    *,
    priority_levels: int = 3,
    mismatch: float = -2.5,
    gap: float = -1.0,
    pre_match_threshold: float = 0.0,
    seed_length: int = 2,
    max_seeds: int = 10,
    extension_band: int = 20,
    time_tolerance: float = 0.0,
) -> AlarmAlignmentResult:
    """Priority-aware BLAST-like local alignment from Book Sec. 5.2."""

    if not first or not second:
        return AlarmAlignmentResult(0.0, 0.0, 0.0, (), (), 0)
    if mismatch >= 2 * gap or gap >= 0:
        raise ValueError("scores must satisfy mismatch < 2*gap < 0")
    pre_match = _jaccard({item.tag for item in first}, {item.tag for item in second})
    if pre_match < pre_match_threshold:
        return AlarmAlignmentResult(0.0, 0.0, pre_match, (), (), 0)
    seeds = []
    for i in range(len(first) - seed_length + 1):
        first_tags = tuple(item.tag for item in first[i : i + seed_length])
        for j in range(len(second) - seed_length + 1):
            if first_tags != tuple(item.tag for item in second[j : j + seed_length]):
                continue
            score = sum(
                _alignment_pair_score(
                    first[i + offset], second[j + offset], first[0].timestamp, second[0].timestamp,
                    priority_levels=priority_levels, mismatch=mismatch, time_tolerance=time_tolerance,
                )
                for offset in range(seed_length)
            )
            seeds.append(AlignmentSeed(i, j, seed_length, score))
    seeds = sorted(seeds, key=lambda item: item.score, reverse=True)[:max_seeds]
    allowed = None
    if seeds:
        allowed = np.zeros((len(first) + 1, len(second) + 1), dtype=bool)
        for seed in seeds:
            diagonal = seed.second_start - seed.first_start
            for i in range(1, len(first) + 1):
                low = max(1, i + diagonal - extension_band)
                high = min(len(second), i + diagonal + extension_band)
                allowed[i, low : high + 1] = True
    table = np.zeros((len(first) + 1, len(second) + 1))
    trace = np.zeros_like(table, dtype=np.int8)
    best_score, best_cell, cells = 0.0, (0, 0), 0
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            if allowed is not None and not allowed[i, j]:
                continue
            cells += 1
            pair = _alignment_pair_score(
                first[i - 1], second[j - 1], first[0].timestamp, second[0].timestamp,
                priority_levels=priority_levels, mismatch=mismatch, time_tolerance=time_tolerance,
            )
            choices = (0.0, table[i - 1, j - 1] + pair, table[i - 1, j] + gap, table[i, j - 1] + gap)
            trace[i, j] = int(np.argmax(choices))
            table[i, j] = choices[trace[i, j]]
            if table[i, j] > best_score:
                best_score, best_cell = float(table[i, j]), (i, j)
    pairs = []
    i, j = best_cell
    while i > 0 and j > 0 and table[i, j] > 0:
        move = trace[i, j]
        if move == 1:
            if first[i - 1].tag == second[j - 1].tag:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break
    maximum = sum(
        priority_match_score(item.priority, priority_levels)
        for item in sorted((first if len(first) <= len(second) else second), key=lambda token: token.priority)
    )
    similarity = float(np.clip(best_score / maximum, 0.0, 1.0)) if maximum else 0.0
    return AlarmAlignmentResult(best_score, similarity, pre_match, tuple(seeds), tuple(reversed(pairs)), cells)


@dataclass(frozen=True)
class ClosedAlarmPattern:
    items: frozenset[str]
    transaction_ids: frozenset[int]
    support: float


def charm_closed_alarm_patterns(
    transactions: Iterable[Iterable[str]], *, minimum_support: float | int
) -> tuple[ClosedAlarmPattern, ...]:
    """Vertical-TID closed frequent pattern mining (Book Algorithms 5.5-5.6)."""

    database = [frozenset(transaction) for transaction in transactions]
    if not database:
        raise ValueError("at least one transaction is required")
    minimum_count = int(np.ceil(minimum_support * len(database))) if isinstance(minimum_support, float) else int(minimum_support)
    if minimum_count < 1:
        raise ValueError("minimum_support must select at least one transaction")
    item_tids: dict[str, frozenset[int]] = {}
    for tid, transaction in enumerate(database):
        for item in transaction:
            item_tids[item] = item_tids.get(item, frozenset()) | frozenset({tid})
    items = sorted((item, tids) for item, tids in item_tids.items() if len(tids) >= minimum_count)
    frequent: dict[frozenset[str], frozenset[int]] = {}

    def extend(prefix: frozenset[str], candidates: list[tuple[str, frozenset[int]]]) -> None:
        for index, (item, tids) in enumerate(candidates):
            pattern = prefix | {item}
            frequent[pattern] = tids
            suffix = []
            for next_item, next_tids in candidates[index + 1 :]:
                intersection = tids & next_tids
                if len(intersection) >= minimum_count:
                    suffix.append((next_item, intersection))
            if suffix:
                extend(pattern, suffix)

    extend(frozenset(), items)
    closed = []
    for pattern, tids in frequent.items():
        if len(pattern) < 2:
            continue
        if any(pattern < other and tids == other_tids for other, other_tids in frequent.items()):
            continue
        closed.append(ClosedAlarmPattern(pattern, tids, len(tids) / len(database)))
    return tuple(sorted(closed, key=lambda item: (-item.support, -len(item.items), sorted(item.items))))


@dataclass(frozen=True)
class RepresentativeAlarmPattern:
    items: frozenset[str]
    transaction_ids: frozenset[int]
    descendants: tuple[ClosedAlarmPattern, ...]


def representative_alarm_patterns(
    patterns: Sequence[ClosedAlarmPattern], *, similarity_threshold: float = 1.0 / 3.0
) -> tuple[RepresentativeAlarmPattern, ...]:
    """Modified delta-cluster and greedy cover, Book equations (5.42)-(5.46)."""

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1]")
    neighborhoods = [
        frozenset(j for j, right in enumerate(patterns) if _jaccard(left.items, right.items) >= similarity_threshold)
        for left in patterns
    ]
    uncovered = set(range(len(patterns)))
    selected = []
    while uncovered:
        index = max(range(len(neighborhoods)), key=lambda item: len(neighborhoods[item] & uncovered))
        group = neighborhoods[index] & uncovered
        if not group:
            group = frozenset({min(uncovered)})
        selected.append(group)
        uncovered -= group
    representatives = []
    for group in selected:
        descendants = tuple(patterns[index] for index in sorted(group))
        representatives.append(
            RepresentativeAlarmPattern(
                frozenset().union(*(item.items for item in descendants)),
                frozenset().union(*(item.transaction_ids for item in descendants)),
                descendants,
            )
        )
    return tuple(representatives)


@dataclass
class MaximumEntropyNextAlarmPredictor:
    """Log-linear maximum-entropy next-alarm model for Book Sec. 5.4."""

    time_scale: float = 3.0
    regularization: float = 1e-6
    max_iterations: int = 500

    def _unpack(self, event: str | AlarmToken, fallback_time: float) -> tuple[str, float]:
        return (event.tag, event.timestamp) if isinstance(event, AlarmToken) else (str(event), fallback_time)

    def fit(self, sequences: Iterable[Sequence[str | AlarmToken]]) -> "MaximumEntropyNextAlarmPredictor":
        self.sequences_ = tuple(tuple(sequence) for sequence in sequences if len(sequence) >= 2)
        if not self.sequences_ or self.time_scale <= 0:
            raise ValueError("non-empty sequences and positive time_scale are required")
        unpacked = [tuple(self._unpack(event, index) for index, event in enumerate(sequence)) for sequence in self.sequences_]
        self.vocabulary_ = tuple(sorted({tag for sequence in unpacked for tag, _ in sequence}))
        index = {tag: position for position, tag in enumerate(self.vocabulary_)}
        forward = np.zeros((len(index), len(index)))
        reverse = np.zeros_like(forward)
        context_count = np.zeros(len(index))
        training = []
        for sequence in unpacked:
            for left_index, (left_tag, left_time) in enumerate(sequence[:-1]):
                context_count[index[left_tag]] += 1
                training.append((index[left_tag], index[sequence[left_index + 1][0]]))
                for right_index, (right_tag, right_time) in enumerate(sequence):
                    if right_index > left_index:
                        forward[index[left_tag], index[right_tag]] += 1
                    elif right_index < left_index:
                        distance = abs(right_time - left_time)
                        reverse[index[left_tag], index[right_tag]] += np.exp(-(distance**2) / (2 * self.time_scale**2))
        denominator = max(len(unpacked), 1)
        self.features_ = np.stack([forward / denominator, reverse / denominator], axis=2)

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            loss, gradient = 0.0, np.zeros_like(weights)
            for context, observed in training:
                logits = self.features_[context] @ weights
                logits -= np.max(logits)
                probabilities = np.exp(logits) / np.sum(np.exp(logits))
                loss -= np.log(max(probabilities[observed], 1e-15))
                gradient += probabilities @ self.features_[context] - self.features_[context, observed]
            loss += 0.5 * self.regularization * float(weights @ weights)
            gradient += self.regularization * weights
            return loss, gradient

        result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iterations})
        if not result.success:
            raise RuntimeError(f"maximum-entropy optimization failed: {result.message}")
        self.weights_ = result.x
        self.context_frequency_ = context_count / max(np.sum(context_count), 1.0)
        return self

    def predict_proba(self, current: Sequence[str | AlarmToken]) -> dict[str, float]:
        if not hasattr(self, "weights_"):
            raise RuntimeError("model is not fitted")
        current_tags = [self._unpack(event, index)[0] for index, event in enumerate(current)]
        candidate_indices = [index for index, tag in enumerate(self.vocabulary_) if tag not in set(current_tags)]
        if not candidate_indices:
            return {}
        vocabulary_index = {tag: index for index, tag in enumerate(self.vocabulary_)}
        context_indices = [vocabulary_index[tag] for tag in dict.fromkeys(current_tags) if tag in vocabulary_index]
        if not context_indices:
            probability = 1.0 / len(candidate_indices)
            return {self.vocabulary_[index]: probability for index in candidate_indices}
        aggregate = np.zeros(len(self.vocabulary_))
        weights = self.context_frequency_[context_indices]
        weights = weights / np.sum(weights) if np.sum(weights) else np.full(len(weights), 1.0 / len(weights))
        for context, context_weight in zip(context_indices, weights, strict=True):
            logits = self.features_[context] @ self.weights_
            logits -= np.max(logits[candidate_indices])
            candidate_probability = np.exp(logits[candidate_indices])
            candidate_probability /= np.sum(candidate_probability)
            aggregate[candidate_indices] += context_weight * candidate_probability
        return {self.vocabulary_[index]: float(aggregate[index]) for index in candidate_indices}

    def predict(self, current: Sequence[str | AlarmToken]) -> str | None:
        probabilities = self.predict_proba(current)
        return max(probabilities, key=probabilities.get) if probabilities else None
