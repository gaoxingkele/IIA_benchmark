"""Root-cause algorithms from Book Chapter 4."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks


def _discrete_vector(values: Sequence[int]) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError("a one-dimensional discrete sequence is required")
    return array


def conditional_entropy(outcome: Sequence[int], *conditions: Sequence[object]) -> float:
    """Empirical H(outcome | conditions) in bits for arbitrary discrete states."""

    y = np.asarray(outcome)
    if y.ndim != 1 or not len(y):
        raise ValueError("outcome must be a non-empty vector")
    arrays = [np.asarray(condition, dtype=object) for condition in conditions]
    if any(array.ndim != 1 or len(array) != len(y) for array in arrays):
        raise ValueError("all conditions must be aligned vectors")
    if not arrays:
        _, counts = np.unique(y, return_counts=True)
        probability = counts / counts.sum()
        return float(-np.sum(probability * np.log2(probability)))
    joint_counts: dict[tuple[object, ...], int] = {}
    condition_counts: dict[tuple[object, ...], int] = {}
    for index, value in enumerate(y):
        condition = tuple(array[index] for array in arrays)
        joint = (value, *condition)
        joint_counts[joint] = joint_counts.get(joint, 0) + 1
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
    total = float(len(y))
    entropy = 0.0
    for joint, count in joint_counts.items():
        condition = joint[1:]
        entropy -= count / total * np.log2(count / condition_counts[condition])
    return float(entropy)


def _history_states(values: np.ndarray, horizon: int, indices: np.ndarray) -> np.ndarray:
    if horizon < 1:
        raise ValueError("history horizon must be positive")
    if horizon == 1:
        return values[indices]
    result = np.empty(len(indices), dtype=object)
    result[:] = [tuple(values[index - horizon + 1 : index + 1]) for index in indices]
    return result


def normalized_transfer_entropy(
    source: Sequence[int],
    target: Sequence[int],
    *,
    lag: int = 1,
    source_horizon: int = 1,
    target_horizon: int = 1,
) -> float:
    """Normalized transfer entropy in Book equation (4.6)."""

    x, y = _discrete_vector(source), _discrete_vector(target)
    if len(x) != len(y) or lag < 0:
        raise ValueError("source/target lengths must match and lag be non-negative")
    start = max(target_horizon, lag + source_horizon - 1)
    indices = np.arange(start, len(y))
    if not len(indices):
        raise ValueError("sequences are too short for the horizons and lag")
    present = y[indices]
    y_history = _history_states(y, target_horizon, indices - 1)
    x_history = _history_states(x, source_horizon, indices - lag)
    base = conditional_entropy(present, y_history)
    if base <= 1e-15:
        return 0.0
    score = discrete_transfer_entropy(
        x,
        y,
        lag=lag,
        source_horizon=source_horizon,
        target_horizon=target_horizon,
    ) / base
    return float(np.clip(score, 0.0, 1.0))


def discrete_transfer_entropy(
    source: Sequence[int],
    target: Sequence[int],
    *,
    lag: int = 1,
    source_horizon: int = 1,
    target_horizon: int = 1,
) -> float:
    """Empirical discrete TE in bits, without Chapter-4.1 normalization."""

    x, y = _discrete_vector(source), _discrete_vector(target)
    if len(x) != len(y) or lag < 0:
        raise ValueError("source/target lengths must match and lag be non-negative")
    start = max(target_horizon, lag + source_horizon - 1)
    indices = np.arange(start, len(y))
    if not len(indices):
        raise ValueError("sequences are too short for the horizons and lag")
    present = y[indices]
    y_history = _history_states(y, target_horizon, indices - 1)
    x_history = _history_states(x, source_horizon, indices - lag)
    return float(
        max(
            conditional_entropy(present, y_history)
            - conditional_entropy(present, y_history, x_history),
            0.0,
        )
    )


def normalized_direct_transfer_entropy(
    source: Sequence[int],
    target: Sequence[int],
    intermediate: Sequence[int],
    *,
    source_lag: int = 1,
    intermediate_lag: int = 1,
    source_horizon: int = 1,
    target_horizon: int = 1,
    intermediate_horizon: int = 1,
) -> float:
    """Normalized direct TE conditioned on an intermediate, Book equation (4.12)."""

    x, y, z = map(_discrete_vector, (source, target, intermediate))
    if len({len(x), len(y), len(z)}) != 1 or min(source_lag, intermediate_lag) < 0:
        raise ValueError("all sequence lengths must match and lags be non-negative")
    start = max(
        target_horizon,
        source_lag + source_horizon - 1,
        intermediate_lag + intermediate_horizon - 1,
    )
    indices = np.arange(start, len(y))
    if not len(indices):
        raise ValueError("sequences are too short")
    present = y[indices]
    yh = _history_states(y, target_horizon, indices - 1)
    xh = _history_states(x, source_horizon, indices - source_lag)
    zh = _history_states(z, intermediate_horizon, indices - intermediate_lag)
    base = conditional_entropy(present, yh, zh)
    if base <= 1e-15:
        return 0.0
    score = discrete_direct_transfer_entropy(
        x,
        y,
        z,
        source_lag=source_lag,
        intermediate_lag=intermediate_lag,
        source_horizon=source_horizon,
        target_horizon=target_horizon,
        intermediate_horizon=intermediate_horizon,
    )
    return float(np.clip(score / base, 0.0, 1.0))


def discrete_direct_transfer_entropy(
    source: Sequence[int],
    target: Sequence[int],
    intermediate: Sequence[int],
    *,
    source_lag: int = 1,
    intermediate_lag: int = 1,
    source_horizon: int = 1,
    target_horizon: int = 1,
    intermediate_horizon: int = 1,
) -> float:
    """Empirical conditional/direct TE in bits for Book Section 4.2.3."""

    x, y, z = map(_discrete_vector, (source, target, intermediate))
    if len({len(x), len(y), len(z)}) != 1 or min(source_lag, intermediate_lag) < 0:
        raise ValueError("all sequence lengths must match and lags be non-negative")
    start = max(
        target_horizon,
        source_lag + source_horizon - 1,
        intermediate_lag + intermediate_horizon - 1,
    )
    indices = np.arange(start, len(y))
    if not len(indices):
        raise ValueError("sequences are too short")
    present = y[indices]
    yh = _history_states(y, target_horizon, indices - 1)
    xh = _history_states(x, source_horizon, indices - source_lag)
    zh = _history_states(z, intermediate_horizon, indices - intermediate_lag)
    return float(
        max(
            conditional_entropy(present, yh, zh)
            - conditional_entropy(present, yh, zh, xh),
            0.0,
        )
    )


def bernoulli_surrogate_threshold(
    source: Sequence[int],
    target: Sequence[int],
    *,
    max_lag: int,
    simulations: int = 99,
    significance: float = 0.05,
    seed: int = 0,
) -> float:
    """Book Sec. 4.1.3 Bernoulli Monte-Carlo significance threshold."""

    x, y = _discrete_vector(source), _discrete_vector(target)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(simulations):
        xs = rng.binomial(1, np.mean(x), len(x))
        ys = rng.binomial(1, np.mean(y), len(y))
        scores.append(max(normalized_transfer_entropy(xs, ys, lag=lag) for lag in range(max_lag + 1)))
    return float(np.quantile(scores, 1.0 - significance)) if scores else 0.0


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    score: float
    lag: int
    threshold: float
    direct: bool


@dataclass
class NormalizedTransferEntropyGraph:
    max_lag: int = 10
    simulations: int = 99
    significance: float = 0.05
    minimum_occurrences: int = 50
    seed: int = 0

    def infer(self, series: Mapping[str, Sequence[int]]) -> list[CausalEdge]:
        arrays = {name: _discrete_vector(values) for name, values in series.items()}
        if len({len(values) for values in arrays.values()}) != 1:
            raise ValueError("all series lengths must match")
        edges = []
        for source, target in product(arrays, repeat=2):
            if source == target or min(np.sum(arrays[source]), np.sum(arrays[target])) < self.minimum_occurrences:
                continue
            scores = [normalized_transfer_entropy(arrays[source], arrays[target], lag=lag) for lag in range(self.max_lag + 1)]
            lag = int(np.argmax(scores))
            threshold = bernoulli_surrogate_threshold(
                arrays[source], arrays[target], max_lag=self.max_lag,
                simulations=self.simulations, significance=self.significance,
                seed=self.seed + len(edges),
            )
            if scores[lag] > threshold:
                edges.append(CausalEdge(source, target, float(scores[lag]), lag, threshold, True))
        direct_edges = []
        for edge in edges:
            intermediates = [
                node for node in arrays
                if node not in {edge.source, edge.target}
                and any(item.source == edge.source and item.target == node for item in edges)
                and any(item.source == node and item.target == edge.target for item in edges)
            ]
            direct = True
            for node in intermediates:
                ndte = normalized_direct_transfer_entropy(
                    arrays[edge.source], arrays[edge.target], arrays[node],
                    source_lag=edge.lag, intermediate_lag=1,
                )
                if ndte <= edge.threshold:
                    direct = False
                    break
            direct_edges.append(CausalEdge(edge.source, edge.target, edge.score, edge.lag, edge.threshold, direct))
        return direct_edges


def information_granules(values: Sequence[float], window_size: int) -> np.ndarray:
    """Create fuzzy lower-support/core/upper-support granules, equations (4.23)-(4.24)."""

    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or window_size < 2 or len(samples) < window_size:
        raise ValueError("valid values and window_size are required")
    count = len(samples) // window_size
    windows = np.sort(
        samples[: count * window_size].reshape(count, window_size), axis=1
    )
    cores = np.median(windows, axis=1)
    half = window_size // 2
    lower_values = windows[:, :half]
    upper_values = windows[:, half if window_size % 2 == 0 else half + 1 :]
    lower_spread = np.sqrt(np.mean((lower_values - cores[:, None]) ** 2, axis=1))
    upper_spread = np.sqrt(np.mean((upper_values - cores[:, None]) ** 2, axis=1))
    return np.column_stack([cores - lower_spread, cores, cores + upper_spread])


def cluster_information_granules(
    granules: Iterable[Iterable[float]], *, min_samples: int = 5, xi: float = 0.05
) -> np.ndarray:
    """OPTICS cluster labels used as the discrete PDF support in Book Sec. 4.2.2."""

    try:
        from sklearn.cluster import OPTICS
    except ImportError as exc:  # pragma: no cover - optional dependency route
        raise RuntimeError("IGTE requires the project 'ml' extra (scikit-learn)") from exc
    matrix = np.asarray(list(granules), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or len(matrix) < 2:
        raise ValueError("granules must be a non-empty matrix of three-dimensional features")
    neighborhood = min(min_samples, max(2, len(matrix) // 2))
    optics = OPTICS(
        min_samples=neighborhood,
        xi=xi,
        cluster_method="dbscan",
        eps=np.inf,
    ).fit(matrix)
    ordering = np.asarray(optics.ordering_, dtype=int)
    reachability = np.asarray(optics.reachability_[ordering], dtype=float)
    finite = reachability[np.isfinite(reachability)]
    if not len(finite):
        labels = np.zeros(len(matrix), dtype=int)
    else:
        profile = reachability.copy()
        profile[~np.isfinite(profile)] = float(np.max(finite))
        prominence = max(float(np.std(finite)) * max(5.0 * xi, 0.1), 1e-12)
        peaks, _ = find_peaks(
            profile,
            prominence=prominence,
            distance=max(2, neighborhood),
        )
        boundaries = [0, *[int(peak) for peak in peaks if 0 < peak < len(ordering)], len(ordering)]
        labels = np.empty(len(matrix), dtype=int)
        for cluster, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            labels[ordering[start:stop]] = cluster
    cluster_order = sorted(
        set(labels), key=lambda label: float(np.median(matrix[labels == label, 1]))
    )
    trend_preserving = {label: rank for rank, label in enumerate(cluster_order)}
    return np.asarray([trend_preserving[label] for label in labels], dtype=int)


def information_granulation_transfer_entropy(
    source: Sequence[float],
    target: Sequence[float],
    *,
    window_size: int,
    lag: int = 1,
    order: int = 2,
    min_samples: int = 5,
    normalized: bool = False,
) -> float:
    x = cluster_information_granules(information_granules(source, window_size), min_samples=min_samples)
    y = cluster_information_granules(information_granules(target, window_size), min_samples=min_samples)
    scorer = normalized_transfer_entropy if normalized else discrete_transfer_entropy
    return scorer(x, y, lag=lag, source_horizon=order, target_horizon=order)


def information_granulation_direct_transfer_entropy(
    source: Sequence[float],
    target: Sequence[float],
    intermediate: Sequence[float],
    *,
    window_size: int,
    lag: int = 1,
    source_lag: int | None = None,
    intermediate_lag: int | None = None,
    order: int = 2,
    min_samples: int = 5,
    normalized: bool = False,
) -> float:
    x = cluster_information_granules(information_granules(source, window_size), min_samples=min_samples)
    y = cluster_information_granules(information_granules(target, window_size), min_samples=min_samples)
    z = cluster_information_granules(information_granules(intermediate, window_size), min_samples=min_samples)
    scorer = (
        normalized_direct_transfer_entropy
        if normalized
        else discrete_direct_transfer_entropy
    )
    resolved_source_lag = lag if source_lag is None else int(source_lag)
    resolved_intermediate_lag = lag if intermediate_lag is None else int(intermediate_lag)
    return scorer(
        x, y, z,
        source_lag=resolved_source_lag,
        intermediate_lag=resolved_intermediate_lag,
        source_horizon=order, target_horizon=order, intermediate_horizon=order,
    )


def clustered_surrogate_threshold(
    source_labels: Sequence[int],
    target_labels: Sequence[int],
    *,
    lag: int = 1,
    order: int = 2,
    simulations: int = 19,
    significance: float = 0.1,
    seed: int = 0,
) -> float:
    """Permutation threshold on clustered granule features (Book Sec. 4.2.3)."""

    if simulations < 1 or not 0.0 < significance < 1.0:
        raise ValueError("positive simulations and significance in (0, 1) are required")
    x, y = _discrete_vector(source_labels), _discrete_vector(target_labels)
    if len(x) != len(y):
        raise ValueError("clustered sequences must be aligned")
    rng = np.random.default_rng(seed)
    scores = [
        discrete_transfer_entropy(
            rng.permutation(x),
            y,
            lag=lag,
            source_horizon=order,
            target_horizon=order,
        )
        for _ in range(simulations)
    ]
    return float(np.quantile(scores, 1.0 - significance))


@dataclass
class RecursiveBayesianAlarmRCA:
    """Recursive binary Bayesian network in Book equations (4.35)-(4.52)."""

    cause_names: Sequence[str]
    response_time_samples: int = 15
    initial_probability: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.cause_names
            or self.response_time_samples < 1
            or not 0.0 <= self.initial_probability <= 0.5
        ):
            raise ValueError(
                "cause_names, positive response_time_samples, and an initial probability "
                "in [0, 0.5] are required"
            )
        self.update_rate = 1.0 - 0.5 ** (1.0 / self.response_time_samples)
        self.patterns_ = [pattern for pattern in product((0, 1), repeat=len(self.cause_names)) if any(pattern)]
        self.patterns_.append(tuple(0 for _ in self.cause_names))
        self.cause_probabilities_ = np.full(
            (len(self.cause_names), 2), self.initial_probability, dtype=float
        )
        self.alarm_conditionals_ = np.full(
            (len(self.patterns_), 2), self.initial_probability, dtype=float
        )
        self.alarm_probability_ = np.full(2, self.initial_probability, dtype=float)

    def update(self, cause_states: Sequence[int], alarm_state: int) -> None:
        pattern = tuple(map(int, cause_states))
        if pattern not in self.patterns_ or alarm_state not in (0, 1):
            raise ValueError("binary cause and alarm states are required")
        rate = self.update_rate
        for index, state in enumerate(pattern):
            self.cause_probabilities_[index] *= 1.0 - rate
            self.cause_probabilities_[index, state] += rate
        pattern_index = self.patterns_.index(pattern)
        self.alarm_conditionals_[pattern_index] *= 1.0 - rate
        self.alarm_conditionals_[pattern_index, alarm_state] += rate
        self.alarm_probability_ *= 1.0 - rate
        self.alarm_probability_[alarm_state] += rate

    def posterior_patterns(self, alarm_state: int = 1) -> np.ndarray:
        weights = []
        for index, pattern in enumerate(self.patterns_):
            prior = np.prod([self.cause_probabilities_[cause, state] for cause, state in enumerate(pattern)])
            weights.append(prior * self.alarm_conditionals_[index, alarm_state])
        weights = np.asarray(weights, dtype=float)
        return weights / np.sum(weights) if np.sum(weights) else np.full(len(weights), 1.0 / len(weights))

    def root_cause(self) -> tuple[str, ...]:
        if self.alarm_probability_[0] > self.alarm_probability_[1]:
            return ()
        pattern = self.patterns_[int(np.argmax(self.posterior_patterns(1)))]
        if not any(pattern):
            return ("unknown",)
        return tuple(name for name, active in zip(self.cause_names, pattern, strict=True) if active)

    def infer_sequence(
        self, cause_states: Iterable[Sequence[int]], alarm_states: Sequence[int]
    ) -> tuple[tuple[str, ...], ...]:
        """Update recursively and return the online decision at every sample."""

        causes = list(cause_states)
        alarms = list(alarm_states)
        if len(causes) != len(alarms):
            raise ValueError("cause and alarm sequences must have equal lengths")
        decisions = []
        for states, alarm in zip(causes, alarms, strict=True):
            self.update(states, int(alarm))
            decisions.append(self.root_cause())
        return tuple(decisions)


@dataclass(frozen=True)
class LinearSegment:
    start: int
    stop: int
    intercept: float
    slope: float
    residual_variance: float
    amplitude: float


def _linear_segment(values: np.ndarray, start: int, stop: int) -> LinearSegment:
    time = np.arange(start, stop, dtype=float)
    design = np.column_stack([np.ones(len(time)), time])
    coefficients = np.linalg.lstsq(design, values[start:stop], rcond=None)[0]
    fitted = design @ coefficients
    variance = float(np.sum((values[start:stop] - fitted) ** 2) / max(len(time) - 2, 1))
    amplitude = float(coefficients[1] * (stop - start - 1))
    return LinearSegment(start, stop, float(coefficients[0]), float(coefficients[1]), variance, amplitude)


def piecewise_linear_representation(
    values: Sequence[float], *, max_segments: int = 6, min_size: int = 8
) -> list[LinearSegment]:
    """Dynamic-programming PLR with BIC selection for Book equations (4.53)-(4.61)."""

    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < min_size or max_segments < 1:
        raise ValueError("insufficient values or invalid max_segments")
    costs = np.full((n + 1, n + 1), np.inf)
    for start in range(n):
        for stop in range(start + min_size, n + 1):
            segment = _linear_segment(y, start, stop)
            costs[start, stop] = segment.residual_variance * max(stop - start - 2, 1)
    best_solution = None
    best_bic = float("inf")
    for count in range(1, min(max_segments, n // min_size) + 1):
        dp = np.full((count + 1, n + 1), np.inf)
        paths: list[list[list[int] | None]] = [[None] * (n + 1) for _ in range(count + 1)]
        dp[0, 0], paths[0][0] = 0.0, [0]
        for k in range(1, count + 1):
            for stop in range(k * min_size, n + 1):
                for start in range((k - 1) * min_size, stop - min_size + 1):
                    value = dp[k - 1, start] + costs[start, stop]
                    if value < dp[k, stop]:
                        dp[k, stop] = value
                        paths[k][stop] = [*paths[k - 1][start], stop] if paths[k - 1][start] else None
        if not np.isfinite(dp[count, n]):
            continue
        bic = n * np.log(max(dp[count, n] / n, 1e-15)) + (3 * count) * np.log(n)
        if bic < best_bic:
            best_bic, best_solution = bic, paths[count][n]
    if not best_solution:
        raise RuntimeError("PLR optimization failed")
    return [_linear_segment(y, start, stop) for start, stop in zip(best_solution[:-1], best_solution[1:], strict=True)]


def lagged_correlation_delay(source: Sequence[float], target: Sequence[float], *, max_lag: int) -> tuple[int, float, float]:
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    scores = []
    for lag in range(max_lag + 1):
        left, right = (x[: len(x) - lag], y[lag:]) if lag else (x, y)
        scores.append(float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else 0.0)
    lag = int(np.argmax(np.abs(scores)))
    n = max(len(x) - lag, 1)
    threshold = 1.85 * n ** -0.41 + 2.37 * n ** -0.53
    return lag, scores[lag], float(threshold)


def piecewise_lagged_correlation_delay(
    source: Sequence[float],
    target: Sequence[float],
    segments: Sequence[LinearSegment],
    *,
    max_lag: int,
) -> tuple[int, tuple[dict[str, float | int | bool], ...]]:
    """Book equations (4.67)-(4.72): significant segment-weighted delay.

    A positive delay means ``source[t-delay]`` explains ``target[t]``.  This
    convention is consistent with the numerical example in equations
    (4.90)-(4.93), where ``y(t)`` uses ``x1(t-10)`` and ``x2(t-8)``.
    """

    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or max_lag < 0:
        raise ValueError("aligned one-dimensional inputs and nonnegative max_lag are required")
    evidence: list[dict[str, float | int | bool]] = []
    weighted_lags, weights = [], []
    for segment_index, segment in enumerate(segments):
        scores = []
        for lag in range(max_lag + 1):
            target_indices = np.arange(max(segment.start, lag), segment.stop)
            if len(target_indices) < 3:
                scores.append(0.0)
                continue
            left, right = x[target_indices - lag], y[target_indices]
            score = (
                float(np.corrcoef(left, right)[0, 1])
                if np.std(left) > 0 and np.std(right) > 0
                else 0.0
            )
            scores.append(score if np.isfinite(score) else 0.0)
        lag = int(np.argmax(np.abs(scores)))
        correlation = float(scores[lag])
        sample_count = max(segment.stop - max(segment.start, lag), 1)
        threshold = float(1.85 * sample_count ** -0.41 + 2.37 * sample_count ** -0.53)
        significant = abs(correlation) >= threshold
        evidence.append(
            {
                "segment_index": segment_index,
                "lag": lag,
                "correlation": correlation,
                "threshold": threshold,
                "significant": significant,
            }
        )
        if significant:
            weighted_lags.append(lag * abs(correlation))
            weights.append(abs(correlation))
    if not weights:
        return 0, tuple(evidence)
    delay = int(np.rint(np.sum(weighted_lags) / np.sum(weights)))
    return delay, tuple(evidence)


@dataclass(frozen=True)
class SegmentContribution:
    segment: LinearSegment
    target_trend: int
    source_trends: tuple[int, ...]
    lags: tuple[int, ...]
    factors: np.ndarray


@dataclass
class PLRContributionRCA:
    max_segments: int = 6
    min_size: int = 8
    max_lag: int = 20

    def analyze(
        self,
        predictors: Iterable[Iterable[float]],
        target: Sequence[float],
        *,
        segment_boundaries: Sequence[int] | None = None,
    ) -> list[SegmentContribution]:
        x, y = np.asarray(list(predictors), dtype=float), np.asarray(target, dtype=float)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("predictors and target must be aligned")
        if segment_boundaries is None:
            segments = piecewise_linear_representation(
                y, max_segments=self.max_segments, min_size=self.min_size
            )
        else:
            boundaries = tuple(int(value) for value in segment_boundaries)
            if (
                len(boundaries) < 2
                or boundaries[0] != 0
                or boundaries[-1] != len(y)
                or any(stop - start < self.min_size for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True))
            ):
                raise ValueError("segment boundaries must cover the target with valid segment sizes")
            segments = [
                _linear_segment(y, start, stop)
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
            ]
        lags = tuple(
            piecewise_lagged_correlation_delay(
                x[:, index], y, segments, max_lag=self.max_lag
            )[0]
            for index in range(x.shape[1])
        )
        results = []
        for segment in segments:
            length = segment.stop - segment.start
            target_threshold = max(2.0 * np.sqrt(segment.residual_variance), 1e-12)
            target_trend = int(np.sign(segment.amplitude)) if abs(segment.amplitude) >= target_threshold else 0
            columns, source_trends, amplitudes, scales = [], [], [], []
            for index, lag in enumerate(lags):
                source_indices = np.clip(
                    np.arange(segment.start, segment.stop) - lag, 0, len(x) - 1
                )
                raw = x[source_indices, index]
                local = _linear_segment(raw, 0, len(raw))
                scale = max(2.0 * np.sqrt(local.residual_variance), 1e-12)
                trend = int(np.sign(local.amplitude)) if abs(local.amplitude) >= scale else 0
                sign_relation = trend * target_trend
                columns.append(sign_relation * (raw - np.mean(raw)) / scale)
                source_trends.append(trend)
                amplitudes.append(local.amplitude)
                scales.append(scale)
            normalized_target = (y[segment.start : segment.stop] - np.mean(y[segment.start : segment.stop])) / target_threshold
            design = np.column_stack([np.ones(length), *columns])
            coefficients = nnls(design, normalized_target)[0][1:]
            contributions = np.abs(coefficients * target_threshold / np.asarray(scales) * np.asarray(amplitudes))
            factors = contributions / np.sum(contributions) if np.sum(contributions) else np.zeros_like(contributions)
            results.append(SegmentContribution(segment, target_trend, tuple(source_trends), lags, factors))
        return results
