"""Equation-level implementations from Book Chapter 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_distribution
from scipy.stats import rankdata


Direction = Literal["high", "low"]


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _geometric_sum(probability: float, terms: int) -> float:
    if np.isclose(probability, 1.0):
        return float(terms)
    return float((1.0 - probability**terms) / (1.0 - probability))


@dataclass(frozen=True)
class AlarmOnOffDelay:
    """Symmetric n-sample alarm on/off delay from Book equation (2.4)."""

    threshold: float
    direction: Direction = "high"
    delay: int = 1

    def __post_init__(self) -> None:
        if self.direction not in ("high", "low") or self.delay < 1:
            raise ValueError("direction must be high/low and delay must be positive")

    def predict(self, values: Iterable[float]) -> np.ndarray:
        samples = np.asarray(list(values), dtype=float)
        beyond = threshold_states(samples, self.threshold, self.direction)
        result = np.zeros(len(samples), dtype=np.int8)
        active = False
        counter = 0
        for index, current in enumerate(beyond):
            evidence_for_change = bool(current) if not active else not bool(current)
            counter = counter + 1 if evidence_for_change else 0
            if counter >= self.delay:
                active = not active
                counter = 0
            result[index] = int(active)
        return result


@dataclass(frozen=True)
class IIDAlarmPerformance:
    false_alarm_rate: float
    missed_alarm_rate: float
    average_alarm_delay: float


@dataclass(frozen=True)
class IIDDelayDesignResult:
    threshold: float
    delay: int
    performance: IIDAlarmPerformance
    normal_exceedance_probability: float
    abnormal_exceedance_probability: float
    loss: float


def iid_delay_timer_performance(
    normal_exceedance_probability: float,
    abnormal_exceedance_probability: float,
    delay: int,
    *,
    sample_period: float = 1.0,
) -> IIDAlarmPerformance:
    """Evaluate Book equations (2.8), (2.15), and (2.16)."""

    q1 = _probability(normal_exceedance_probability, "normal probability")
    p1 = _probability(abnormal_exceedance_probability, "abnormal probability")
    if delay < 1 or sample_period <= 0:
        raise ValueError("delay and sample_period must be positive")
    n, q2, p2 = int(delay), 1.0 - q1, 1.0 - p1
    far_num = q1**n * _geometric_sum(q2, n)
    far_den = far_num + q2**n * _geometric_sum(q1, n)
    mar_num = p2**n * _geometric_sum(p1, n)
    mar_den = mar_num + p1**n * _geometric_sum(p2, n)
    far = far_num / far_den if far_den else float(q1 >= q2)
    mar = mar_num / mar_den if mar_den else float(p2 >= p1)
    if p1 == 0.0:
        aad = float("inf")
    elif p2 == 0.0:
        aad = float(n - 1) * sample_period
    else:
        aad = sample_period * (1.0 - p1**n - p2 * p1**n) / (p2 * p1**n)
    return IIDAlarmPerformance(float(far), float(mar), float(aad))


def design_iid_delay_timer(
    normal_values: Iterable[float],
    abnormal_values: Iterable[float],
    *,
    thresholds: Sequence[float],
    delays: Sequence[int],
    direction: Direction = "high",
    targets: tuple[float, float, float] = (0.05, 0.05, 10.0),
    weights: tuple[float, float, float] = (1.0, 1.0, 0.25),
    sample_period: float = 1.0,
) -> IIDDelayDesignResult:
    """Book Section 2.1 threshold-delay grid search using empirical PDFs."""

    normal = np.asarray(list(normal_values), dtype=float)
    abnormal = np.asarray(list(abnormal_values), dtype=float)
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if (
        normal.ndim != 1
        or abnormal.ndim != 1
        or not len(normal)
        or not len(abnormal)
        or target.shape != (3,)
        or weight.shape != (3,)
        or np.any(target <= 0)
        or np.any(weight < 0)
        or sample_period <= 0
    ):
        raise ValueError("valid data, three positive targets, and non-negative weights are required")
    candidates: list[IIDDelayDesignResult] = []
    for threshold in thresholds:
        q1 = float(np.mean(threshold_states(normal, float(threshold), direction)))
        p1 = float(np.mean(threshold_states(abnormal, float(threshold), direction)))
        for delay in delays:
            performance = iid_delay_timer_performance(
                q1, p1, int(delay), sample_period=sample_period
            )
            observed = np.asarray(
                (
                    performance.false_alarm_rate,
                    performance.missed_alarm_rate,
                    performance.average_alarm_delay,
                ),
                dtype=float,
            )
            if not np.isfinite(observed).all():
                continue
            loss = float(np.sum(weight * observed / target))
            candidates.append(
                IIDDelayDesignResult(
                    float(threshold),
                    int(delay),
                    performance,
                    q1,
                    p1,
                    loss,
                )
            )
    if not candidates:
        raise ValueError("threshold-delay grid produced no finite candidate")
    return min(
        candidates,
        key=lambda result: (
            result.loss,
            result.performance.missed_alarm_rate,
            result.performance.false_alarm_rate,
            result.performance.average_alarm_delay,
        ),
    )


@dataclass(frozen=True)
class PettittResult:
    change_index: int
    statistic: float
    p_value: float


def pettitt_test(values: Iterable[float]) -> PettittResult:
    """Pettitt's rank-based single change-point test (Book Sec. 2.1.3)."""

    samples = np.asarray(list(values), dtype=float)
    if samples.ndim != 1 or len(samples) < 3:
        raise ValueError("values must contain at least three one-dimensional samples")
    ranks = rankdata(samples, method="average")
    sizes = np.arange(1, len(samples), dtype=float)
    u = 2.0 * np.cumsum(ranks)[:-1] - sizes * (len(samples) + 1.0)
    location = int(np.argmax(np.abs(u)))
    statistic = float(abs(u[location]))
    n = float(len(samples))
    p_value = min(1.0, float(2.0 * np.exp(-6.0 * statistic**2 / (n**3 + n**2))))
    return PettittResult(location + 1, statistic, p_value)


def recursive_pettitt_segments(
    values: Iterable[float], *, alpha: float = 0.05, min_size: int = 20
) -> list[tuple[int, int]]:
    """Recursively segment a series at significant Pettitt change points."""

    samples = np.asarray(list(values), dtype=float)
    if not 0.0 < alpha < 1.0 or min_size < 2:
        raise ValueError("alpha must be in (0, 1) and min_size at least two")
    segments: list[tuple[int, int]] = []

    def split(start: int, stop: int) -> None:
        if stop - start < 2 * min_size:
            segments.append((start, stop))
            return
        result = pettitt_test(samples[start:stop])
        change = start + result.change_index
        if result.p_value < alpha and change - start >= min_size and stop - change >= min_size:
            split(start, change)
            split(change, stop)
        else:
            segments.append((start, stop))

    if len(samples):
        split(0, len(samples))
    return sorted(segments)


def binary_run_lengths(states: Iterable[int | bool], target: int = 1) -> np.ndarray:
    array = np.asarray(list(states), dtype=int)
    if array.ndim != 1:
        raise ValueError("states must be one-dimensional")
    selected = array == int(target)
    changes = np.diff(np.r_[False, selected, False].astype(np.int8))
    return (np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).astype(int)


@dataclass(frozen=True)
class BetaPosterior:
    successes: int
    trials: int
    mean: float
    lower: float
    upper: float
    credibility: float


def beta_binomial_posterior(
    successes: int, trials: int, *, confidence: float = 0.95
) -> BetaPosterior:
    """Uniform-prior binomial posterior and narrowest credible interval."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    a, b = successes + 1.0, trials - successes + 1.0
    mean, alpha = a / (a + b), 1.0 - confidence

    def width(lower_mass: float) -> float:
        return float(
            beta_distribution.ppf(lower_mass + confidence, a, b)
            - beta_distribution.ppf(lower_mass, a, b)
        )

    lower_mass = float(minimize_scalar(width, bounds=(0.0, alpha), method="bounded").x)
    lower = float(beta_distribution.ppf(lower_mass, a, b))
    upper = float(beta_distribution.ppf(lower_mass + confidence, a, b))
    uncertainty = max(mean - lower, upper - mean)
    credibility = float(mean / uncertainty) if uncertainty > 0 else float("inf")
    return BetaPosterior(successes, trials, float(mean), lower, upper, credibility)


def bayesian_duration_tail(
    durations: Iterable[int], minimum_duration: int, *, confidence: float = 0.95
) -> BetaPosterior:
    samples = np.asarray(list(durations), dtype=int)
    if samples.ndim != 1 or not len(samples) or np.any(samples < 1):
        raise ValueError("durations must be a non-empty vector of positive integers")
    if minimum_duration < 1:
        raise ValueError("minimum_duration must be positive")
    return beta_binomial_posterior(
        int(np.sum(samples >= minimum_duration)), len(samples), confidence=confidence
    )


def threshold_states(
    values: Iterable[float], threshold: float, direction: Direction = "high"
) -> np.ndarray:
    samples = np.asarray(list(values), dtype=float)
    if direction == "high":
        return samples >= threshold
    if direction == "low":
        return samples <= threshold
    raise ValueError("direction must be 'high' or 'low'")


@dataclass(frozen=True)
class NonIIDDelayDesignResult:
    threshold: float
    delay: int
    false_alarm: BetaPosterior
    missed_alarm: BetaPosterior
    normal_alarm_runs: int
    abnormal_no_alarm_runs: int
    zero_event_fallback: bool
    loss: float


def design_non_iid_delay_timer(
    normal_values: Iterable[float],
    abnormal_values: Iterable[float],
    *,
    thresholds: Sequence[float],
    delays: Sequence[int],
    direction: Direction = "high",
    target_far: float = 0.05,
    target_mar: float = 0.05,
    far_weight: float = 0.5,
    confidence: float = 0.95,
) -> NonIIDDelayDesignResult:
    """Book Sec. 2.2 threshold-delay search using duration-tail posteriors."""

    normal = np.asarray(list(normal_values), dtype=float)
    abnormal = np.asarray(list(abnormal_values), dtype=float)
    if not len(normal) or not len(abnormal) or not 0.0 <= far_weight <= 1.0:
        raise ValueError("non-empty data and far_weight in [0, 1] are required")
    candidates: list[NonIIDDelayDesignResult] = []
    for threshold in thresholds:
        high_runs = binary_run_lengths(threshold_states(normal, threshold, direction), 1)
        low_runs = binary_run_lengths(threshold_states(abnormal, threshold, direction), 0)
        for delay in delays:
            far = (
                bayesian_duration_tail(high_runs, int(delay), confidence=confidence)
                if len(high_runs)
                else beta_binomial_posterior(0, len(normal), confidence=confidence)
            )
            mar = (
                bayesian_duration_tail(low_runs, int(delay), confidence=confidence)
                if len(low_runs)
                else beta_binomial_posterior(0, len(abnormal), confidence=confidence)
            )
            loss = far_weight * abs(far.mean - target_far) + (1.0 - far_weight) * abs(mar.mean - target_mar)
            candidates.append(
                NonIIDDelayDesignResult(
                    float(threshold),
                    int(delay),
                    far,
                    mar,
                    int(len(high_runs)),
                    int(len(low_runs)),
                    bool(not len(high_runs) or not len(low_runs)),
                    float(loss),
                )
            )
    if not candidates:
        raise ValueError("candidate thresholds produced no finite duration design")
    return min(candidates, key=lambda x: (x.loss, -min(x.false_alarm.credibility, x.missed_alarm.credibility), x.delay))


@dataclass(frozen=True)
class AlarmEpisodeMetrics:
    durations: np.ndarray
    deviations: np.ndarray


def alarm_episode_metrics(
    values: Iterable[float], threshold: float, direction: Direction = "high"
) -> AlarmEpisodeMetrics:
    samples = np.asarray(list(values), dtype=float)
    active = threshold_states(samples, threshold, direction)
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    deviations = [
        float(np.max(samples[start:stop] - threshold))
        if direction == "high"
        else float(np.max(threshold - samples[start:stop]))
        for start, stop in zip(starts, stops, strict=True)
    ]
    return AlarmEpisodeMetrics((stops - starts).astype(int), np.asarray(deviations))


@dataclass(frozen=True)
class DeadbandIndexResult:
    angle_degrees: float
    slope: float
    lower_degrees: float
    upper_degrees: float
    duration_max: float
    deviation_max: float
    suitable: bool


def _book_percentile(samples: np.ndarray, probability: float = 0.95) -> float:
    ordered = np.sort(samples)
    index = max(0, min(len(ordered) - 1, int(np.ceil(probability * len(ordered))) - 1))
    return float(ordered[index])


def deadband_index(
    values: Iterable[float],
    threshold: float,
    *,
    direction: Direction = "high",
    duration_max: float | None = None,
    deviation_max: float | None = None,
) -> DeadbandIndexResult:
    """Compute Book equations (2.53)-(2.65), including the 45-degree test."""

    samples = np.asarray(list(values), dtype=float)
    metrics = alarm_episode_metrics(samples, threshold, direction)
    if len(metrics.durations) < 2:
        raise ValueError("at least two alarm episodes are required")
    if duration_max is None or deviation_max is None:
        reference = alarm_episode_metrics(samples, float(np.mean(samples)), direction)
        if not len(reference.durations) or not np.any(reference.deviations > 0):
            raise ValueError("reference episodes cannot define normalization limits")
        duration_max = duration_max or _book_percentile(reference.durations.astype(float))
        deviation_max = deviation_max or _book_percentile(reference.deviations)
    if duration_max <= 0 or deviation_max <= 0:
        raise ValueError("normalization limits must be positive")
    duration_n = metrics.durations / float(duration_max)
    deviation_n = metrics.deviations / float(deviation_max)
    denominator = float(np.sum(deviation_n**2))
    if denominator == 0:
        slope, standard_error = float("inf"), 0.0
    else:
        slope = float(np.sum(deviation_n * duration_n) / denominator)
        residual = duration_n - slope * deviation_n
        standard_error = float(np.sqrt(np.sum(residual**2) / max(len(residual) - 1, 1) / denominator))
    angle = float(np.degrees(np.arctan(slope)))
    angle_error = float(np.degrees(np.arctan(standard_error)))
    return DeadbandIndexResult(
        angle,
        slope,
        max(0.0, angle - 1.96 * angle_error),
        min(90.0, angle + 1.96 * angle_error),
        float(duration_max),
        float(deviation_max),
        bool(angle > 45.0 + 1.96 * angle_error),
    )


@dataclass(frozen=True)
class DeadbandDesignResult:
    width: float
    remaining_probability: BetaPosterior
    target_remaining_probability: float
    loss: float


def design_deadband_width(
    deviations: Iterable[float],
    *,
    maximum_width: float,
    target_remaining_probability: float = 0.05,
    candidates: Sequence[float] | None = None,
    confidence: float = 0.95,
) -> DeadbandDesignResult:
    """Solve Book equations (2.68)-(2.80) with Beta posteriors."""

    samples = np.asarray(list(deviations), dtype=float)
    if samples.ndim != 1 or not len(samples) or np.any(samples < 0) or maximum_width < 0:
        raise ValueError("valid non-negative deviations and maximum_width are required")
    _probability(target_remaining_probability, "target_remaining_probability")
    grid = np.unique(np.r_[0.0, samples, maximum_width]) if candidates is None else np.asarray(candidates, dtype=float)
    grid = grid[(grid >= 0.0) & (grid <= maximum_width)]
    if not len(grid):
        raise ValueError("no deadband candidate lies inside [0, maximum_width]")
    results = []
    for width in grid:
        posterior = beta_binomial_posterior(int(np.sum(samples >= width)), len(samples), confidence=confidence)
        results.append(DeadbandDesignResult(float(width), posterior, float(target_remaining_probability), abs(posterior.mean - target_remaining_probability)))
    return min(results, key=lambda item: (item.loss, item.width))


@dataclass(frozen=True)
class AlarmProbabilityPlot:
    threshold: float
    boundaries: np.ndarray
    states: np.ndarray
    transition_matrix: np.ndarray
    first_alarm_state: int
    p_to_alarm: dict[int, float]
    t_to_alarm: dict[int, float]
    p_after_alarm: dict[int, float]
    t_after_alarm: dict[int, float]


def quantize_alarm_probability_states(
    values: Iterable[float], threshold: float, *, minimum_state_samples: int = 600
) -> tuple[np.ndarray, np.ndarray, int]:
    """Book Algorithms 1-2 equal-count state construction."""

    samples = np.asarray(list(values), dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or minimum_state_samples < 1:
        raise ValueError("valid samples and minimum_state_samples are required")
    lower, upper = np.sort(samples[samples < threshold]), np.sort(samples[samples >= threshold])
    if not len(lower) or not len(upper):
        raise ValueError("threshold must divide the observed samples")
    n_lower = max(1, len(lower) // minimum_state_samples)
    n_upper = max(1, len(upper) // minimum_state_samples)
    lower_cuts = [float(lower[min(k * minimum_state_samples, len(lower) - 1)]) for k in range(1, n_lower)]
    upper_cuts = [float(upper[min(k * minimum_state_samples, len(upper) - 1)]) for k in range(1, n_upper)]
    cuts = sorted({cut for cut in lower_cuts if cut < threshold})
    first_alarm_state = len(cuts) + 1
    cuts.append(float(threshold))
    cuts.extend(sorted({cut for cut in upper_cuts if cut > threshold}))
    boundaries = np.asarray([-np.inf, *cuts, np.inf], dtype=float)
    states = np.searchsorted(boundaries[1:-1], samples, side="right") + 1
    return boundaries, states.astype(int), first_alarm_state


def estimate_transition_matrix(states: Iterable[int]) -> np.ndarray:
    sequence = np.asarray(list(states), dtype=int)
    if sequence.ndim != 1 or len(sequence) < 2 or np.min(sequence) < 1:
        raise ValueError("states must contain at least two positive integers")
    matrix = np.zeros((int(np.max(sequence)), int(np.max(sequence))), dtype=float)
    for source, destination in zip(sequence[:-1], sequence[1:], strict=True):
        matrix[source - 1, destination - 1] += 1.0
    for index in range(len(matrix)):
        total = float(np.sum(matrix[index]))
        if total:
            matrix[index] /= total
        else:
            matrix[index, index] = 1.0
    return matrix


def _alarm_side_hitting_statistics(
    transition: np.ndarray, first_alarm_state: int, target_state: int
) -> tuple[float, float]:
    a, m = first_alarm_state - 1, target_state - 1
    transient = np.arange(a, m)
    q = transition[np.ix_(transient, transient)]
    system = np.eye(len(transient)) - q
    success = transition[np.ix_(transient, np.arange(m, len(transition)))].sum(axis=1)
    probabilities = np.linalg.lstsq(system, success, rcond=None)[0]
    times = np.linalg.lstsq(system, np.ones(len(transient)), rcond=None)[0]
    return float(np.clip(probabilities[0], 0.0, 1.0)), float(max(times[0], 0.0))


def build_alarm_probability_plot(
    values: Iterable[float], threshold: float, *, minimum_state_samples: int = 600
) -> AlarmProbabilityPlot:
    """Estimate the four statistics in Book equations (2.87)-(2.100)."""

    boundaries, states, a = quantize_alarm_probability_states(values, threshold, minimum_state_samples=minimum_state_samples)
    transition = estimate_transition_matrix(states)
    p_to: dict[int, float] = {}
    t_to: dict[int, float] = {}
    for k in range(1, a):
        probability, time = 1.0, 0.0
        for state in range(k, a):
            escape = 1.0 - transition[state - 1, state - 1]
            if escape <= 0:
                probability, time = 0.0, float("inf")
                break
            probability *= transition[state - 1, state] / escape
            time += 1.0 / escape
        p_to[k], t_to[k] = float(probability), float(time)
    p_after, t_after = {}, {}
    for target in range(a + 1, len(transition) + 1):
        p_after[target], t_after[target] = _alarm_side_hitting_statistics(transition, a, target)
    return AlarmProbabilityPlot(float(threshold), boundaries, states, transition, a, p_to, t_to, p_after, t_after)


@dataclass(frozen=True)
class AlarmProbabilityThresholdResult:
    threshold: float
    score: float
    plot: AlarmProbabilityPlot


def select_alarm_probability_threshold(
    values: Iterable[float],
    thresholds: Sequence[float],
    *,
    minimum_state_samples: int = 600,
    probability_weight: float = 0.5,
) -> AlarmProbabilityThresholdResult:
    """Select the APP threshold using Book equations (2.101)-(2.104)."""

    if not 0.0 <= probability_weight <= 1.0:
        raise ValueError("probability_weight must be in [0, 1]")
    samples = np.asarray(list(values), dtype=float)
    plots = [build_alarm_probability_plot(samples, float(x), minimum_state_samples=minimum_state_samples) for x in thresholds]
    all_times = [v for p in plots for v in [*p.t_to_alarm.values(), *p.t_after_alarm.values()] if np.isfinite(v)]
    if not all_times:
        raise ValueError("no finite APP transition times are available")
    maximum_time = max(all_times)
    results = []
    for plot in plots:
        pairs = [*zip(plot.p_to_alarm.values(), plot.t_to_alarm.values(), strict=True), *zip(plot.p_after_alarm.values(), plot.t_after_alarm.values(), strict=True)]
        scores = [probability_weight * p + (1.0 - probability_weight) * t / maximum_time for p, t in pairs if np.isfinite(t)]
        results.append(AlarmProbabilityThresholdResult(plot.threshold, float(np.mean(scores)) if scores else float("-inf"), plot))
    return max(results, key=lambda item: (item.score, -item.threshold))
