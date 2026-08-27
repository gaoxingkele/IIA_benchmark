"""Method-agnostic AFC-RobustBench-style robustness evaluation protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from iia_benchmark.data.schema import AlarmEpisode
from iia_benchmark.models.flood import perturb_alarm_episode


@dataclass(frozen=True)
class PerturbationScenario:
    kind: str
    severity: float

    def __post_init__(self) -> None:
        if self.kind not in {"missing", "spurious", "timing", "detector_delay", "mixed"}:
            raise ValueError(f"unsupported perturbation kind: {self.kind}")
        if not 0 <= self.severity <= 1:
            raise ValueError("severity must be in [0, 1]")


@dataclass(frozen=True)
class RobustnessPoint:
    perturbation: str
    severity: float
    observation_progress: float
    clean_score: float
    mean_score: float
    standard_deviation: float
    confidence_low: float
    confidence_high: float
    degradation: float
    draws: int


@dataclass(frozen=True)
class AFCRobustnessReport:
    clean_scores: dict[float, float]
    points: tuple[RobustnessPoint, ...]
    normalized_robustness_auc: dict[str, float]
    seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "clean_scores": {str(key): value for key, value in self.clean_scores.items()},
            "points": [asdict(point) for point in self.points],
            "normalized_robustness_auc": self.normalized_robustness_auc,
            "seeds": list(self.seeds),
        }


def default_perturbation_grid(
    severities: Sequence[float] = (0.05, 0.1, 0.2),
) -> tuple[PerturbationScenario, ...]:
    return tuple(
        PerturbationScenario(kind, float(severity))
        for kind in ("missing", "spurious", "timing", "detector_delay", "mixed")
        for severity in severities
    )


def _episode_scale(episode: AlarmEpisode) -> tuple[int, float, float]:
    timestamps = np.asarray([event.timestamp for event in episode.events], dtype=float)
    if timestamps.size < 2:
        return len(episode.events), 1.0, 1.0
    ordered = np.sort(timestamps)
    positive_gaps = np.diff(ordered)
    positive_gaps = positive_gaps[positive_gaps > 0]
    median_gap = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
    return len(episode.events), median_gap, max(float(ordered[-1] - ordered[0]), median_gap)


def apply_robustness_scenario(
    episode: AlarmEpisode,
    scenario: PerturbationScenario,
    *,
    seed: int,
    spurious_tags: Sequence[str] | None = None,
) -> AlarmEpisode:
    """Map normalized severity to one isolated or mixed event corruption."""

    count, median_gap, duration = _episode_scale(episode)
    severity = scenario.severity
    parameters: dict[str, float | int | Sequence[str] | None] = {
        "missing_probability": 0.0,
        "spurious_count": 0,
        "timing_jitter": 0.0,
        "detector_delay": 0.0,
        "spurious_tags": spurious_tags,
    }
    if scenario.kind in {"missing", "mixed"}:
        parameters["missing_probability"] = severity
    if scenario.kind in {"spurious", "mixed"}:
        parameters["spurious_count"] = int(np.ceil(severity * max(1, count))) if severity else 0
    if scenario.kind in {"timing", "mixed"}:
        parameters["timing_jitter"] = severity * median_gap
    if scenario.kind in {"detector_delay", "mixed"}:
        parameters["detector_delay"] = severity * duration
    return perturb_alarm_episode(episode, seed=seed, **parameters)


def _prefix_on_clean_clock(
    observed: AlarmEpisode, clean: AlarmEpisode, progress: float
) -> AlarmEpisode:
    if not 0 < progress <= 1:
        raise ValueError("observation progress must be in (0, 1]")
    if not clean.events:
        return observed
    start = min(event.timestamp for event in clean.events)
    stop = max(event.timestamp for event in clean.events)
    cutoff = start + progress * (stop - start)
    return AlarmEpisode(
        episode_id=f"{observed.episode_id}_prefix_{progress:.3f}",
        events=tuple(event for event in observed.events if event.timestamp <= cutoff),
        label=observed.label,
        root_cause=observed.root_cause,
    )


def _accuracy(truth: Sequence[str], prediction: Sequence[str]) -> float:
    if len(truth) != len(prediction):
        raise ValueError("predictor returned an incompatible number of labels")
    return float(np.mean(np.asarray(truth, dtype=object) == np.asarray(prediction, dtype=object)))


def run_afc_robustness_benchmark(
    episodes: Sequence[AlarmEpisode],
    predictor: Callable[[Sequence[AlarmEpisode]], Sequence[str]],
    *,
    scenarios: Sequence[PerturbationScenario] | None = None,
    observation_progress: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    seeds: Sequence[int] = tuple(range(10)),
    score: Callable[[Sequence[str], Sequence[str]], float] = _accuracy,
    spurious_tags: Sequence[str] | None = None,
) -> AFCRobustnessReport:
    """Evaluate severity/progress profiles with Monte-Carlo uncertainty.

    Perturbations are applied to test episodes only.  Training data and model
    state belong to ``predictor`` and remain untouched, preventing leakage.
    """

    samples = tuple(episodes)
    if not samples or any(episode.label is None for episode in samples):
        raise ValueError("episodes must be nonempty and carry labels")
    progress_values = tuple(sorted(set(float(item) for item in observation_progress)))
    if not progress_values or any(not 0 < item <= 1 for item in progress_values):
        raise ValueError("observation_progress values must be in (0, 1]")
    seed_values = tuple(int(item) for item in seeds)
    if not seed_values:
        raise ValueError("at least one Monte-Carlo seed is required")
    scenario_values = tuple(scenarios or default_perturbation_grid())
    truth = tuple(str(episode.label) for episode in samples)

    clean_scores: dict[float, float] = {}
    for progress in progress_values:
        clean_prefixes = tuple(_prefix_on_clean_clock(item, item, progress) for item in samples)
        clean_scores[progress] = float(score(truth, predictor(clean_prefixes)))

    points: list[RobustnessPoint] = []
    for scenario_index, scenario in enumerate(scenario_values):
        for progress in progress_values:
            draw_scores: list[float] = []
            for seed in seed_values:
                perturbed = tuple(
                    apply_robustness_scenario(
                        episode,
                        scenario,
                        seed=seed * 1_000_003 + scenario_index * 10_007 + episode_index,
                        spurious_tags=spurious_tags,
                    )
                    for episode_index, episode in enumerate(samples)
                )
                prefixes = tuple(
                    _prefix_on_clean_clock(observed, clean, progress)
                    for observed, clean in zip(perturbed, samples)
                )
                draw_scores.append(float(score(truth, predictor(prefixes))))
            values = np.asarray(draw_scores, dtype=float)
            mean_score = float(np.mean(values))
            points.append(
                RobustnessPoint(
                    scenario.kind,
                    scenario.severity,
                    progress,
                    clean_scores[progress],
                    mean_score,
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                    clean_scores[progress] - mean_score,
                    len(seed_values),
                )
            )

    auc: dict[str, float] = {}
    kinds = sorted({scenario.kind for scenario in scenario_values})
    for kind in kinds:
        for progress in progress_values:
            selected = sorted(
                (point for point in points if point.perturbation == kind and point.observation_progress == progress),
                key=lambda point: point.severity,
            )
            x = np.asarray([0.0] + [point.severity for point in selected])
            y = np.asarray([clean_scores[progress]] + [point.mean_score for point in selected])
            if x[-1] == 0:
                normalized = float(y[0])
            else:
                normalized = float(np.trapezoid(y, x) / x[-1])
            auc[f"{kind}@{progress:g}"] = normalized
    return AFCRobustnessReport(clean_scores, tuple(points), auc, seed_values)
