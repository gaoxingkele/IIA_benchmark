from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def transfer_entropy(source: Sequence[int], target: Sequence[int], *, lag: int = 1) -> float:
    """Estimate first-order discrete transfer entropy ``source -> target``.

    The estimator is intentionally transparent and dependency-light.  Benchmark
    reports must state the discretization policy and use permutation thresholds;
    raw scores from datasets with different sample counts are not comparable.
    """
    x = np.asarray(source, dtype=int)
    y = np.asarray(target, dtype=int)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or lag < 1:
        raise ValueError("source/target must be equal one-dimensional arrays and lag >= 1")
    if len(x) <= lag + 1:
        raise ValueError("sequences are too short for the requested lag")
    triples: Counter[tuple[int, int, int]] = Counter()
    target_past: Counter[tuple[int, int]] = Counter()
    joint_past: Counter[tuple[int, int]] = Counter()
    past_only: Counter[int] = Counter()
    for time in range(max(1, lag), len(x)):
        present = int(y[time])
        y_past = int(y[time - 1])
        x_past = int(x[time - lag])
        triples[(present, y_past, x_past)] += 1
        target_past[(present, y_past)] += 1
        joint_past[(y_past, x_past)] += 1
        past_only[y_past] += 1
    total = sum(triples.values())
    score = 0.0
    for (present, y_past, x_past), count in triples.items():
        conditional_xy = count / joint_past[(y_past, x_past)]
        conditional_y = target_past[(present, y_past)] / past_only[y_past]
        score += (count / total) * np.log2(conditional_xy / conditional_y)
    return float(max(score, 0.0))


@dataclass
class TransferEntropyRanker:
    max_lag: int = 10
    permutations: int = 99
    significance: float = 0.01
    seed: int = 0

    def rank(
        self, series: Mapping[str, Sequence[int]], *, target: str
    ) -> list[tuple[str, float, int, float]]:
        if target not in series:
            raise KeyError(f"unknown target: {target}")
        if self.max_lag < 1 or self.permutations < 0:
            raise ValueError("max_lag must be positive and permutations non-negative")
        target_values = np.asarray(series[target], dtype=int)
        rng = np.random.default_rng(self.seed)
        ranked: list[tuple[str, float, int, float]] = []
        for name, values in series.items():
            if name == target:
                continue
            source = np.asarray(values, dtype=int)
            lag_scores = [
                transfer_entropy(source, target_values, lag=lag)
                for lag in range(1, self.max_lag + 1)
            ]
            best_index = int(np.argmax(lag_scores))
            best_score = float(lag_scores[best_index])
            null_scores: list[float] = []
            for _ in range(self.permutations):
                surrogate = rng.permutation(source)
                null_scores.append(
                    max(
                        transfer_entropy(surrogate, target_values, lag=lag)
                        for lag in range(1, self.max_lag + 1)
                    )
                )
            threshold = (
                float(np.quantile(null_scores, 1 - self.significance))
                if null_scores
                else 0.0
            )
            ranked.append((name, best_score, best_index + 1, threshold))
        return sorted(ranked, key=lambda item: (item[1] > item[3], item[1]), reverse=True)

