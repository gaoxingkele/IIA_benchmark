"""Validation-only thresholding and complete-timestamp TSAD diagnostics.

This is the strict deployment track, not a replacement for author or TAB
protocols. No test scores enter calibration and no point adjustment is applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping

import numpy as np

from .mtsad_metrics import pointwise_metrics


def _scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Require a nonempty finite score for every timestamp")
    return values


@dataclass(frozen=True)
class FrozenThreshold:
    threshold: float
    false_alarm_percent: float
    calibration_split: str
    calibration_sha256: str
    calibration_points: int
    calibration_dataset_sha256: str
    model_artifact_sha256: str
    rule: str = "validation_score_percentile_linear_strict_greater"

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_validation(
    validation_scores: np.ndarray, *, false_alarm_percent: float,
    calibration_split: str, calibration_dataset_sha256: str,
    model_artifact_sha256: str,
) -> FrozenThreshold:
    """Freeze an alarm-budget quantile from a predeclared validation split.

    This does not certify that the validation split is clean: callers must
    supply its audited manifest and keep anomaly-contamination metadata.
    """
    if calibration_split != "validation":
        raise ValueError("Strict calibration requires the validation split")
    if not 0 < false_alarm_percent < 100:
        raise ValueError("false_alarm_percent must lie between 0 and 100")
    for digest in (calibration_dataset_sha256, model_artifact_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Require lowercase SHA256 provenance")
    values = _scores(validation_scores)
    digest = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
    return FrozenThreshold(
        float(np.percentile(values, 100 - false_alarm_percent, method="linear")),
        float(false_alarm_percent), calibration_split, digest, int(values.size),
        calibration_dataset_sha256, model_artifact_sha256,
    )


def evaluate_entities(
    entities: Mapping[str, tuple[np.ndarray, np.ndarray]], *,
    expected_lengths: Mapping[str, int], calibration: FrozenThreshold,
) -> dict:
    """Require one aligned, ordered score per original timestamp per entity.

    The caller must audit timestamp IDs and order before passing arrays. Exact
    lengths reject dropped window prefixes/tails; finite checks reject holes.
    Entity identity is retained for macro results and later grouped bootstrap.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not entities or set(entities) != set(expected_lengths):
        raise ValueError("Entity IDs must match the frozen data manifest")
    if calibration.calibration_split != "validation" or not np.isfinite(calibration.threshold):
        raise ValueError("Invalid strict calibration artifact")

    def metrics(truth: np.ndarray, scores: np.ndarray) -> dict:
        result = pointwise_metrics(truth, scores > calibration.threshold)
        result["auroc"] = float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2 else None
        result["average_precision"] = float(average_precision_score(truth, scores)) if truth.any() else None
        return result

    per_entity, labels, scores_all = {}, [], []
    for entity_id, (truth, raw_scores) in entities.items():
        scores = _scores(raw_scores)
        truth = np.asarray(truth)
        if truth.ndim != 1 or truth.size != scores.size or scores.size != expected_lengths[entity_id]:
            raise ValueError("Timestamp coverage differs from the frozen data manifest")
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("Ground truth must be binary")
        truth = truth.astype(bool)
        per_entity[entity_id] = metrics(truth, scores)
        labels.append(truth)
        scores_all.append(scores)
    return {
        "protocol": "strict_validation_only_no_pa_v1",
        "calibration": calibration.to_dict(),
        "coverage": "complete",
        "point_adjustment": False,
        "per_entity": per_entity,
        "micro": metrics(np.concatenate(labels), np.concatenate(scores_all)),
        "macro_f1": float(np.mean([item["f1"] for item in per_entity.values()])),
        "boundary": "Aligned timestamp order is caller-audited; no VUS/Affiliation or uncertainty is computed here. This diagnostic alone is not a leaderboard result.",
    }
