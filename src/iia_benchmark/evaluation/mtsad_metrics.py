from __future__ import annotations

import numpy as np


def pointwise_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Binary point-wise precision/recall/F1 with no time tolerance."""

    truth = np.asarray(truth, dtype=bool).reshape(-1)
    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have equal length")
    tp = float(np.sum(truth & prediction))
    fp = float(np.sum(~truth & prediction))
    fn = float(np.sum(truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def point_adjust(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Apply the reference point-adjustment used by the benchmark lineage.

    Transcribed from the Time-Series-Library / Anomaly-Transformer helper
    ``utils.tools.adjustment``: once any point of a ground-truth anomaly
    segment is detected, the whole segment is credited as detected.  The
    reference implementation loops point by point; the same semantics are
    expressed here with cumulative sums so that the eight-million-point
    concatenated layouts stay tractable.
    """

    truth = np.asarray(truth, dtype=int).reshape(-1)
    adjusted = np.asarray(prediction, dtype=int).reshape(-1).copy()
    if truth.shape != adjusted.shape:
        raise ValueError("truth and prediction must have equal length")
    ranges = anomaly_ranges(truth)
    if not ranges:
        return adjusted.astype(bool)
    starts = np.array([start for start, _ in ranges], dtype=np.int64)
    ends = np.array([end for _, end in ranges], dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(adjusted)])
    detected = (cumulative[ends + 1] - cumulative[starts]) > 0
    for start, end in zip(starts[detected], ends[detected]):
        adjusted[start:end + 1] = 1
    return adjusted.astype(bool)


def adjusted_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Point-wise metrics after point adjustment (the legacy reported protocol)."""

    return pointwise_metrics(truth, point_adjust(truth, prediction))


def anomaly_ranges(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous inclusive [start, end] ranges of ones.

    Vectorised on purpose: the harness evaluates eight-million-point
    concatenated layouts, where a per-element Python loop costs minutes per
    call and the function is called several times per protocol.
    """

    flags = np.asarray(labels, dtype=bool).reshape(-1)
    if not flags.any():
        return []
    padded = np.concatenate((np.zeros(1, dtype=np.int8), flags.astype(np.int8), np.zeros(1, dtype=np.int8)))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _bias(overlap: float, real_len: int, position: float, bias: str) -> float:
    if real_len <= 0:
        return 0.0
    if bias == "flat":
        return overlap
    if bias == "front":
        return overlap * (1.0 - position)
    if bias == "back":
        return overlap * position
    if bias == "middle":
        return overlap * (1.0 - abs(2.0 * position - 1.0))
    if bias == "reciprocal":
        return overlap * (1.0 / (1.0 + position))
    raise ValueError(f"unknown positional bias: {bias}")


def range_based_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    alpha: float = 0.0,
    bias: str = "flat",
) -> dict[str, float]:
    """Range-based (affiliation-style) precision/recall/F1.

    Follows the range-based anomaly-detection metrics of Tatbul et al. (2018):
    recall rewards covering a real anomaly range, precision rewards predicted
    ranges that fall inside real ranges, and ``alpha`` mixes in an existence
    reward. ``alpha=0`` reproduces the commonly reported "range-based F1".
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    real = anomaly_ranges(truth)
    predicted = anomaly_ranges(prediction)
    if not real and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "real_ranges": 0.0, "predicted_ranges": 0.0}
    if not real or not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "real_ranges": float(len(real)), "predicted_ranges": float(len(predicted))}

    # Both loops below are sweeps: the cursor only moves forward, so the work is
    # proportional to the number of interval pairs that actually overlap.  A
    # naive double loop is quadratic and stalls for minutes once a threshold
    # produces hundreds of thousands of scattered predicted ranges.
    precisions: list[float] = []
    cursor = 0
    for pred_start, pred_end in predicted:
        while cursor < len(real) and real[cursor][1] < pred_start:
            cursor += 1
        scan = cursor
        best = 0.0
        pred_len = pred_end - pred_start + 1
        while scan < len(real) and real[scan][0] <= pred_end:
            overlap_start = max(pred_start, real[scan][0])
            overlap_end = min(pred_end, real[scan][1])
            if overlap_end >= overlap_start:
                best = max(best, (overlap_end - overlap_start + 1) / pred_len)
            scan += 1
        precisions.append(best)

    recalls: list[float] = []
    cursor = 0
    for real_start, real_end in real:
        while cursor < len(predicted) and predicted[cursor][1] < real_start:
            cursor += 1
        scan = cursor
        real_len = real_end - real_start + 1
        best_existence = 0.0
        best_overlap = 0.0
        while scan < len(predicted) and predicted[scan][0] <= real_end:
            overlap_start = max(real_start, predicted[scan][0])
            overlap_end = min(real_end, predicted[scan][1])
            if overlap_end >= overlap_start:
                best_existence = 1.0
                position = (overlap_start - real_start) / real_len
                overlap = (overlap_end - overlap_start + 1) / real_len
                best_overlap = max(
                    best_overlap, _bias(overlap, real_len, position, bias)
                )
            scan += 1
        recalls.append(alpha * best_existence + (1.0 - alpha) * best_overlap)

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "real_ranges": float(len(real)),
        "predicted_ranges": float(len(predicted)),
    }


def best_f1_threshold(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Threshold that maximises point-wise F1 on the scored series (oracle search)."""

    truth = np.asarray(truth, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape:
        raise ValueError("truth and scores must have equal length")
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("no finite scores to threshold")
    positive = scores[finite]
    labels = truth[finite]
    order = np.argsort(-positive, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.concatenate([[0], np.cumsum(sorted_labels)])
    total_true = int(sorted_labels.sum())
    # Ascending copy of -scores lets us count, for every candidate threshold,
    # how many points are strictly above it.
    negated = np.sort(-positive)
    thresholds = np.unique(positive)
    predicted_counts = np.searchsorted(negated, -thresholds, side="left")
    true_positive = cumulative_true[predicted_counts]
    predicted_total = predicted_counts
    false_positive = predicted_total - true_positive
    false_negative = total_true - true_positive
    denominator = (2 * true_positive + false_positive + false_negative).astype(np.float64)
    f1 = np.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    best_index = int(np.argmax(f1))
    tp = float(true_positive[best_index])
    fp = float(false_positive[best_index])
    fn = float(false_negative[best_index])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": float(f1[best_index]),
        "precision": precision,
        "recall": recall,
        "threshold": float(thresholds[best_index]),
    }


def percentile_threshold(
    train_scores: np.ndarray, test_scores: np.ndarray, anomaly_ratio: float
) -> float:
    """Threshold at the ``100 - anomaly_ratio`` percentile of train+test scores."""

    if not 0.0 < anomaly_ratio <= 100.0:
        raise ValueError("anomaly_ratio must lie in (0, 100]")
    combined = np.concatenate(
        [
            np.asarray(train_scores, dtype=np.float64).reshape(-1),
            np.asarray(test_scores, dtype=np.float64).reshape(-1),
        ]
    )
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        raise ValueError("no finite scores available for the percentile threshold")
    return float(np.percentile(combined, 100.0 - anomaly_ratio))


def evaluate_scores(
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    point_adjustment: bool = True,
) -> dict[str, float]:
    """Evaluate a scored test series under one explicit protocol choice.

    Timestamps that the detector could not score (the uncovered prefix of a
    windowed series, marked NaN) are excluded from every metric instead of
    being silently counted as missed detections.
    """

    truth = np.asarray(truth, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != scores.shape:
        raise ValueError("truth and scores must have equal length")
    scored = np.isfinite(scores)
    if not scored.any():
        raise ValueError("no scored timestamps to evaluate")
    truth_scored = truth[scored]
    scores_scored = scores[scored]
    prediction = scores_scored > threshold
    point = pointwise_metrics(truth_scored, prediction)
    adjusted = adjusted_metrics(truth_scored, prediction)
    ranges = range_based_metrics(truth_scored, prediction, alpha=0.0, bias="flat")
    result = {
        "threshold": float(threshold),
        "scored_points": int(scored.sum()),
        "unscored_points": int((~scored).sum()),
        "pointwise_precision": point["precision"],
        "pointwise_recall": point["recall"],
        "pointwise_f1": point["f1"],
        "adjusted_precision": adjusted["precision"],
        "adjusted_recall": adjusted["recall"],
        "adjusted_f1": adjusted["f1"],
        "range_precision": ranges["precision"],
        "range_recall": ranges["recall"],
        "range_f1": ranges["f1"],
    }
    result["reported_protocol_f1"] = (
        result["adjusted_f1"] if point_adjustment else result["pointwise_f1"]
    )
    return result
