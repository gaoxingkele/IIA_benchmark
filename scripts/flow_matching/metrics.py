"""Shared transfer metrics: median MAE, mean RMSE and exact empirical CRPS."""
from __future__ import annotations

import numpy as np

PROTOCOL = 'transfer_ensemble_v1'


def apply_time_cutoffs(mask, cutoffs):
    """Exclude repeated timestamps in [case, time, feature] evaluation masks."""
    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 3 or len(cutoffs) != len(result):
        raise ValueError('Invalid time cutoff dimensions')
    for index, cutoff in enumerate(cutoffs):
        count = int(cutoff)
        if count < 0 or count > result.shape[1]:
            raise ValueError('Invalid time cutoff length')
        result[index, :count, :] = False
    return result


def ensemble_totals(target, samples, eval_mask):
    target, samples, mask = np.asarray(target), np.asarray(samples), np.asarray(eval_mask, dtype=bool)
    if samples.ndim != target.ndim + 1 or samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:] or mask.shape != target.shape:
        raise ValueError('Expected samples [case, sample, ...] matching target and evaluation mask')
    count = int(mask.sum())
    if not count or not samples.shape[1]:
        raise ValueError('No evaluated observations or ensemble members')
    # Select targets first: native missing/non-evaluated values cannot contaminate scores.
    predictions = np.moveaxis(samples, 1, -1)[mask].astype(np.float64)
    truth = target[mask].astype(np.float64)
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError('Non-finite evaluated predictions or truth')
    size = predictions.shape[-1]
    ordered = np.sort(predictions, axis=-1)
    weights = 2 * np.arange(1, size + 1) - size - 1
    crps = np.abs(predictions - truth[:, None]).mean(-1) - (ordered * weights).sum(-1) / size ** 2
    lower, upper = np.quantile(predictions, [.05, .95], axis=-1)
    return {'eval_count': count,
            'absolute_error': float(np.abs(np.median(predictions, axis=-1) - truth).sum()),
            'squared_error': float(np.square(predictions.mean(-1) - truth).sum()),
            'crps_sum': float(crps.sum()), 'magnitude_sum': float(np.abs(truth).sum()),
            'covered': int(((truth >= lower) & (truth <= upper)).sum()),
            'interval_width_sum': float((upper - lower).sum())}


def finalize(totals):
    count = totals['eval_count']
    if count <= 0:
        raise ValueError('No evaluated targets')
    return {'mae': totals['absolute_error'] / count,
            'rmse': (totals['squared_error'] / count) ** .5,
            'crps': totals['crps_sum'] / count,
            'normalized_crps': totals['crps_sum'] / totals['magnitude_sum'] if totals['magnitude_sum'] > 0 else None,
            'coverage_90': totals['covered'] / count,
            'interval_width_90': totals['interval_width_sum'] / count}
