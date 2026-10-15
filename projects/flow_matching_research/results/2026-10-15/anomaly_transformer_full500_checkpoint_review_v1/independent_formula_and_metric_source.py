"""Independent NumPy scoring/evaluation for the paper reconstruction tracks.

No imports from the training implementation, Torch or sklearn. Float64 formula
checks are distinct from the native float32 replay's exact byte comparisons.
"""
from __future__ import annotations

import hashlib
import numpy as np

FORMULA_TOLERANCE = dict(rtol=2e-6, atol=2e-6)


def array_digest(values):
    values = np.ascontiguousarray(values)
    return dict(dtype=values.dtype.str, shape=list(values.shape),
                content_sha256=hashlib.sha256(values.tobytes()).hexdigest())


def require_identical(left, right):
    if array_digest(left) != array_digest(right):
        raise ValueError('Native and saved array bytes, shape or dtype differ')


def formula_scores(reconstruction, target, priors, series, plain=False):
    """Evaluate every head/layer/window independently in double precision."""
    x = np.asarray(reconstruction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or not series or len(series) != len(priors):
        raise ValueError('Matching complete batch and nonempty attention layers required')
    rec = np.mean(np.square(x-y), axis=-1)
    layers = []
    for prior, values in zip(priors, series):
        s = np.asarray(values, dtype=np.float64)
        if s.ndim != 4 or s.shape[0] != len(x) or s.shape[2:] != (x.shape[1],)*2:
            raise ValueError('Complete batch/head/query/key attention required')
        if not np.isfinite(s).all() or np.any(s < 0):
            raise ValueError('Invalid series probabilities')
        s = s.mean(axis=1)
        if plain:
            if prior is not None:
                raise ValueError('Plain Transformer must have no prior computation')
            # The production tensor is float32; retain its stated log floor.
            tiny = np.finfo(np.asarray(values).dtype).tiny
            layer = -np.sum(s*np.log(np.maximum(s, tiny)), axis=-1)
        else:
            p = np.asarray(prior, dtype=np.float64)
            if p.shape != np.asarray(values).shape or not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError('Prior and series head shapes must match')
            mass = np.sum(p, axis=-1, keepdims=True)
            if np.any(mass <= 0):
                raise ValueError('Prior cannot be normalized')
            p = (p/mass).mean(axis=1)
            layer = np.sum((p-s)*(np.log(p+.0001)-np.log(s+.0001)), axis=-1)
        layers.append(layer)
    layers = np.stack(layers)
    result = {'all.reconstruction': rec}
    if not plain:
        def softmax_negative(discrepancy):
            values = -discrepancy
            values -= values.max(axis=-1, keepdims=True)
            values = np.exp(values)
            return values/values.sum(axis=-1, keepdims=True)
        association = softmax_negative(layers.mean(axis=0))
        result.update({'all.association': association, 'all.addition': rec+association,
                       'all.multiplication': rec*association})
        for index, layer in enumerate(layers):
            result[f'layer{index+1}.multiplication'] = rec*softmax_negative(layer)
    return result, rec, layers


def compare_formula(native, independent):
    if set(native) != set(independent):
        raise ValueError('All independent scoring components required')
    errors = {}
    for name in native:
        actual, expected = np.asarray(native[name]), np.asarray(independent[name])
        if (actual.shape != expected.shape or not np.isfinite(actual).all()
                or not np.isfinite(expected).all()
                or not np.allclose(actual, expected, **FORMULA_TOLERANCE)):
            raise ValueError('Independent formula mismatch: '+name)
        errors[name] = float(np.max(np.abs(actual.astype(np.float64)-expected)))
    return errors


def binary_counts(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError('Equal one-dimensional truth/prediction required')
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    p = tp/(tp+fp) if tp+fp else 0.
    r = tp/(tp+fn) if tp+fn else 0.
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.,
                true_positives=tp, false_positives=fp, false_negatives=fn)


def segment_adjust(truth, predicted):
    truth = np.asarray(truth, dtype=bool)
    result = np.asarray(predicted, dtype=bool).copy()
    if result.shape != truth.shape or truth.ndim != 1:
        raise ValueError('Equal one-dimensional segment arrays required')
    starts = np.flatnonzero(truth & np.r_[True, ~truth[:-1]])
    stops = np.flatnonzero(truth & np.r_[~truth[1:], True])+1
    for start, stop in zip(starts, stops):
        if result[start:stop].any():
            result[start:stop] = True
    return result


def ranking_metrics(truth, scores):
    """Mann-Whitney AUROC and threshold-group AP, including all score ties."""
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives or not negatives:
        return None, None
    order = np.argsort(scores, kind='stable')
    sorted_scores, labels = scores[order], truth[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])+1, len(scores)]
    group_positive = np.add.reduceat(labels.astype(np.int64), boundaries[:-1])
    group_negative = np.diff(boundaries)-group_positive
    lower_negative = np.cumsum(group_negative)-group_negative
    auc = float(np.sum(group_positive*(lower_negative+.5*group_negative))/(positives*negatives))
    tp = np.cumsum(group_positive[::-1])
    seen = np.cumsum((group_positive+group_negative)[::-1])
    ap = float(np.sum(group_positive[::-1]/positives*tp/seen))
    return auc, ap


def independent_evaluation(truth, scores, validation, ratio, delta, roc_ratios, plain=False):
    truth, scores, validation = np.asarray(truth), np.asarray(scores), np.asarray(validation)
    if (truth.ndim != 1 or scores.shape != truth.shape or validation.ndim != 1 or not validation.size
            or not np.isin(truth, [0, 1]).all()
            or not np.isfinite(scores).all() or not np.isfinite(validation).all()):
        raise ValueError('Complete finite validation/test scores and binary labels required')
    def at(threshold):
        predicted = scores > threshold
        return dict(threshold=float(threshold), no_PA=binary_counts(truth, predicted),
                    paper_PA=binary_counts(truth, segment_adjust(truth, predicted)))
    points = []
    for r in roc_ratios:
        item = dict(r_percent=r, **at(np.percentile(validation, 100-r)))
        for key in ['no_PA', 'paper_PA']:
            item[key]['false_positive_rate'] = item[key]['false_positives']/int((truth == 0).sum()) if (truth == 0).any() else None
            item[key]['true_positive_rate'] = item[key]['true_positives']/int((truth == 1).sum()) if (truth == 1).any() else None
        points.append(item)
    auc, ap = ranking_metrics(truth, scores)
    result = dict(validation_ratio=at(np.percentile(validation, 100-ratio)), fixed_delta=at(delta),
        continuous_unadjusted_score_AUROC=auc, continuous_score_AP=ap,
        seven_validation_calibrated_ROC_points=points,
        seven_point_curve_integration_not_paper_certified=True,
        test_features_or_labels_used_for_calibration=False)
    if plain:
        result.update(fixed_delta_is_diagnostic_only_for_plain_Transformer=True,
            primary_calibration='predeclared_validation_ratio_inherited_from_comparison_contract')
    return result


def require_same_metrics(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Different metric keys: '+path)
        for key in expected:
            require_same_metrics(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Different metric grid: '+path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            require_same_metrics(a, b, path+f'[{i}]')
    elif expected is None or isinstance(expected, (str, bool)):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError('Different metric metadata: '+path)
    elif isinstance(expected, int) or path.endswith('.threshold') or path.endswith('.r_percent'):
        if actual != expected:
            raise ValueError('Different exact count/threshold: '+path)
    elif not np.isfinite(actual) or abs(actual-expected) > 2e-12:
        raise ValueError('Independent ranking/rate metric mismatch: '+path)
