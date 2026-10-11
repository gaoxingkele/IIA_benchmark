"""Exact tied-score image/pixel metrics and explicitly local AUPRO convention."""
import math
import numpy as np
from scipy.ndimage import generate_binary_structure, label

METRICS = ('image_AUROC', 'image_AP', 'image_F1_max',
           'pixel_AUROC', 'pixel_AP', 'pixel_F1_max', 'pixel_AUPRO')


def binary_metrics(target, scores):
    truth, values = np.asarray(target).ravel(), np.asarray(scores).ravel()
    if (truth.shape != values.shape or not len(truth)
            or not np.isfinite(values).all() or not np.isin(truth, (0, 1)).all()):
        raise ValueError('Complete finite binary targets and scores required')
    positives, negatives = int(truth.sum()), int(len(truth) - truth.sum())
    if positives == 0 or negatives == 0:
        raise ValueError('Both positive and negative original units required')
    order = np.argsort(values, kind='stable')[::-1]
    sorted_scores, sorted_truth = values[order], truth[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(truth) - 1]
    tp = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    fp = (ends + 1) - tp
    recall, precision = tp / positives, tp / (tp + fp)
    f1 = 2 * tp / (positives + tp + fp)
    best = int(np.argmax(f1))
    auc = float(np.trapezoid(np.r_[0., recall], np.r_[0., fp / negatives]))
    ap = float(np.dot(np.diff(np.r_[0., recall]), precision))
    return dict(AUROC=auc, AP=ap, F1_max=float(f1[best]),
                oracle_F1_threshold=float(sorted_scores[ends[best]]),
                positive_units=positives, negative_units=negatives,
                F1_threshold_source='complete test-label sweep; never strict TAB calibration')


def regions_per_image(masks, connectivity):
    truth = np.asarray(masks)
    if truth.ndim != 3 or not np.isin(truth, (0, 1)).all() or connectivity not in (4, 8):
        raise ValueError('Binary [image,height,width] masks and registered connectivity required')
    structure = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    regions = []
    for mask in truth:
        component, count = label(mask, structure=structure)
        regions.append((component, count))
    return regions


def aupro(masks, heatmaps, *, fpr_limit, connectivity):
    truth, scores = np.asarray(masks), np.asarray(heatmaps)
    if truth.shape != scores.shape or not np.isfinite(scores).all() or not 0 < fpr_limit <= 1:
        raise ValueError('Full finite masks/heatmaps and registered FPR interval required')
    regions = regions_per_image(truth, connectivity)
    total_regions = sum(count for _, count in regions)
    backgrounds = int(np.count_nonzero(truth == 0))
    if not total_regions or not backgrounds:
        raise ValueError('Original positive regions and background pixels required')
    positive_weights = np.zeros(truth.shape, dtype=np.float64)
    for image, (component, count) in enumerate(regions):
        sizes = np.bincount(component.ravel(), minlength=count + 1).astype(np.float64)
        weights = np.zeros(count + 1)
        weights[1:] = 1 / (sizes[1:] * total_regions)
        positive_weights[image] = weights[component]
    flat = scores.ravel()
    order = np.argsort(flat, kind='stable')[::-1]
    ends = np.r_[np.flatnonzero(np.diff(flat[order])), len(flat) - 1]
    pro = np.cumsum(positive_weights.ravel()[order], dtype=np.float64)[ends]
    fpr = np.cumsum((truth.ravel()[order] == 0), dtype=np.int64)[ends] / backgrounds
    x, y = np.r_[0., fpr], np.r_[0., pro]
    inside = np.flatnonzero(x <= fpr_limit)
    last = int(inside[-1])
    xs, ys = x[:last + 1], y[:last + 1]
    if xs[-1] < fpr_limit:
        following = last + 1
        boundary_y = ys[-1] + ((fpr_limit - xs[-1]) / (x[following] - xs[-1])) * (y[following] - ys[-1])
        xs, ys = np.r_[xs, fpr_limit], np.r_[ys, boundary_y]
    return dict(value=float(np.trapezoid(ys, xs) / fpr_limit),
                regions=total_regions, background_pixels=backgrounds,
                fpr_limit=fpr_limit, connectivity=connectivity,
                convention='equal region weight; exact unique-score threshold groups; linear FPR cutoff interpolation')


def independent_aupro(masks, heatmaps, *, fpr_limit, connectivity):
    """Different integration: each region's positive scores against background CDF.

    Analytic area contribution handles tied positive/background score groups,
    including a tie group crossing the FPR cutoff. No global PRO curve reused.
    """
    truth, scores = np.asarray(masks), np.asarray(heatmaps)
    if truth.shape != scores.shape or not np.isfinite(scores).all() or not 0 < fpr_limit <= 1:
        raise ValueError('Complete finite original masks and scores required')
    background = np.sort(scores[truth == 0])
    if not len(background):
        raise ValueError('Original background required')
    values = []
    for image, (component, count) in enumerate(regions_per_image(truth, connectivity)):
        for region in range(1, count + 1):
            region_scores = scores[image][component == region]
            low = (len(background) - np.searchsorted(background, region_scores, side='right')) / len(background)
            high = (len(background) - np.searchsorted(background, region_scores, side='left')) / len(background)
            contribution = np.zeros(len(region_scores), dtype=np.float64)
            before = high <= fpr_limit
            contribution[before] = fpr_limit - (low[before] + high[before]) / 2
            crossing = (low < fpr_limit) & (high > fpr_limit)
            contribution[crossing] = (fpr_limit - low[crossing]) ** 2 / (2 * (high[crossing] - low[crossing]))
            values.append(float(contribution.mean() / fpr_limit))
    if not values:
        raise ValueError('Original positive regions required')
    return float(np.mean(values))


def per_category_metrics(heatmaps, image_scores, masks, labels, categories, category_names,
                         *, fpr_limit, connectivity):
    n = len(labels)
    if (heatmaps.shape != masks.shape or len(heatmaps) != n or image_scores.shape != (n,)
            or categories.shape != (n,) or len(category_names) < 1
            or set(np.unique(categories)) != set(range(len(category_names)))):
        raise ValueError('Every native category/test image and original array order required')
    rows = []
    for index, name in enumerate(category_names):
        subset = categories == index
        image = binary_metrics(labels[subset], image_scores[subset])
        pixel = binary_metrics(masks[subset], heatmaps[subset])
        region = aupro(masks[subset], heatmaps[subset], fpr_limit=fpr_limit, connectivity=connectivity)
        values = {prefix + key: result[key] for prefix, result in [('image_', image), ('pixel_', pixel)]
                  for key in ('AUROC', 'AP', 'F1_max')}
        values['pixel_AUPRO'] = region['value']
        rows.append(dict(category=name, category_index=index, original_test_images=int(subset.sum()),
                         metrics=values, image_curve=image, pixel_curve=pixel, region_metric=region))
    means = {name: float(np.mean([row['metrics'][name] for row in rows])) for name in METRICS}
    return dict(per_category=rows, category_macro_mean=means,
                category_macro_percent={name: value * 100 for name, value in means.items()},
                original_categories=len(category_names), complete_original_test_images=n,
                category_aggregation='unweighted category mean; no pooled category weighting',
                strict_TAB_result=False, author_metric_implementation_equivalence_certified=False)


def image_scores_from_heatmaps(heatmaps, top_fraction):
    if heatmaps.ndim != 3 or not 0 < top_fraction <= 1:
        raise ValueError('Complete image heatmaps and registered top fraction required')
    flat = heatmaps.reshape(len(heatmaps), -1)
    top = max(1, math.ceil(flat.shape[1] * top_fraction))
    return flat.max(1) + np.partition(flat, flat.shape[1] - top, axis=1)[:, -top:].mean(1)
