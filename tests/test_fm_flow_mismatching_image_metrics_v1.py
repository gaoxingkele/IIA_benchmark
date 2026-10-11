import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from scripts.flow_matching.flow_mismatching_image_metrics_v1 import (
    aupro, binary_metrics, image_scores_from_heatmaps, independent_aupro, per_category_metrics,
)


@pytest.mark.parametrize('seed', [7, 31, 1103])
def test_all_ties_and_exact_scores_match_independent_sklearn(seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, 101)
    for scores in (rng.integers(-3, 4, 101), rng.standard_normal(101), np.ones(101)):
        actual = binary_metrics(truth, scores)
        precision, recall, _ = precision_recall_curve(truth, scores)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-100)
        assert actual['AUROC'] == pytest.approx(roc_auc_score(truth, scores), abs=1e-12)
        assert actual['AP'] == pytest.approx(average_precision_score(truth, scores), abs=1e-12)
        assert actual['F1_max'] == pytest.approx(float(f1.max()), abs=1e-12)


@pytest.mark.parametrize('limit', [.1, .3, 1.])
def test_tied_aupro_cutoff_has_analytic_half_limit_area(limit):
    masks = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    scores = np.ones_like(masks, dtype=np.float32)
    actual = aupro(masks, scores, fpr_limit=limit, connectivity=8)
    assert actual['value'] == pytest.approx(limit / 2, abs=1e-12)
    assert independent_aupro(masks, scores, fpr_limit=limit, connectivity=8) == pytest.approx(limit / 2, abs=1e-12)


def test_perfect_separation_and_region_not_pixel_weighting():
    masks = np.array([[[1, 1, 0, 1], [0, 0, 0, 0]]], dtype=np.uint8)
    scores = np.array([[[2, 2, 1, 0], [1, 1, 1, 1]]], dtype=np.float32)
    assert aupro(masks, scores, fpr_limit=.3, connectivity=4)['value'] == pytest.approx(.5)
    assert independent_aupro(masks, scores, fpr_limit=.3, connectivity=4) == pytest.approx(.5)
    perfect = masks.astype(np.float32) * 10
    assert aupro(masks, perfect, fpr_limit=.3, connectivity=8)['value'] == pytest.approx(1.)


@pytest.mark.parametrize('connectivity', [4, 8])
def test_random_regions_and_fpr_crossing_match_independent_background_cdf(connectivity):
    rng = np.random.default_rng(51)
    masks = rng.integers(0, 2, (3, 5, 7)).astype(np.uint8)
    scores = rng.integers(0, 7, masks.shape).astype(np.float32)
    for limit in (.05, .3, .73, 1.):
        assert aupro(masks, scores, fpr_limit=limit, connectivity=connectivity)['value'] == pytest.approx(
            independent_aupro(masks, scores, fpr_limit=limit, connectivity=connectivity), abs=1e-12)


def test_connectivity_is_explicit_and_images_are_separate():
    masks = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.uint8)
    scores = masks.astype(np.float32)
    assert aupro(masks, scores, fpr_limit=.3, connectivity=4)['regions'] == 3
    assert aupro(masks, scores, fpr_limit=.3, connectivity=8)['regions'] == 2


def test_category_macro_mean_does_not_weight_image_counts():
    labels = np.array([1, 0, 1, 1, 0, 0, 0])
    categories = np.array([0, 0, 1, 1, 1, 1, 1])
    masks = labels[:, None, None] * np.array([[[1, 0], [0, 0]]])
    heatmaps = masks.astype(np.float32)
    scores = np.array([2, 0, 0, 0, 2, 2, 2.])
    result = per_category_metrics(heatmaps, scores, masks, labels, categories, ['small', 'large'],
                                  fpr_limit=.3, connectivity=8)
    assert result['category_macro_mean']['image_AUROC'] == pytest.approx(.5)
    assert result['complete_original_test_images'] == 7
    assert result['strict_TAB_result'] is False


def test_top_one_percent_uses_frozen_ceil_and_bad_data_rejected():
    values = np.arange(128 * 128, dtype=np.float32).reshape(1, 128, 128)
    expected = 16383 + np.mean(np.arange(16220, 16384, dtype=np.float32))
    assert image_scores_from_heatmaps(values, .01)[0] == expected
    for truth, score in [([1, 1], [1, 2]), ([1, 0], [1, np.nan]), ([2, 0], [1, 0])]:
        with pytest.raises(ValueError): binary_metrics(truth, score)
    with pytest.raises(ValueError): aupro(np.ones((1, 2, 2)), np.ones((1, 2, 2)), fpr_limit=.3, connectivity=8)
