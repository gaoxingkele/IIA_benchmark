import numpy as np

from iia_benchmark.models import (
    HDAMTemplateMatcher,
    extract_hdam_template,
    high_density_alarm_matrix,
    structured_hdam_alignment,
)


def test_high_density_alarm_matrix_binning() -> None:
    series = np.array([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=float)
    np.testing.assert_array_equal(
        high_density_alarm_matrix(series, bin_size=2, aggregation="count"),
        [[1, 2], [1, 0]],
    )
    np.testing.assert_array_equal(
        high_density_alarm_matrix(series, bin_size=2, aggregation="binary"),
        [[1, 1], [1, 0]],
    )


def test_structured_convolution_recovers_time_shift() -> None:
    template = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    candidate = np.pad(template, ((0, 0), (2, 1)))
    alignment = structured_hdam_alignment(template, candidate)
    assert alignment.score == 1.0
    assert alignment.template_start == 0
    assert alignment.candidate_start == 2
    assert alignment.overlap_width == 3


def _episodes() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((8, 3, 16))
    y = np.array(["forward"] * 4 + ["reverse"] * 4)
    for index in range(4):
        offset = index % 2
        X[index, 0, 2 + offset] = 1
        X[index, 1, 5 + offset] = 1
        X[index, 2, 8 + offset] = 1
        X[index + 4, 2, 2 + offset] = 1
        X[index + 4, 1, 5 + offset] = 1
        X[index + 4, 0, 8 + offset] = 1
    return X, y


def test_template_extraction_aligns_variable_duration_floods() -> None:
    base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    matrices = [np.pad(base, ((0, 0), (left, 4 - left))) for left in (0, 1, 2)]
    template = extract_hdam_template(matrices, "fault", template_width=3)
    assert template.values.shape == (3, 3)
    assert template.stability == 1.0
    np.testing.assert_allclose(template.values, base)


def test_hdam_templates_classify_and_emit_probabilities() -> None:
    X, y = _episodes()
    classifier = HDAMTemplateMatcher(template_width=7).fit(X, y)
    np.testing.assert_array_equal(classifier.predict(X), y)
    np.testing.assert_allclose(np.sum(classifier.predict_proba(X), axis=1), 1.0)
    assert all(template.sample_count == 4 for template in classifier.templates_)
    assert all(0 <= template.stability <= 1 for template in classifier.templates_)


def test_dynamic_hdam_prefix_classification() -> None:
    X, y = _episodes()
    classifier = HDAMTemplateMatcher(template_width=7).fit(X, y)
    evolution = classifier.predict_evolution(X, [7, 16])
    assert tuple(evolution) == (7, 16)
    np.testing.assert_array_equal(evolution[16], y)
