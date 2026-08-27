import numpy as np

from iia_benchmark.models.casim import (
    CASIMClassifier,
    CASIMExpandingWindowClassifier,
    CASIMFeatureTransformer,
    LocalOutlierProbability,
    _canonical_kernels,
    casim_loop_features,
)


def _episodes(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((16, 4, 24), dtype=float)
    y = np.repeat(["early", "late"], 8)
    for index in range(8):
        X[index, 0, 3:9] = 1
        X[index, 1, 8:13] = 1
        X[index + 8, 2, 12:18] = 1
        X[index + 8, 3, 17:22] = 1
    X = np.clip(X + rng.binomial(1, 0.01, X.shape), 0, 1)
    return X, y


def test_canonical_kernel_family_is_complete() -> None:
    kernels = _canonical_kernels()
    assert len(kernels) == 84
    assert len({tuple(kernel) for kernel in kernels}) == 84
    assert all(np.sum(kernel == 2) == 3 and np.sum(kernel == -1) == 6 for kernel in kernels)


def test_casim_transform_has_paper_minimum_672_features() -> None:
    X, _ = _episodes()
    transformer = CASIMFeatureTransformer(n_features=672, random_state=7)
    features = transformer.fit_transform(X)
    assert features.shape == (16, 672)
    assert len(transformer.kernels_) == 84
    assert all(1 <= len(kernel.channels) <= 4 for kernel in transformer.kernels_)
    assert np.all(np.isfinite(features))


def test_loop_feature_appends_top_two_probability_gap() -> None:
    result = casim_loop_features(np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]))
    np.testing.assert_allclose(result[:, -1], [0.5, 0.2])
    assert result.shape == (2, 4)


def test_loop_scores_distant_samples_as_more_novel() -> None:
    known = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    loop = LocalOutlierProbability(n_neighbors=2).fit(known)
    close, distant = loop.predict_proba(np.array([[0.05, 0.05], [4.0, 4.0]]))
    assert 0 <= close <= 1
    assert distant > close
    assert distant > 0.9


def test_casim_classifier_probability_and_smote_invariants() -> None:
    X, y = _episodes()
    model = CASIMClassifier(n_features=32, n_classifiers=2, k_loop=3, random_state=2).fit(X, y)
    probabilities = model.predict_proba(X[:4])
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
    assert probabilities.shape == (4, 2)
    labels, counts = np.unique(model.loop_training_labels_, return_counts=True)
    assert set(labels) == set(y)
    assert np.min(counts) >= 4
    assert np.mean(model.predict(X, novelty_threshold=1.0) == y) >= 0.75


def test_expanding_window_trains_separate_prefix_stages() -> None:
    X, y = _episodes()
    model = CASIMExpandingWindowClassifier(
        (12, 24), n_features=16, n_classifiers=1, k_loop=2, random_state=1
    ).fit(X, y)
    predictions = model.predict_evolution(X[:2], novelty_threshold=1.0)
    assert tuple(predictions) == (12, 24)
    assert all(value.shape == (2,) for value in predictions.values())
