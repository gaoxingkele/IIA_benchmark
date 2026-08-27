import numpy as np
import pytest

from iia_benchmark.models import (
    KernelPCAFaultIsolator,
    ModifiedTFIDFVectorizer,
    TFIDFLSTMAlarmFloodClassifier,
    optimize_alert_threshold,
    optimize_ngram_size,
)


def _sequences():
    forward = [("A", "B", "C", "D"), ("A", "B", "C", "D"), ("A", "B", "C", "C")]
    reverse = [("D", "C", "B", "A"), ("D", "C", "B", "A"), ("D", "C", "B", "B")]
    return forward + reverse, np.array(["forward"] * 3 + ["reverse"] * 3)


def test_modified_tfidf_retains_position_and_ngram_order() -> None:
    vectorizer = ModifiedTFIDFVectorizer(ngram_size=1, position_decay=2.0)
    values = vectorizer.fit_transform([("A", "B"), ("B", "A")])
    assert values.shape == (2, 2)
    assert not np.allclose(values[0], values[1])
    bigrams = ModifiedTFIDFVectorizer(ngram_size=2).fit([("A", "B"), ("B", "A")])
    assert set(bigrams.vocabulary_) == {("A", "B"), ("B", "A")}


def test_ngram_optimization_returns_a_frozen_score_grid() -> None:
    sequences, _ = _sequences()
    selection = optimize_ngram_size(sequences, [1, 2], n_clusters=2, random_state=3)
    assert selection.ngram_size in {1, 2}
    assert set(selection.scores) == {1, 2}
    assert np.isfinite(selection.silhouette)


def test_alert_threshold_optimizes_balanced_accuracy() -> None:
    result = optimize_alert_threshold(
        np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])
    )
    assert result.balanced_accuracy == 1.0
    assert 0.2 < result.threshold <= 0.8


def test_kernel_pca_fault_isolation_and_feature_ranking() -> None:
    normal = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.04, 0.03]])
    fault = np.array([[1.0, 0.0], [0.0, 1.0]])
    isolator = KernelPCAFaultIsolator(
        n_components=1, kernel="linear", normal_quantile=0.9
    ).fit(normal)
    calibration = np.vstack((normal, fault))
    threshold = isolator.calibrate_threshold(
        calibration, np.array([False] * len(normal) + [True] * len(fault))
    )
    assert threshold.balanced_accuracy >= 0.75
    assert np.all(isolator.predict(fault))
    assert isolator.influential_features(fault, top_k=1).shape == (2, 1)


def test_lstm_early_classifier_is_trainable_and_normalized() -> None:
    pytest.importorskip("torch")
    sequences, labels = _sequences()
    classifier = TFIDFLSTMAlarmFloodClassifier(
        ngram_size=2,
        hidden_size=8,
        epochs=40,
        learning_rate=0.03,
        random_state=7,
    ).fit(sequences, labels)
    probabilities = classifier.predict_proba(sequences)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6)
    assert classifier.training_loss_[-1] < classifier.training_loss_[0]
    assert np.mean(classifier.predict(sequences) == labels) >= 5 / 6
    evolution = classifier.predict_evolution(sequences, [2, 4])
    assert tuple(evolution) == (2, 4)
