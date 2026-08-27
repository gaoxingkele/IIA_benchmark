import numpy as np
import pytest

from iia_benchmark.models import (
    OptimalTimeEncodedHistogramClassifier,
    TimedAlarmSequence,
    time_encoded_histogram,
)


def test_time_encoded_histogram_has_exact_exponential_weights() -> None:
    sequence = TimedAlarmSequence.from_events([(0.0, "A"), (0.5, "B"), (1.0, "A")])
    values = time_encoded_histogram(sequence, ("A", "B"), attenuation=np.log(2))
    expected = np.array([1.0 + 0.5, np.sqrt(0.5)])
    expected /= np.sum(expected)
    np.testing.assert_allclose(values, expected)


def test_early_alarms_receive_more_weight() -> None:
    sequence = TimedAlarmSequence.from_events([(0.0, "A"), (1.0, "B")])
    values = time_encoded_histogram(sequence, ("A", "B"), attenuation=2.0)
    assert values[0] > values[1]
    np.testing.assert_allclose(np.sum(values), 1.0)


def _sequences():
    sequences = []
    labels = []
    for index in range(5):
        jitter = index * 0.01
        sequences.append(
            TimedAlarmSequence.from_events(
                [(0.0, "A"), (1.0 + jitter, "B"), (2.0 + jitter, "C")]
            )
        )
        labels.append("forward")
        sequences.append(
            TimedAlarmSequence.from_events(
                [(0.0, "C"), (1.0 + jitter, "B"), (2.0 + jitter, "A")]
            )
        )
        labels.append("reverse")
    return sequences, np.asarray(labels)


def test_hybrid_three_phase_training_and_joint_attenuation() -> None:
    pytest.importorskip("torch")
    sequences, labels = _sequences()
    classifier = OptimalTimeEncodedHistogramClassifier(
        latent_size=3,
        autoencoder_hidden_size=8,
        transformer_width=4,
        attention_heads=2,
        pretrain_epochs=20,
        classifier_epochs=20,
        joint_epochs=30,
        learning_rate=0.03,
        random_state=5,
    ).fit(sequences, labels)
    assert set(classifier.training_history_) == {"autoencoder", "classifier", "joint"}
    assert all(values[-1] < values[0] for values in classifier.training_history_.values())
    assert classifier.attenuation_ > 0
    np.testing.assert_allclose(
        np.sum(classifier.encode_histograms(sequences), axis=1), 1.0, atol=1e-6
    )
    probabilities = classifier.predict_proba(sequences)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6)
    assert np.mean(classifier.predict(sequences) == labels) >= 0.9


def test_hybrid_emits_prefix_predictions() -> None:
    pytest.importorskip("torch")
    sequences, labels = _sequences()
    classifier = OptimalTimeEncodedHistogramClassifier(
        latent_size=3,
        autoencoder_hidden_size=6,
        transformer_width=4,
        attention_heads=2,
        pretrain_epochs=10,
        classifier_epochs=10,
        joint_epochs=15,
        learning_rate=0.04,
        random_state=9,
    ).fit(sequences, labels)
    evolution = classifier.predict_evolution(sequences, [1, 3])
    assert tuple(evolution) == (1, 3)
    assert evolution[3].shape == labels.shape
