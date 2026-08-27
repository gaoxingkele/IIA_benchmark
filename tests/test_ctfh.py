import numpy as np

from iia_benchmark.models import (
    CTFHAlarmFloodClassifier,
    alarm_evolution_matrix,
    combinatorial_temporal_fingerprints,
    extract_aem_peaks,
)


def test_alarm_evolution_matrix_is_sliding_activation_rate() -> None:
    series = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
    aem = alarm_evolution_matrix(series, window_size=2, stride=1)
    np.testing.assert_array_equal(aem.window_starts, [0, 1, 2])
    np.testing.assert_allclose(aem.values, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])


def test_peak_plateau_and_hashes_are_deterministic() -> None:
    series = np.zeros((3, 12))
    series[0, 1:3] = 1
    series[1, 5:7] = 1
    series[2, 9:11] = 1
    aem = alarm_evolution_matrix(series, window_size=2)
    peaks = extract_aem_peaks(aem, threshold=0.5, temporal_radius=1)
    assert len({(item.window_index, item.tag_index) for item in peaks}) == len(peaks)
    first = combinatorial_temporal_fingerprints(peaks, max_delta=10, fanout=3)
    second = combinatorial_temporal_fingerprints(peaks, max_delta=10, fanout=3)
    assert first == second
    assert all(0 <= item.code < 2**64 for item in first)


def _episodes() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((8, 3, 20))
    y = np.array(["forward"] * 4 + ["reverse"] * 4)
    for index in range(4):
        offset = index % 2
        X[index, 0, 2 + offset] = 1
        X[index, 1, 7 + offset] = 1
        X[index, 2, 12 + offset] = 1
        X[index + 4, 2, 2 + offset] = 1
        X[index + 4, 1, 7 + offset] = 1
        X[index + 4, 0, 12 + offset] = 1
    return X, y


def test_consensus_profiles_classify_temporal_fingerprints() -> None:
    X, y = _episodes()
    classifier = CTFHAlarmFloodClassifier(
        window_size=1,
        peak_threshold=1.0,
        max_delta=12,
        fanout=3,
        delta_quantization=2,
    ).fit(X, y)
    np.testing.assert_array_equal(classifier.predict(X), y)
    np.testing.assert_allclose(np.sum(classifier.predict_proba(X), axis=1), 1.0)
    assert all(0 <= profile.variability_index <= 1 for profile in classifier.profiles_)
    assert all(profile.sample_count == 4 for profile in classifier.profiles_)


def test_online_prefix_predictions_are_emitted() -> None:
    X, y = _episodes()
    classifier = CTFHAlarmFloodClassifier(
        window_size=1,
        peak_threshold=1.0,
        max_delta=12,
        fanout=3,
        delta_quantization=2,
    ).fit(X, y)
    evolution = classifier.predict_evolution(X, [10, 20])
    assert tuple(evolution) == (10, 20)
    np.testing.assert_array_equal(evolution[20], y)


def test_fingerprint_transform_rejects_negative_alarm_counts() -> None:
    classifier = CTFHAlarmFloodClassifier(window_size=1)
    try:
        classifier.transform(np.array([[[-1.0, 0.0]]]))
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative activations must be rejected")
