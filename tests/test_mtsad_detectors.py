from __future__ import annotations

import numpy as np
import pytest

from iia_benchmark.models.mtsad_detectors import (
    MahalanobisDetector,
    PCADetector,
    build_detector,
)


def test_build_detector_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        build_detector("nope")


def test_pca_flags_a_shifted_point() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 4))
    detector = PCADetector(n_components=4).fit(train)
    normal = rng.normal(size=(50, 4))
    anomalous = np.vstack([normal, np.full((1, 4), 12.0)])
    scores = detector.score(anomalous)
    assert scores[-1] == scores.max()


def test_mahalanobis_is_translation_invariant() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(400, 3))
    shifted = train + 5.0
    # Fitting and scoring on the shifted data must reproduce the original
    # scores, because the distance is defined relative to the fitted centre.
    baseline = MahalanobisDetector().fit(train).score(train)
    moved = MahalanobisDetector().fit(shifted).score(shifted)
    np.testing.assert_allclose(moved, baseline, rtol=1e-6, atol=1e-6)


def test_window_detectors_smoke() -> None:
    torch = pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import (
        AnomalyTransformerDetector,
        USADDetector,
    )

    rng = np.random.default_rng(2)
    windows = rng.normal(size=(64, 8, 3)).astype(np.float32)
    usad = USADDetector(window=8, hidden_size=8, latent_size=4, epochs=1, device="cpu")
    usad.fit(windows)
    scores = usad.score(windows)
    assert scores.shape == (64,)
    assert np.isfinite(scores).all()

    transformer = AnomalyTransformerDetector(
        window=8, d_model=8, n_heads=2, e_layers=1, d_ff=8, epochs=1, batch_size=32,
        device="cpu",
    )
    transformer.fit(windows)
    scores = transformer.score(windows)
    # The Anomaly Transformer scores every position inside each window.
    assert scores.shape == (64, 8)
    assert np.isfinite(scores).all()
