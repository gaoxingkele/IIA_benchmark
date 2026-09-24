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


def test_dcdetector_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import DCdetectorDetector

    rng = np.random.default_rng(3)
    windows = rng.normal(size=(24, 12, 3)).astype(np.float32)
    detector = DCdetectorDetector(
        window=12,
        patch_sizes=(3, 4),
        d_model=16,
        n_heads=2,
        e_layers=1,
        epochs=1,
        batch_size=8,
        device="cpu",
    )
    detector.fit(windows)
    scores = detector.score(windows)
    assert scores.shape == (24, 12)
    assert np.isfinite(scores).all()
    # The score is a softmax over positions, so every window sums to one.
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(24), rtol=1e-4)


def test_timesnet_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import TimesNetDetector

    rng = np.random.default_rng(5)
    # A period-4 pattern makes the FFT period detector meaningful.
    base = np.tile(np.array([0.0, 1.0, 0.0, -1.0]), 8)
    windows = np.stack(
        [base + 0.01 * rng.normal(size=32) for _ in range(16)]
    )[:, :, None].astype(np.float32)
    detector = TimesNetDetector(
        window=32,
        d_model=8,
        d_ff=8,
        e_layers=1,
        top_k=2,
        epochs=1,
        batch_size=8,
        device="cpu",
    )
    detector.fit(windows)
    scores = detector.score(windows)
    assert scores.shape == (16, 32)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_itransformer_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import ITransformerDetector

    rng = np.random.default_rng(9)
    windows = rng.normal(size=(16, 24, 5)).astype(np.float32)
    detector = ITransformerDetector(
        window=24,
        d_model=16,
        d_ff=16,
        e_layers=1,
        n_heads=4,
        epochs=1,
        batch_size=8,
        device="cpu",
    )
    detector.fit(windows)
    scores = detector.score(windows)
    assert scores.shape == (16, 24)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_tranad_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import TranADDetector

    rng = np.random.default_rng(17)
    windows = rng.normal(size=(32, 10, 4)).astype(np.float32)
    detector = TranADDetector(window=10, epochs=1, batch_size=16, device="cpu")
    detector.fit(windows)
    scores = detector.score(windows)
    # One score per window: the error at the window's final timestamp.
    assert scores.shape == (32,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_catch_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import CATCHDetector

    rng = np.random.default_rng(23)
    windows = rng.normal(size=(16, 32, 6)).astype(np.float32)
    detector = CATCHDetector(
        window=32,
        patch_size=8,
        patch_stride=8,
        inference_patch_size=16,
        inference_patch_stride=1,
        cf_dim=8,
        d_model=8,
        d_ff=16,
        head_dim=8,
        n_heads=2,
        e_layers=1,
        epochs=1,
        batch_size=8,
        mask_update_every=4,
        device="cpu",
    )
    detector.fit(windows)
    scores = detector.score(windows)
    assert scores.shape == (16, 32)
    assert np.isfinite(scores).all()


def test_gcad_smoke() -> None:
    pytest.importorskip("torch")
    from iia_benchmark.models.mtsad_detectors import GCADDetector

    rng = np.random.default_rng(29)
    windows = rng.normal(size=(24, 12, 5)).astype(np.float32)
    detector = GCADDetector(
        window=12,
        pred_len=2,
        n_block=1,
        ff_dim=16,
        epochs=1,
        batch_size=8,
        sparse_th=0.0,
        device="cpu",
    )
    detector.fit(windows)
    scores = detector.score(windows)
    # next_steps mode drops the pred_len windows whose future is unavailable.
    assert scores.shape == (24 - 2,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
