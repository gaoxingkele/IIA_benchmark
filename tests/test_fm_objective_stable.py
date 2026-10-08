"""Check unchanged entropic objective and convergence on sharp real-scale costs."""
import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_objective_detectors import minibatch_plan, CFMObjectiveDetector
from iia_benchmark.models.fm_objective_stable import stable_plan, StableCFMObjectiveDetector


def test_continuation_matches_original_entropic_optimum():
    x0 = torch.tensor([[0.], [2.], [5.]])
    x1 = x0[[2, 0, 1]]
    expected = minibatch_plan(x0, x1, 'sinkhorn', 5., tolerance=1e-10)
    actual = stable_plan(x0, x1, 5., tolerance=1e-10)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(actual.sum(0), 1 / 3, atol=1e-10)
    np.testing.assert_allclose(actual.sum(1), 1 / 3, atol=1e-10)


def test_sharp_high_dimensional_transport_does_not_fall_back():
    rng = torch.Generator().manual_seed(8)
    x0 = torch.randn((32, 2500), generator=rng)
    x1 = torch.randn((32, 2500), generator=rng)
    plan = stable_plan(x0, x1, 2.)
    np.testing.assert_allclose(plan.sum(0), 1 / 32, atol=1e-6)
    np.testing.assert_allclose(plan.sum(1), 1 / 32, atol=1e-6)
    assert not np.allclose(plan, np.full_like(plan, 1 / 32 ** 2))


def test_solver_failure_stays_visible():
    with pytest.raises(RuntimeError, match='did not converge'):
        stable_plan(torch.tensor([[0.], [5.], [15.]]), torch.tensor([[2.], [7.], [20.]]), 2., tolerance=1e-15, max_iterations=1)


def test_repair_preserves_easy_problem_backbone_loss_and_scores():
    values = np.random.default_rng(5).normal(size=(4, 3, 2)).astype('float32')
    parameters = dict(objective='sf2m', sigma=1., window=3, hidden_size=8, epochs=1, batch_size=4, monte_carlo_samples=1, device='cpu')
    original = CFMObjectiveDetector(**parameters).fit(values)
    corrected = StableCFMObjectiveDetector(**parameters).fit(values)
    np.testing.assert_allclose(corrected.training_losses_, original.training_losses_, rtol=1e-6)
    np.testing.assert_allclose(corrected.score(values), original.score(values), rtol=1e-5, atol=1e-6)
