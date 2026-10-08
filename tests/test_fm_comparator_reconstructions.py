"""Equation/behavior checks; small CPU arrays are not benchmark evidence."""
import numpy as np
import pytest
import torch

from iia_benchmark.models.fm_comparator_reconstructions import (
    BSplineActivation, ComparatorWindowDetector, DTLACore, KGLCore,
    SHCLTransformerCore, gaussian_kl, hierarchical_masks,
    modified_reverse_huber, scaled_softmax,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_mrh_published_branch_values_and_gradient():
    d = torch.tensor([0.5, 2.0], requires_grad=True)
    loss = modified_reverse_huber(d, 1.0)
    torch.testing.assert_close(loss, torch.tensor([0.25, 0.5]))
    loss.sum().backward()
    torch.testing.assert_close(d.grad, torch.tensor([1.0, -0.25]))
    with pytest.raises(ValueError):
        modified_reverse_huber(d, 0)


def test_scaled_softmax_preserves_zero_magnitude():
    torch.testing.assert_close(scaled_softmax(torch.zeros(2, 4)), torch.zeros(2, 4))
    value = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(scaled_softmax(value), value.softmax(-1) * value)


def test_hierarchical_masks_follow_paper_and_are_complementary():
    even, odd = hierarchical_masks(7, "point")
    assert even.tolist() == [True, False, True, False, True, False, True]
    assert torch.all(even ^ odd)
    even, odd = hierarchical_masks(7, "pair")
    assert even.tolist() == [False, False, True, True, False, False, True]
    assert torch.all(even ^ odd)


def test_gaussian_kl_direction_and_identity():
    zeros = torch.zeros(1, 2, 3)
    torch.testing.assert_close(gaussian_kl(zeros, zeros, zeros, zeros), torch.zeros(1, 2))
    value = gaussian_kl(torch.ones_like(zeros), zeros, zeros, zeros)
    torch.testing.assert_close(value, torch.full((1, 2), 1.5))


def test_cubic_basis_partition_and_trainable_coefficients():
    module = BSplineActivation(2)
    value = torch.tensor([[0.0, 0.1]])
    torch.testing.assert_close(module.basis(value).sum(-1), torch.ones_like(value))
    module(value).sum().backward()
    assert module.coefficients.grad.abs().sum() > 0


@pytest.mark.parametrize("method,options", [
    ("dt_la", {"d_model": 16, "heads": 2, "layers": 1}),
    ("shcl_transformer", {"d_model": 16, "heads": 2, "layers": 1}),
    ("kgl", {"d_model": 4, "heads": 2}),
])
def test_fit_score_and_frozen_inference(method, options):
    windows = np.random.default_rng(9).normal(size=(5, 8, 3)).astype("float32")
    detector = ComparatorWindowDetector(method, options, epochs=1, batch_size=2)
    detector.fit(windows)
    result = detector.score(windows)
    assert result.shape == (5, 8)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(detector.score(windows), result)
    parameters = {key: value.clone() for key, value in detector._model.state_dict().items()}
    detector.score(windows * 20)
    for key, value in parameters.items():
        torch.testing.assert_close(detector._model.state_dict()[key], value)


@pytest.mark.parametrize("loss,weight,score", [
    ("l2", 0, "input"), ("l2", 0.5, "latent"),
    ("mrh", 0, "softmax"), ("mrh", 0.5, "scaled_softmax"),
])
def test_dtla_ablation_paths(loss, weight, score):
    core = DTLACore(3, 16, 2, 1, latent_loss=loss, sparse_weight=weight, score_mode=score)
    x = torch.randn(2, 8, 3)
    value = core.loss(x)
    value.backward()
    assert torch.isfinite(value)
    assert core.score(x).shape == (2, 8)


@pytest.mark.parametrize("options", [
    {"branches": ("point",)}, {"branches": ("pair",)},
    {"masking": "random"}, {"objective": "reconstruction"},
])
def test_shcl_ablation_paths(options):
    core = SHCLTransformerCore(3, 16, 2, 1, **options)
    value = core.loss(torch.randn(2, 7, 3))
    value.backward()
    assert torch.isfinite(value)


@pytest.mark.parametrize("removed", ["use_gat", "use_kan", "use_lstm"])
def test_kgl_each_module_can_be_removed(removed):
    core = KGLCore(3, 4, 2, **{removed: False})
    x = torch.randn(2, 8, 3)
    core.loss(x).backward()
    assert core.score(x).shape == (2, 8)
    torch.testing.assert_close(core.score(x)[:, 0], torch.zeros(2))


def test_bad_inputs_and_unfitted_scoring():
    detector = ComparatorWindowDetector("dt_la")
    with pytest.raises(RuntimeError):
        detector.score(np.zeros((2, 8, 3)))
    with pytest.raises(ValueError):
        detector.fit(np.zeros((8, 3)))


@pytest.mark.parametrize("backbone", ["transformer", "vae", "cnn", "rnn", "lstm"])
def test_shcl_all_paper_backbones_fit_and_keep_shared_mask_score_contract(backbone):
    windows = np.random.default_rng(22).normal(size=(4, 8, 3)).astype("float32")
    detector = ComparatorWindowDetector("shcl_transformer", {"d_model":16,"heads":2,"layers":2,"backbone":backbone},
                                         epochs=1,batch_size=2)
    detector.fit(windows)
    scores = detector.score(windows)
    assert scores.shape == (4, 8)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(detector.score(windows), scores)
