import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts.flow_matching.otcfm_sbcfm_pairwise_compat_v1 import pairwise_formula_adapter, install


def original_formula():
    root = Path(__file__).resolve().parents[1]
    path = root / 'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/runner/src/models/cfm_module.py'
    cls = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n, ast.ClassDef) and n.name == 'SBCFMLitModule')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'calc_mu_sigma')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['calc_mu_sigma']


@pytest.mark.parametrize('trajectory', [False, True])
def test_real_original_SB_formula_values_and_gradients_are_unchanged_for_normalized_pairs(trajectory):
    original = original_formula()
    model = SimpleNamespace(is_trajectory=trajectory, hparams=SimpleNamespace(sigma_min=1.))
    x0 = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    x1 = x0.flip(0); t = torch.tensor([[.01], [.2], [.7], [.99]], dtype=torch.float64, requires_grad=True)
    if trajectory:
        with pytest.raises(AssertionError): original(model, x0, x1, t)
    expected = original(SimpleNamespace(is_trajectory=False, hparams=model.hparams), x0, x1, t)
    actual = pairwise_formula_adapter(original)(model, x0, x1, t)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert model.is_trajectory is trajectory
    for a, b in zip(actual, expected):
        ga, = torch.autograd.grad(a.sum(), t, retain_graph=True)
        gb, = torch.autograd.grad(b.sum(), t, retain_graph=True)
        assert torch.equal(ga, gb)
    assert torch.equal(actual[0], (1 - t) * x0 + t * x1)
    assert torch.equal(actual[1], torch.sqrt(t - t.square()))


def test_pairwise_repair_install_is_idempotent_and_local_to_selected_class():
    class Released: calc_mu_sigma = original_formula()
    class Isolated(Released): pass
    original = Released.calc_mu_sigma
    install(Isolated); fixed = Isolated.calc_mu_sigma; install(Isolated)
    assert fixed is Isolated.calc_mu_sigma and Released.calc_mu_sigma is original
