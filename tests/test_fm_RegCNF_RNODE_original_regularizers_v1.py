"""Original code semantics only; analytic fixtures are not benchmark results."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/models/fm_RegCNF_RNODE_author_source.v1.json'


def original():
    c = json.loads(CONFIG.read_text())
    root = Path(c['source_root'])
    proof = json.loads((root.parent / 'verified_acquisition.json').read_text())
    assert proof['passed'] and proof['complete_source_files'] == 32
    assert proof['source_commit'] == c['source_commit']
    for row in proof['source_files']:
        assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
    path = root / 'lib/layers/wrappers/cnf_regularization.py'
    spec = importlib.util.spec_from_file_location('_finlay_original_regularization_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_original_kinetic_scaling_and_differentiation():
    reg = original()
    x = torch.arange(15, dtype=torch.float64).reshape(3, 5).requires_grad_()
    t = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    dx = 2 * x + t
    value = reg.quadratic_cost(x, t, None, dx, None, None)
    torch.testing.assert_close(value, dx.square().sum(dim=1) / (2 * 5))
    value.sum().backward()
    torch.testing.assert_close(x.grad, 2 * dx.detach() / 5)


def test_all_four_author_regularizers_execute_the_original_functions():
    reg = original()
    x = torch.arange(15, dtype=torch.float64).reshape(3, 5).requires_grad_()
    t = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    dx = 2 * x + t
    context = SimpleNamespace(sqjacnorm=torch.full((3,), 4.0, dtype=torch.float64))
    torch.testing.assert_close(reg.directional_derivative(x, t, None, dx, None, None),
                               0.5 * (2 * dx).square().mean(dim=1))
    # For dx/dt=1, differentiating the intermediate time VJP with respect
    # to its auxiliary vector must recover ones, regardless of its initial value.
    total = reg.total_derivative(x, t, None, dx, None, None)
    torch.testing.assert_close(total, 0.5 * (2 * dx + torch.ones_like(dx)).square().mean(dim=1))
    assert reg.jacobian_frobenius_regularization_fn(x, t, None, dx, None, context) is context.sqjacnorm
    assert torch.isfinite(total).all()
    c = json.loads(CONFIG.read_text())
    assert len(c['source_regularizers']) == 4 and len(c['author_image_recipes']) == 4
    assert not c['full_original_image_training_complete'] and not c['single_cell_adaptation_complete']
