"""Original augmented vector field with analytic Jacobian and full1024batch."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from iia_benchmark.models import otcfm_cnf_trace_shape_v1 as s

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/runner/src/models/components/augmentation.py'


def original():
    spec = importlib.util.spec_from_file_location('_source_CNF_augmentation_shape_fixture', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('batch', [1, 5, 1024])
def test_exact_negative_trace_per_sample_and_weight_gradient(batch):
    source = original()
    augmentation = source.AugmentationModule(cnf_estimator='exact')
    reg = augmentation.regs[0]
    old_class_forward = type(reg).forward
    shared = SimpleNamespace(augmentation_list=augmentation.regs)
    s.install_model(SimpleNamespace(augmentations=augmentation, aug_net=shared))
    x = torch.arange(batch * 5, dtype=torch.float64).reshape(batch, 5).requires_grad_()
    matrix = torch.arange(25, dtype=torch.float64).reshape(5, 5).requires_grad_()
    dx = x @ matrix.T
    value = reg(None, x, dx, None)
    assert value.shape == (batch,)
    torch.testing.assert_close(value, -torch.trace(matrix).expand(batch))
    gx, gm = torch.autograd.grad(value.mean(), [x, matrix])
    torch.testing.assert_close(gx, torch.zeros_like(x))
    torch.testing.assert_close(gm, -torch.eye(5, dtype=torch.float64))
    assert type(reg).forward is old_class_forward
    assert source.CNFReg().forward.__func__ is old_class_forward


def test_actual_augmented_vectorfield_shape_state_graph_and_original_failure():
    source = original()
    class Drift(torch.nn.Module):
        def forward(self, t, x):
            return x * torch.arange(1, 6).to(x)
    aug = source.AugmentationModule(cnf_estimator='exact')
    field = source.AugmentedVectorField(Drift(), aug.regs, 5)
    state = torch.arange(6144, dtype=torch.float64).reshape(1024, 6).requires_grad_()
    with pytest.raises(RuntimeError, match='size of tensor'):
        field(torch.tensor(0.), state)
    s.install_model(SimpleNamespace(augmentations=aug, aug_net=field))
    result = field(torch.tensor(0.), state)
    assert result.shape == (1024, 6)
    torch.testing.assert_close(result[:, 0], torch.full((1024,), -15., dtype=torch.float64))
    result.sum().backward()
    torch.testing.assert_close(state.grad[:, 0], torch.zeros(1024, dtype=torch.float64))
    torch.testing.assert_close(state.grad[:, 1:], torch.arange(1, 6, dtype=torch.float64).expand(1024, 5))
