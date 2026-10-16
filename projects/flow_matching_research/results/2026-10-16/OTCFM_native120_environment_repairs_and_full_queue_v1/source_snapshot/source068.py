import pytest
import torch
from scripts.flow_matching.otcfm_empty_ode_args_compat_v1 import empty_args_adapter, install


class Legacy:
    def drift(self, t, x):
        return x.square() * t + x


@pytest.mark.parametrize('args', [None, {}])
def test_empty_abi_bridge_preserves_exact_actual_values_and_gradients(args):
    field = Legacy(); t = torch.tensor(.3)
    x = torch.arange(20, dtype=torch.float64).reshape(4, 5).requires_grad_()
    expected = field.drift(t, x)
    expected_gradient, = torch.autograd.grad(expected.sum(), x)
    actual = empty_args_adapter(Legacy.drift)(field, t, x, args=args)
    actual_gradient, = torch.autograd.grad(actual.sum(), x)
    assert torch.equal(actual, expected) and torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize('args', [{'condition': 1}, [], (), 0, False])
def test_bridge_rejects_any_unsupported_auxiliary_scientific_input(args):
    with pytest.raises(ValueError): empty_args_adapter(Legacy.drift)(Legacy(), .3, 2., args=args)


def test_install_is_idempotent_and_does_not_modify_original_class():
    class Isolated(Legacy): pass
    untouched = Legacy.drift
    install(Isolated, 'drift'); adapter = Isolated.drift
    install(Isolated, 'drift')
    assert Isolated.drift is adapter and Legacy.drift is untouched
    x = torch.tensor(2.)
    assert torch.equal(Isolated().drift(.5, x, args={}), Legacy().drift(.5, x))
