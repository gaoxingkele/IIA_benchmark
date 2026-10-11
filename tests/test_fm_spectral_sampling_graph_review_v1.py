import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from iia_benchmark.models.fm_spectral_sampling_graph_review_v1 import reviewed_forward,sampling_first_derivative_graph


class AnalyticNetwork(torch.nn.Module):
    def forward(self,x):
        witness=(x.sin()+x*x).sum()
        (grad,)=torch.autograd.grad(witness,x,create_graph=True)
        return grad


def test_only_higher_order_graph_keyword_changes_and_first_derivative_values_are_identical():
    original=AnalyticNetwork.forward
    changed,receipt=reviewed_forward(original)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg=='create_graph': node.value=ast.Constant(value=False)
    import hashlib
    assert receipt['reviewed_AST_sha256']==hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    net=AnalyticNetwork()
    x=torch.tensor([-.3,0.,.7],requires_grad=True)
    first=net(x)
    second=changed(net,x)
    assert torch.equal(first,second) and first.requires_grad and not second.requires_grad


def test_training_is_rejected_and_original_callable_restored_even_after_sampling_failure():
    model=SimpleNamespace(net=AnalyticNetwork())
    original=model.net.forward.__func__
    with pytest.raises(ValueError):
        with sampling_first_derivative_graph(model): pass
    with pytest.raises(RuntimeError):
        with torch.no_grad(),sampling_first_derivative_graph(model):
            assert model.net.forward.__func__ is not original
            raise RuntimeError('native sampling failed')
    assert model.net.forward.__func__ is original and 'forward' not in model.net.__dict__


def test_preexisting_forward_or_non_native_gradient_form_is_not_overwritten():
    model=SimpleNamespace(net=AnalyticNetwork())
    model.net.forward=lambda x:x
    with torch.no_grad(),pytest.raises(ValueError):
        with sampling_first_derivative_graph(model): pass
    assert 'forward' in model.net.__dict__
    with pytest.raises(ValueError): reviewed_forward(test_preexisting_forward_or_non_native_gradient_form_is_not_overwritten)
