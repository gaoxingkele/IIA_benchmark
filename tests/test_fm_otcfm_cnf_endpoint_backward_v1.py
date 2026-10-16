"""Actual original step semantics on a differentiable fixture,not ODE performance."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import torch
from iia_benchmark.models import otcfm_cnf_endpoint_backward_v1 as s
from iia_benchmark.models import otcfm_cnf_leaveout_v1 as original

ROOT=Path(__file__).resolve().parents[1]
TEXT=(ROOT/'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/runner/src/models/cfm_module.py').read_text()


@pytest.mark.parametrize('timepoints,left',[(4,1),(4,2),(5,1),(5,2),(5,3)])
def test_all_original_1024batch_endpoint_objectives_and_gradients_match(timepoints,left):
    def compute(streamed):
        torch.manual_seed(42)
        matrix=(torch.randn(5,5,dtype=torch.float64)*.03).requires_grad_()
        offset=torch.randn(5,dtype=torch.float64,requires_grad=True)
        batch=[torch.randn(1024,5,dtype=torch.float64,requires_grad=True) for _ in range(timepoints)]
        def integrate(value,ts):
            span=ts[0]-ts[-1]
            last=torch.cat([torch.trace(matrix).expand(len(value),1)*span,
                            torch.sin(value@matrix.T*span+offset)],dim=1)
            return ts,torch.stack([last,last])
        model=SimpleNamespace(is_trajectory=True,dim=5,hparams=SimpleNamespace(leaveout_timepoint=left),
            unpack_batch=lambda b:torch.stack(b,dim=1),aug_node=integrate,
            augmentations=lambda v:(v[:,:1],v[:,0]*0,v[:,1:]))
        tree=s.endpoint_step_tree(TEXT,original.step_tree) if streamed else original.step_tree(TEXT,True)
        namespace=dict(torch=torch,Any=Any,MultivariateNormal=torch.distributions.MultivariateNormal)
        exec(compile(tree,'<original_CNF_fixture>','exec'),namespace)
        from types import MethodType
        model.step=MethodType(namespace['step'],model)
        if streamed:
            value,proof=s.backward_all_endpoints(model,batch,left)
            assert proof['complete_objective_terms']==timepoints-1 and proof['optimizer_updates']==0
        else:
            reg,loss=model.step(batch,training=True);value=reg+loss;value.backward()
        return value.detach(),matrix.grad,offset.grad,[x.grad for x in batch]
    joint,streamed=compute(False),compute(True)
    for a,b in zip(joint[:3],streamed[:3]):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    for index,(a,b) in enumerate(zip(joint[3],streamed[3])):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
        if index==left:assert torch.count_nonzero(b)==0
        else:assert torch.count_nonzero(b)>0


def test_selector_ast_preserves_every_original_likelihood_equation():
    old=original.step_tree(TEXT,True);new=s.endpoint_step_tree(TEXT,original.step_tree)
    fn=new.body[0];assert fn.args.kwonlyargs.pop().arg=='endpoint_index';fn.args.kw_defaults.pop()
    loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For))
    removed=loop.body.pop(1)
    assert ast.unparse(removed.test)=='endpoint_index is not None and len(even_ts) - 1 - t != endpoint_index'
    assert ast.dump(old)==ast.dump(new)
