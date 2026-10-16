import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import torch
from iia_benchmark.models import otcfm_cnf_leaveout_v1 as s

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/runs/flow_matching_campaign/sources/ot_cfm/original/runner/src/models/cfm_module.py'


@pytest.mark.parametrize('timepoints,left',[(4,1),(4,2),(5,1),(5,2),(5,3)])
def test_actual_original_objective_endpoint_gradient_exposes_and_corrects_reverse_index(timepoints,left):
    text=SOURCE.read_text(encoding='utf-8')
    def objective(correct):
        x=torch.arange(1,timepoints+1,dtype=torch.float32).reshape(1,timepoints,1).expand(3,timepoints,5).clone().requires_grad_()
        seen=[]
        def integrate(value,ts):
            seen.append(int(value[0,0].item())-1)
            padded=torch.cat([value[:,:2]*0,value],dim=1)
            return ts,torch.stack([padded,padded])
        model=SimpleNamespace(is_trajectory=True,dim=5,hparams=SimpleNamespace(leaveout_timepoint=left),
            unpack_batch=lambda batch:batch,aug_node=integrate,
            augmentations=lambda value:(value[:,0],value[:,1],value[:,2:]))
        fn=s.compile_step(text,dict(torch=torch,Any=Any,MultivariateNormal=torch.distributions.MultivariateNormal),correct)
        reg,loss=fn(model,x,training=True);(reg+loss).backward()
        omitted=timepoints-1-left if not correct else left
        assert seen==[i for i in reversed(range(timepoints)) if i!=omitted]
        assert torch.equal(x.grad[:,omitted,:],torch.zeros_like(x.grad[:,omitted,:]))
        assert all(torch.count_nonzero(x.grad[:,i,:]).item()>0 for i in range(timepoints) if i!=omitted)
        return seen
    before,after=objective(False),objective(True)
    assert left not in after
    assert (left in before)==(timepoints-1-left!=left)


def test_ast_changes_only_one_trajectory_endpoint_comparator():
    text=SOURCE.read_text(encoding='utf-8');old=s.step_tree(text,False);new=s.step_tree(text,True)
    class Reverse(ast.NodeTransformer):
        def visit_IfExp(self,n):
            if ast.dump(n)==ast.dump(ast.parse('(len(even_ts)-1-t) if self.is_trajectory else t',mode='eval').body):return ast.Name(id='t',ctx=ast.Load())
            return self.generic_visit(n)
    assert ast.dump(old)==ast.dump(Reverse().visit(new))


def test_complete_four_and_five_timepoint_mapping_is_not_a_paper_result_claim():
    four=s.endpoint_mapping(4);five=s.endpoint_mapping(5)
    assert [r['original_omitted_observation_index'] for r in four]==[2,1]
    assert [r['corrected_omitted_observation_index'] for r in four]==[1,2]
    assert [r['original_objective_includes_requested_leftout'] for r in five]==[True,False,True]
    with pytest.raises(ValueError):s.endpoint_mapping(3)
