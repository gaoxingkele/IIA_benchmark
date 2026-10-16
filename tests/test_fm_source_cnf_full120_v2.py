import ast
import copy
import inspect
from types import SimpleNamespace
import pytest
import torch
from scripts.flow_matching import run_source_cnf_full120_v2 as s
from scripts.flow_matching import audit_source_cnf_full120_v2 as replay
from scripts.flow_matching import run_source_cnf_full120_v1 as old


def test_all120_recipes_change_only_truthful_validation_metadata_and_isolated_output_bindings():
    before,after=old.scientific_cases(),s.scientific_cases()
    assert len(before)==len(after)==120
    for a,b in zip(before,after):
        b=copy.deepcopy(b)
        assert not b['scientific_protocol']['validation_loss_includes_leftout_timepoint']
        assert a['scientific_protocol']['validation_loss_includes_leftout_timepoint']
        b['scientific_protocol']['validation_loss_includes_leftout_timepoint']=True
        b['model_config']=a['model_config'];b['artifact_directory']=a['artifact_directory']
        assert a==b


def test_dispatch_rebinds_actual_source_and_replay_to_v2_without_changing_functions():
    for key in ['instantiate_case','worker','controller','lifecycle_worker','lifecycle_gate']:
        a,b=getattr(old,key),getattr(s,key)
        assert a.__code__ is b.__code__
        assert b.__globals__['MODULE']==s.MODULE and b.__globals__['QUEUE']==s.QUEUE
    assert replay.worker.__globals__['execution'] is s
    assert replay.CONFIG.endswith('.v2.json')


def test_actual_corrected_source_excludes_requested_endpoint_in_validation_too():
    from iia_benchmark.models import otcfm_cnf_leaveout_v1 as adapter
    from typing import Any
    text=(s.ROOT/s.original.AUTHOR/'runner/src/models/cfm_module.py').read_text()
    namespace=dict(torch=torch,Any=Any,MultivariateNormal=torch.distributions.MultivariateNormal)
    fn=adapter.compile_step(text,namespace,True)
    observed=[]
    def integrate(value,ts):
        observed.append(value[0,0].item())
        last=torch.cat([torch.zeros(len(value),1),value],dim=1)
        return ts,torch.stack([last,last])
    model=SimpleNamespace(is_trajectory=True,dim=5,hparams=SimpleNamespace(leaveout_timepoint=1),
        unpack_batch=lambda b:torch.stack(b,dim=1),aug_node=integrate,
        augmentations=lambda x:(x[:,:1],x[:,0]*0,x[:,1:]))
    fn(model,[torch.full((1024,5),float(i)) for i in range(4)],training=False)
    assert observed==[3.,2.,0.]
