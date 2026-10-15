"""Independent analytic and corrupted-state tests;not benchmark performance."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from iia_benchmark.models.fm_native import TimeMLP,GraphSpectralPath,MixerVelocity,normalized_laplacian
from iia_benchmark.models.fm_comparator_reconstructions import KGLCore
from scripts.flow_matching import neural_full510_checkpoint_state_v1 as state
from scripts.flow_matching.dfm_checkpoint_density_math_v1 import analytic_field


@pytest.mark.parametrize('trace',['exact','hutchinson'])
def test_analytic_Tanh_MLP_trace_matches_independent_full_Jacobian(trace):
    torch.manual_seed(17);model=TimeMLP(3,8).double()
    x=torch.randn(4,3,dtype=torch.float64,requires_grad=True);time=torch.tensor([.1,.3,.7,1.],dtype=torch.float64)
    actual=model(time,x);eps=torch.tensor([[1.,-1.,1.]]*4,dtype=torch.float64)
    if trace=='exact':
        wanted=sum(torch.autograd.grad(actual[:,j].sum(),x,retain_graph=True)[0][:,j] for j in range(3))
    else:wanted=(torch.autograd.grad((actual*eps).sum(),x)[0]*eps).sum(-1)
    value,divergence=analytic_field(model,x.detach().numpy(),time.numpy(),None if trace=='exact' else eps.numpy())
    assert np.allclose(value,actual.detach().numpy(),atol=1e-12,rtol=1e-12)
    assert np.allclose(divergence,wanted.detach().numpy(),atol=1e-12,rtol=1e-12)


def checkpoint():
    config=dict(entrypoint='iia_benchmark.models.fm_comparator_reconstructions:ComparatorWindowDetector',
        parameters=dict(method='kgl',options=dict(d_model=64,heads=4,use_gat=True,use_kan=True,use_lstm=True),epochs=10,batch_size=64,
            learning_rate=.001,seed=1103,device='cpu'))
    model=KGLCore(3,**config['parameters']['options'])
    job=dict(window=100,seed=1103,parameter_overrides=dict(seed=1103,device='cpu'))
    cp=dict(configuration=config,overrides=job['parameter_overrides'],window=100,adjacency=None,modules={'_model':model.state_dict()})
    return cp,job,config


@pytest.mark.parametrize('fault',['module','key','dtype','shape','nonfinite','device','window','adjacency'])
def test_complete_original_checkpoint_schema_cannot_be_silently_changed(fault):
    cp,job,config=checkpoint();values=cp['modules']['_model'];key=next(iter(values))
    if fault=='module':cp['modules']['extra']={}
    elif fault=='key':values.pop(key)
    elif fault=='dtype':values[key]=values[key].double()
    elif fault=='shape':values[key]=values[key][:-1]
    elif fault=='nonfinite':values[key].reshape(-1)[0]=float('nan')
    elif fault=='device':cp['overrides']=dict(seed=1103,device='cuda:0')
    elif fault=='window':cp['window']=50
    else:cp['adjacency']=np.eye(3,dtype=np.float32)
    with pytest.raises(ValueError):state.restore_model(cp,job,config,3)


def test_original_parameter_flags_and_all_buffers_restore_exactly():
    cp,job,config=checkpoint();restored=state.restore_model(cp,job,config,3)
    assert state.check_state(restored,cp)
    assert all(p.requires_grad for p in restored._model.parameters())
    next(restored._model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError):state.check_state(restored,cp)


def test_safe_weights_only_numpy_adjacency_load_and_actual_tied_basis_retained(tmp_path):
    a=np.ones((3,3),dtype=np.float32)-np.eye(3,dtype=np.float32)
    path=GraphSpectralPath(normalized_laplacian(a),2.)
    # Sign changes are equally valid eigenbases;actual checkpoint bytes matter.
    path.basis[:,1:]*=-1
    model=MixerVelocity(3,50,128,2,.1)
    config=dict(entrypoint='iia_benchmark.models.fm_native:GRASPDetector',parameters=dict(window=50,hidden=128,blocks=2,
        dropout=.1,tau=2.,graph_threshold=.5,variant='grasp',epochs=1500,batch_size=256,learning_rate=.001,
        source_samples=5,flow_evaluations=10,seed=1103,device='auto'))
    job=dict(window=50,seed=1103,parameter_overrides=dict(seed=1103,device='cpu'))
    cp=dict(configuration=config,overrides=job['parameter_overrides'],window=50,adjacency=a,modules=dict(path=path.state_dict(),model=model.state_dict()))
    target=tmp_path/'own_source_checkpoint.pt';torch.save(cp,target)
    loaded=state.load_checkpoint(target);restored=state.restore_model(loaded,job,config,3)
    assert np.array_equal(restored.adjacency_,a)
    assert torch.equal(restored.path.basis,path.basis)
    assert set(state.check_state(restored,loaded))=={n+'.'+k for n,values in cp['modules'].items() for k in values}
