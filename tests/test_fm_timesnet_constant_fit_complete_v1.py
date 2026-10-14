from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.flow_matching import audit_timesnet_constant_fit_complete_v1 as audit


def fixture():
    params=dict(window=100,d_model=128,d_ff=128,e_layers=3,top_k=3,num_kernels=6,
                dropout=.1,learning_rate=.0001,batch_size=128)
    model=dict(entrypoint='iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter',
               parameters=dict(method='timesnet',parameters=params,epochs=3,seed=1103,device='cpu'))
    job=dict(id=audit.CANONICAL,seed=1103,dataset='smap',window=100,
             parameter_overrides=dict(seed=1103,device='cpu'))
    queue=dict(jobs=[job]+[dict(id='retained_'+str(i)) for i in range(34)],evaluation=dict(cpu_threads=2))
    template=dict(settings=dict(minimum_commit_headroom_bytes=28),resource_lock='original_readonly')
    config=dict(original_scope=deepcopy(audit.SCOPE),job_id=audit.CANONICAL,cpu_threads=2,
                mode='point',**deepcopy(template))
    result=dict(job=job,id=job['id'],status='completed',parameter_count=28132633,paper_equivalence=False,
                experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest())
    guard=dict(job_id=job['id'],exit_code=0,resource_guard_aborted=False,full_registered_budget_retained=True)
    return config,queue,job,result,model,guard,template


def test_complete_frozen_original_identity_accepted():
    audit.validate_identity(*fixture())


@pytest.mark.parametrize('change',['scope','queue','seed','window','width','epochs','batch',
                                 'guard','threads','aborted','exit','capacity','fingerprint','author_claim'])
def test_shortened_or_mislabeled_full_scientific_identity_rejected(change):
    config,queue,job,result,model,guard,template=fixture()
    if change=='scope':config['original_scope']['papers']=44
    elif change=='queue':queue['jobs'].pop()
    elif change=='seed':job['seed']=1104
    elif change=='window':job['window']=50
    elif change=='width':model['parameters']['parameters']['d_model']=64
    elif change=='epochs':model['parameters']['epochs']=1
    elif change=='batch':model['parameters']['parameters']['batch_size']=64
    elif change=='guard':config['settings']['minimum_commit_headroom_bytes']=16
    elif change=='threads':config['cpu_threads']=1
    elif change=='aborted':guard['resource_guard_aborted']=True
    elif change=='exit':guard['exit_code']=1
    elif change=='capacity':result['parameter_count']-=1
    elif change=='fingerprint':result['experiment_sha256']='different'
    elif change=='author_claim':result['paper_equivalence']=True
    with pytest.raises(ValueError):
        audit.validate_identity(config,queue,job,result,model,guard,template)


@pytest.mark.parametrize('observed',[np.array([1.,2.001]),np.array([1.]),np.array([1.,np.nan])])
def test_changed_incomplete_or_nonfinite_stream_rejected(observed):
    with pytest.raises(ValueError):audit.require_exact_scores(observed,np.array([1.,2.]))


def test_exact_stream_accepted():
    audit.require_exact_scores(np.array([1.,2.]),np.array([1.,2.]))


def test_actual_tiny_checkpoint_restores_all_scores_without_fitting():
    import torch
    from iia_benchmark.models.mtsad_detectors import TimesNetDetector
    torch.set_num_threads(2)
    params=dict(window=10,d_model=4,d_ff=4,e_layers=1,top_k=2,num_kernels=2,
                dropout=.1,learning_rate=.0001,batch_size=2)
    config=dict(parameters=dict(parameters=params,epochs=3))
    job=dict(seed=1103,window=10,parameter_overrides=dict(seed=1103,device='cpu'))
    detector=TimesNetDetector(**params,epochs=3,seed=1103,device='cpu')
    detector._device=torch.device('cpu');detector._model=detector._build(2)
    checkpoint=dict(configuration=config,overrides=job['parameter_overrides'],window=10,
                    adjacency=None,modules={'_model':detector._model.state_dict()})
    count=sum(p.numel() for p in detector._model.parameters())
    restored=audit.restore_model(checkpoint,job,config,2,count)
    windows=np.random.default_rng(1).normal(size=(3,10,2))
    audit.require_exact_scores(restored.score(windows),detector.score(windows))
    checkpoint['modules']['_model']={k:v.double() if v.is_floating_point() else v for k,v in checkpoint['modules']['_model'].items()}
    with pytest.raises(ValueError):audit.restore_model(checkpoint,job,config,2,count)
