import copy
import itertools

import pytest

from scripts.flow_matching.run_ls4_native_ushcn_v1 import ObservedLoader, budget, periodic_epochs, verify_job_scope
from scripts.flow_matching import run_ls4_native_ushcn_queue_v1 as queue_runner


def config():
    return dict(epochs=500,batch_size=64,variables=5,train_series=1001,test_series=112)


def scope():
    q=dict(split_branches=['source_literal','id_corrected'],folds=list(range(5)),
           profiles=['interpolation','extrapolation'],seeds=[42,1103,1104,1105,1106])
    q['jobs']=[dict(id=str(i),split_branch=b,fold=f,profile=p,seed=s)
               for i,(b,f,p,s) in enumerate(itertools.product(q['split_branches'],q['folds'],q['profiles'],q['seeds']))]
    return q


def test_full_500_epoch_budget_and_native_evaluation_schedule():
    assert budget(config())==dict(train_batches=16,test_batches=2,optimizer_updates=8000,evaluation_calls=50)
    assert periodic_epochs(500,10)==list(range(10,500,10))
    assert len(periodic_epochs(500,10)+[-1])==50


@pytest.mark.parametrize('field,value',[('epochs',1),('batch_size',32),('variables',4),('train_series',0)])
def test_smaller_scientific_scope_is_rejected(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):budget(c)


def test_entire_branch_fold_profile_seed_scope_required():
    q=scope();verify_job_scope(q)
    q['jobs'][0]['fold']=8
    with pytest.raises(ValueError,match='Cartesian'):verify_job_scope(q)
    q=scope();q['jobs'].pop()
    with pytest.raises(ValueError,match='All100'):verify_job_scope(q)


class Loader:
    batch_size=64
    def __init__(self,counts):self.counts=counts
    def __len__(self):return len(self.counts)
    def __iter__(self):
        for n in self.counts:yield dict(observed_data=[0]*n)


def test_complete_tail_series_required():
    observed=ObservedLoader(Loader([64,48]),112,lambda *a:None)
    assert len(list(observed))==2 and observed.calls==1
    with pytest.raises(ValueError,match='coverage'):
        list(ObservedLoader(Loader([64,32]),112,lambda *a:None))


def test_partial_training_or_evaluation_cannot_publish(monkeypatch):
    c=config();job=dict(output_directory='fake',model_config='model')
    result=dict(job=job,queue_sha256='x',config_sha256='x',epochs_completed=500,optimizer_updates=8000,
        evaluations=[dict(epoch=e,full_test_series=112,full_test_batches=2) for e in periodic_epochs(500,10)+[-1]],
        author_equivalence_certified=False,artifacts=[])
    monkeypatch.setattr(queue_runner,'sha',lambda p:'x')
    monkeypatch.setattr(queue_runner,'read',lambda p:c if str(p).endswith('model') else result)
    queue=dict(queue_path='queue')
    assert queue_runner.verify_result(queue,job)==result
    result['optimizer_updates']=7999
    with pytest.raises(ValueError,match='scope'):queue_runner.verify_result(queue,job)
    result['optimizer_updates']=8000;result['evaluations'][-1]['full_test_series']=111
    with pytest.raises(ValueError,match='coverage'):queue_runner.verify_result(queue,job)
