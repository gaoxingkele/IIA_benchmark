import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching import run_ls4_cauchy_corrected_v3 as worker
from scripts.flow_matching import run_ls4_cauchy_corrected_queue_v3 as controller
from scripts.flow_matching.ls4_cauchy_fallback_repair_v1 import official_function,digest
from scripts.flow_matching.transition_ls4_cauchy_idle_v3 import idle


def test_official_cauchy_contains_both_conjugate_terms_and_gradients():
    import torch
    source=Path('experiments/runs/fm_spectral_table3_baseline_acquisition_v1/sources/sashimi_s4/original/src/models/functional/cauchy.py')
    namespace=dict(torch=torch);exec(compile(official_function(source.read_text()),str(source),'exec'),namespace)
    v=torch.tensor([2+1j],dtype=torch.complex128,requires_grad=True)
    w=torch.tensor([-.3+2j],dtype=torch.complex128)
    z=torch.tensor([.1+1j,.7-.5j],dtype=torch.complex128)
    expected=v[0]/(z-w[0])+v[0].conj()/(z-w[0].conj())
    actual=namespace['cauchy_naive'](v,z,w)
    assert torch.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert not torch.allclose(actual,v[0]/(z-w[0]))
    actual.abs().sum().backward();assert torch.isfinite(v.grad).all()


def test_official_cauchy_boundary_change_rejected():
    with pytest.raises(ValueError,match='boundary'):
        official_function('def cauchy_naive(v,z,w,conj=True):\n    return v\n')


def test_correction_preserves_canonical_full_budgets_and_rejects_reduction(tmp_path,monkeypatch):
    model=dict(dataset='fred_md',protocol='paper_literal',seed=42,samples=107,length=728,
        train_samples=85,test_samples=22,optim=dict(epochs=7000,batch_size=64),model=dict(d_model=64),
        sigma=.1,source_config='original.yaml',budget=dict(generator_updates=14000))
    old=dict(jobs=[dict(id='job',model_config='old_model.json',output_directory='old')],settings=dict(gate=42))
    (tmp_path/'old_model.json').write_text(json.dumps(model))
    (tmp_path/'old_queue.json').write_text(json.dumps(old))
    (tmp_path/'new_model.json').write_text(json.dumps(dict(model,author_equivalence_certified=False)))
    new=dict(original_canonical_queue='old_queue.json',original_canonical_queue_sha256=digest(tmp_path/'old_queue.json'),
        jobs=[dict(id='job',model_config='new_model.json',original_output_directory='old')],settings=old['settings'])
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'worker',lambda queue:SimpleNamespace(verify=lambda q:None))
    worker.verify(new)
    reduced=dict(model,author_equivalence_certified=False);reduced['optim']=dict(epochs=1,batch_size=64)
    (tmp_path/'new_model.json').write_text(json.dumps(reduced))
    with pytest.raises(ValueError,match='full budgets'):worker.verify(new)


def test_uncorrected_full_data_proof_is_not_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    q=tmp_path/'queue.json';q.write_text('{}')
    proof=dict(passed=True,diagnostic_only=True,cases=20,queue_sha256=digest(q))
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    queue=dict(queue_path='queue.json',data_audit_output='proof.json')
    with pytest.raises(ValueError,match='proof differs'):controller.proof_ready(queue,'data_audit')
    proof['backend_correction']=dict(algorithmic_correction=True);path.write_text(json.dumps(proof))
    assert controller.proof_ready(queue,'data_audit')


def test_handoff_rejects_running_model_and_reused_pid():
    class Parent:
        pid=42
        def create_time(self):return 123.
        def cmdline(self):return ['python','old_controller']
        def children(self,recursive=True):return []
    parent=Parent();expected=dict(pid=42,create_time=123.,command=parent.cmdline())
    state=dict(pid=42,state='waiting_full_gpu_resources',active_pid=None)
    assert idle(parent,state,expected)
    assert not idle(parent,dict(state,active_pid=99),expected,pid_exists=lambda pid:True)
    assert not idle(parent,state,dict(expected,create_time=122.))
    assert not idle(parent,dict(state,state='running_full_gpu_pipeline'),expected)


def test_extended_full_gpu_proof_requires_both_dataset_protocol_pairs(tmp_path,monkeypatch):
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    q=tmp_path/'queue.json';q.write_text('{}')
    cases=[dict(dataset=d,protocol=p,original_classifier_epochs=100,original_predictor_epochs=100,
                complete_metric_interface_passed=True)
           for d in ('solar_weekly','temperature_rain') for p in ('released','paper_literal')]
    proof=dict(passed=True,diagnostic_only=True,cases=cases,queue_sha256=digest(q),
               backend_correction=dict(algorithmic_correction=True))
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    queue=dict(queue_path='queue.json',gpu_preflight_output='proof.json',
               datasets=[dict(dataset='solar_weekly'),dict(dataset='temperature_rain')])
    assert controller.proof_ready(queue,'gpu_preflight')
    proof['cases']=cases[:-1];path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='All4'):controller.proof_ready(queue,'gpu_preflight')


@pytest.mark.parametrize('queue_path',['configs/experiments/fm_ls4_original_queue.v1.json',
    'configs/experiments/fm_ls4_monash_extended_queue.v2.json'])
def test_real_original_model_schemas_and_all_twenty_full_slots(tmp_path,monkeypatch,queue_path):
    root=Path.cwd();old=json.loads((root/queue_path).read_text())
    (tmp_path/'old_queue.json').write_text(json.dumps(old))
    new=copy.deepcopy(old);new.update(original_canonical_queue='old_queue.json',
        original_canonical_queue_sha256=digest(tmp_path/'old_queue.json'))
    for index,(a,b) in enumerate(zip(old['jobs'],new['jobs'])):
        original=json.loads((root/a['model_config']).read_text())
        old_path=tmp_path/a['model_config'];old_path.parent.mkdir(parents=True,exist_ok=True)
        old_path.write_text(json.dumps(original))
        model=tmp_path/('corrected_%d.json'%index)
        model.write_text(json.dumps(dict(original,author_equivalence_certified=False,
            implementation='corrected',reproduction_status='registered')))
        b.update(model_config=model.name,original_output_directory=a['output_directory'])
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'worker',lambda queue:SimpleNamespace(verify=lambda q:None))
    worker.verify(new)
    model=tmp_path/'corrected_0.json';value=json.loads(model.read_text())
    value['source_config']='different_architecture.yaml';model.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='full budgets'):worker.verify(new)
