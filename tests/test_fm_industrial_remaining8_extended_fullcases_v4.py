"""Small training fixtures test fidelity only; no benchmark performance credit."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import continue_industrial_remaining8_extended_fullcases_v4 as s
from scripts.flow_matching import industrial_extended_fullcase_common_v4 as common
from scripts.flow_matching.timegan_native_common_v1 import read


def jobs():
    old=dict(id='case',dataset='pronto',seed=1104,window=100,model_config='old',output_directory='old',
        parameter_overrides=dict(seed=1104,device='cpu'),frozen_files=[dict(path='source',sha256='frozen')])
    new=dict(deepcopy(old),id='case__transport_extension_v4',model_config='new',output_directory=s.STATE+'/jobs/case__transport_extension_v4/source')
    a=dict(parameters=dict(epochs=20,sigma=1.,score_weight=1.,batch_size=64))
    b=dict(parameters=dict(a['parameters'],sinkhorn_max_final_iterations=200000),entrypoint='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector')
    return old,new,a,b


def test_only_minimal_numeric_cap_identity_and_output_change_is_allowed():s.validate_job(*jobs())


@pytest.mark.parametrize('fault',['seed','window','data','epoch','sigma','score_weight','cap','pin','output'])
def test_original_all_fields_and_objective_parameters_must_be_preserved(fault):
    a,b,c,d=jobs()
    if fault=='seed':b['seed']=1
    elif fault=='window':b['window']=50
    elif fault=='data':b['dataset']='other'
    elif fault=='epoch':d['parameters']['epochs']=10
    elif fault=='sigma':d['parameters']['sigma']=2.
    elif fault=='score_weight':d['parameters']['score_weight']=0.
    elif fault=='cap':d['parameters']['sinkhorn_max_final_iterations']=20000
    elif fault=='pin':b['frozen_files']=[]
    else:b['output_directory']='original'
    with pytest.raises(ValueError):s.validate_job(a,b,c,d)


def fitted_fixture(objective,weight):
    import torch
    from iia_benchmark.models.fm_objective_stable_extended import ExtendedStableCFMObjectiveDetector,extended_plan
    from scripts.flow_matching.diagnose_industrial_remaining8_entropic_couplings_v1 import sequence
    torch.set_num_threads(2)
    p=dict(objective=objective,sigma=1.,window=3,hidden_size=4,epochs=2,batch_size=3,score_weight=weight,
        score_times=[.5],monte_carlo_samples=1,sinkhorn_max_final_iterations=200000)
    x=np.random.default_rng(8).normal(size=(5,3,2)).astype(np.float32)
    def solve(a,b,e):
        audit=[];plan=extended_plan(a,b,e,audit=audit);return plan,audit[-1]
    expected=sequence(x,p,7,solve)
    reference=ExtendedStableCFMObjectiveDetector(**p,seed=7,device='cpu');reference.fit(x)
    return x,p,expected,reference


@pytest.mark.parametrize('objective,weight',[('sb',1.),('sf2m',1.),('sf2m',0.)])
def test_private_source_fit_real_Adam_counter_and_restore_are_exact(monkeypatch,objective,weight):
    import torch
    from iia_benchmark.models.fm_objective_stable_extended import ExtendedStableCFMObjectiveDetector
    x,p,expected,reference=fitted_fixture(objective,weight)
    model=ExtendedStableCFMObjectiveDetector(**p,seed=7,device='cpu');audit=common.instrument(model,expected);model.fit(x)
    assert audit['optimizer_updates']==4 and audit['optimizer_instances']==1 and len(audit['transport_problems'])==4
    assert model.training_losses_==reference.training_losses_
    assert all(torch.equal(value,reference._model.state_dict()[key]) for key,value in model._model.state_dict().items())
    assert np.array_equal(model.score(x),reference.score(x))
    configuration=dict(entrypoint='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector',parameters=p)
    overrides=dict(seed=7,device='cpu');checkpoint=dict(configuration=configuration,overrides=overrides,window=3,modules={'_model':model._model.state_dict()})
    monkeypatch.setattr(ExtendedStableCFMObjectiveDetector,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Replay must never fit')))
    restored=common.restore(checkpoint,configuration,overrides,(3,2))
    assert np.array_equal(restored.score(x),reference.score(x)) and not restored.training_losses_


@pytest.mark.parametrize('fault',['x0_sha256','coupling_sha256','pair_indices'])
def test_actual_full_training_must_match_qualified_transport_and_sampled_pairs(fault):
    from iia_benchmark.models.fm_objective_stable_extended import ExtendedStableCFMObjectiveDetector
    x,p,expected,_=fitted_fixture('sb',1.)
    if fault=='pair_indices':expected[0][fault]=[-1]*3
    else:expected[0][fault]='wrong'
    model=ExtendedStableCFMObjectiveDetector(**p,seed=7,device='cpu');audit=common.instrument(model,expected)
    with pytest.raises(ValueError):model.fit(x)
    assert audit['optimizer_updates']==0


def idle_fixture():
    owner=deepcopy(s.EXPECTED);first=read(s.ROOT/s.QUEUE)['jobs'][0]['job']['id']
    state=dict(pid=owner['pid'],active_pid=None,active_job=None,shared_lock_owned=False,state='waiting_for_memory',jobs={first:'partial_preserved'})
    locks=[str((s.ROOT/p).resolve()).lower() for p in [s.OLD_CONTROLLER_LOCK,s.OLD_WORKER_LOCK]]
    children=[dict(name='conhost.exe',exe='C:/Windows/System32/conhost.exe')]
    return owner,deepcopy(owner),state,children,locks


def test_only_exact_idle_metadata_identity_and_own_locks_are_eligible():assert s.idle_allowed(*idle_fixture())


@pytest.mark.parametrize('fault',['pid','create_time','command','active','child','shared_lock','jobs','state','missing_lock','owned_shared'])
def test_handoff_rejects_reused_identity_active_workers_and_changed_ownership(fault):
    expected,identity,state,children,locks=idle_fixture()
    if fault=='pid':identity['pid']+=1
    elif fault=='create_time':identity['create_time']+=1
    elif fault=='command':identity['command']=['other']
    elif fault=='active':state['active_pid']=123
    elif fault=='child':children.append(dict(name='python.exe',exe='C:/python.exe'))
    elif fault=='shared_lock':locks.append(str((s.ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock').resolve()).lower())
    elif fault=='jobs':state['jobs']['another_started_case']='running'
    elif fault=='state':state['state']='running'
    elif fault=='missing_lock':locks.pop()
    else:state['shared_lock_owned']=True
    assert not s.idle_allowed(expected,identity,state,children,locks)


@pytest.mark.parametrize('race',[False,True])
def test_atomic_metadata_handoff_rechecks_after_pause_and_resumes_if_child_raced(tmp_path,monkeypatch,race):
    monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/s.STATE).mkdir(parents=True);calls=[];observations=[]
    class Fake:
        def suspend(self):calls.append('suspend')
        def resume(self):calls.append('resume')
        def terminate(self):calls.append('terminate')
        def wait(self,timeout):calls.append('wait')
    fake=Fake();monkeypatch.setattr(s.psutil,'Process',lambda pid:fake);monkeypatch.setattr(s.time,'sleep',lambda _:None)
    def snapshot(c):
        observations.append(1)
        if race and len(observations)==4:raise ValueError('numeric child appeared')
        return dict(identity=c['previous_owner'],preserved_legacy_failure_sha256={'failure':'unchanged'}),b'{}'
    monkeypatch.setattr(s,'idle_snapshot',snapshot);monkeypatch.setattr(s,'legacy_layout',lambda:{'failure':'unchanged'})
    c=dict(previous_owner=s.EXPECTED)
    if race:
        with pytest.raises(ValueError):s.handoff(c)
        assert calls==['suspend','resume'] and not (tmp_path/s.STATE/'handoff.json').exists()
    else:
        s.handoff(c);assert calls==['suspend','terminate','wait']
        assert read(tmp_path/s.STATE/'handoff.json')['numerical_workers_stopped'] is False


def test_full8_registration_keeps_original32_GiB_and_excludes_delegated_first():
    c=s.registration();s.verify(c)
    assert len(c['cases'])==8 and c['settings']['minimum_commit_headroom_bytes']==32*2**30 and c['settings']['emergency_commit_headroom_bytes']==16*2**30
    assert sorted(x['optimizer_updates'] for x in c['cases'])==[20]*5+[40]*3
    assert not any(x['canonical_original_id']==c['delegated_first_canonical_id'] for x in c['cases'])


def test_metadata_registration_and_probe_import_no_neural_libraries():
    code=('import sys;from scripts.flow_matching.continue_industrial_remaining8_extended_fullcases_v4 import registration,verify;'
        'c=registration();verify(c);assert len(c["cases"])==8;'
        'assert not any(n in sys.modules for n in ["numpy","torch","scipy","sklearn","pandas"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
