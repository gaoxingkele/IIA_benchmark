from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import diagnose_industrial_remaining8_entropic_couplings_v1 as s


@pytest.mark.parametrize('objective,weight',[('sb',1.),('sf2m',1.),('sf2m',0.)])
def test_full_source_RNG_endpoints_match_without_network_including_partial_batch(monkeypatch,objective,weight):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    import torch
    from iia_benchmark.models import fm_objective_stable as original
    torch.set_num_threads(2)
    windows=np.random.default_rng(4).normal(size=(5,3,2)).astype(np.float32)
    parameters=dict(objective=objective,sigma=1.,window=3,hidden_size=4,epochs=2,batch_size=3,score_weight=weight)
    captured=[]
    def solve(x0,x1,epsilon):
        captured.append((hashlib.sha256(x0.numpy().tobytes()).hexdigest(),hashlib.sha256(x1.numpy().tobytes()).hexdigest(),len(x0)))
        return np.full((len(x0),len(x0)),1/len(x0)**2,dtype=np.float64)
    monkeypatch.setattr(original,'stable_plan',solve)
    model=original.StableCFMObjectiveDetector(**parameters,seed=7,device='cpu');model.fit(windows)
    # This fixture training checks omitted network/Adam operations, not performance.
    records=s.sequence(windows,parameters,7,lambda a,b,e:(np.full((len(a),len(a)),1/len(a)**2,dtype=np.float64),{}))
    assert [(r['x0_sha256'],r['x1_sha256'],r['batch_size']) for r in records]==captured
    assert len(records)==4 and [r['batch_size'] for r in records]==[3,2,3,2]
    assert not torch.cuda.is_initialized()


def test_real_grouping_keeps_all_eight_remaining_cases_and_full_training_gates():
    c=s.registration();s.verify(c)
    assert len(c['remaining_original_cases'])==8 and len(c['groups'])==3
    assert c['expected_new_numerical_sequences']==2 and c['expected_new_coupling_problems']==60
    assert c['groups']['pronto_day3__seed1107']['fit_windows']==70
    assert c['settings']['minimum_commit_headroom_bytes']==22*2**30
    assert c['settings']['emergency_commit_headroom_bytes']==16*2**30
    assert c['original_full_training_commit_admission_bytes']==32*2**30


def test_solver_failure_is_observed_once_without_resampling():
    windows=np.ones((5,3,2),dtype=np.float32);events=[];calls=[]
    def fail(x0,x1,e):calls.append(1);raise RuntimeError('preserve exact original endpoints')
    with pytest.raises(RuntimeError,match='preserve exact'):
        s.sequence(windows,dict(epochs=20,batch_size=3,sigma=1.),7,fail,lambda r,a,b:events.append(r))
    assert len(calls)==len(events)==1 and events[0]['epoch']==1 and events[0]['batch_start']==0


def test_cold_registration_does_not_import_numerical_libraries():
    code=('import sys;from scripts.flow_matching.diagnose_industrial_remaining8_entropic_couplings_v1 import registration,verify;'
        'c=registration();verify(c);assert not any(n in sys.modules for n in ["numpy","torch","scipy","pandas"]);'
        'assert c["new_neural_forwards"]==c["new_optimizer_updates"]==0')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
