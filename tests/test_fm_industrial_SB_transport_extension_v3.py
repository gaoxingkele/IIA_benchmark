from copy import deepcopy
import pytest
from scripts.flow_matching.continue_industrial_SB_transport_extension_v3 import validate_case,STATE,MODEL,registration


def fixtures():
    old=dict(id='case',output_directory='old',model_config='oldmodel',dataset='pronto',seed=1103,window=100,
        parameter_overrides=dict(device='cpu',seed=1103),frozen_files=[dict(path='raw',sha256='pin')])
    new=dict(deepcopy(old),id='case__transport_extension_v3',output_directory=STATE+'/full_job',model_config=MODEL)
    a=dict(parameters=dict(epochs=20,batch_size=64,sigma=1.,score_times=[.1,.3,.5,.7,.9]))
    b=dict(parameters=dict(a['parameters'],sinkhorn_max_final_iterations=200000),entrypoint='iia_benchmark.models.fm_objective_stable_extended:ExtendedStableCFMObjectiveDetector')
    return old,new,a,b


def test_only_explicit_solver_extension_and_output_identity_are_permitted():
    validate_case(*fixtures())


@pytest.mark.parametrize('fault',['seed','window','data','epoch','score_grid','caps','old_pin'])
def test_full_original_budget_data_seed_and_scoring_may_not_change(fault):
    old,new,a,b=fixtures()
    if fault=='seed':new['seed']=1104
    elif fault=='window':new['window']=50
    elif fault=='data':new['dataset']='other'
    elif fault=='epoch':b['parameters']['epochs']=19
    elif fault=='score_grid':b['parameters']['score_times']=[.5]
    elif fault=='caps':b['parameters']['sinkhorn_max_final_iterations']=20000
    else:new['frozen_files']=[]
    with pytest.raises(ValueError):validate_case(old,new,a,b)


def test_mathematical_diagnostic_and_full_training_admission_are_distinct_and_explicit():
    c=registration()
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert c['settings']['emergency_commit_headroom_bytes']==16*2**30
    assert c['diagnostic_settings']['minimum_commit_headroom_bytes']==22*2**30
    assert c['diagnostic_is_coupling_only_no_optimizer_or_neural_forward']
    assert c['original_full_industrial_scope_CPU']==550 and c['original_full_industrial_scope_GPU']==850


def test_controller_source_registration_and_verification_do_not_import_neural_libraries():
    import subprocess,sys
    from pathlib import Path
    code='import sys;from scripts.flow_matching.continue_industrial_SB_transport_extension_v3 import registration,verify;verify(registration());assert not any(x in sys.modules for x in ["torch","numpy","pandas"])'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
