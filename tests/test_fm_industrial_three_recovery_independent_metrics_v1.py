from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import audit_industrial_three_recovery_independent_metrics_v1 as s


def fixture():
    old=dict(id='case',dataset='skab',seed=1106,window=60,model_config='model',output_directory='original',
        parameter_overrides=dict(seed=1106,device='cpu'),frozen_files=[dict(path='source',sha256='frozen')])
    new=dict(deepcopy(old),id='case__resource_recovery_v2',output_directory='separate')
    return dict(kind='industrial',original_id='case',original_job=deepcopy(old),job=new),dict(case=old)


def test_only_documented_identity_and_output_mapping_is_allowed():
    case,jobs=fixture();assert s.validate_identity(case,jobs)==case['job']


@pytest.mark.parametrize('fault',['seed','window','model','device','frozen_files','original','identity','same_output'])
def test_never_substitute_altered_data_budget_model_or_seed(fault):
    case,jobs=fixture();job=case['job']
    if fault=='seed':job['seed']=1107
    elif fault=='window':job['window']=100
    elif fault=='model':job['model_config']='other'
    elif fault=='device':job['parameter_overrides']['device']='cuda'
    elif fault=='frozen_files':job['frozen_files']=[]
    elif fault=='original':case['original_job']['dataset']='other'
    elif fault=='identity':job['id']='alias_without_bound_identity'
    else:job['output_directory']='original'
    with pytest.raises(ValueError):s.validate_identity(case,jobs)


def test_cold_registration_keeps_full_scope_and_imports_no_numerical_libraries():
    code=('import sys;from scripts.flow_matching.audit_industrial_three_recovery_independent_metrics_v1 import registration,verify;'
        'c=registration();o,q,m,cases=verify(c);assert len(cases)==3 and len(q["jobs"])==550;'
        'assert c["settings"]["minimum_commit_headroom_bytes"]==22*2**30;'
        'assert c["source_historical_current_gate_certification"] is False;'
        'assert not any(n in sys.modules for n in ["numpy","torch","scipy","sklearn","pandas"])')
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
