import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest
from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v2 as s


def test_all_original_scientific_settings_and_five_cases_unchanged():
    c=s.registration();s.verify(c);old=s.original.registration()
    for key in ['jobs','settings','resource_lock','environment','source_evaluation','label_policies','python']:
        assert c[key]==old[key]
    assert not c['numerical_worker_body_changed'] and c['schema_version']==2


@pytest.mark.parametrize('field',['jobs','settings','numerical_worker_body_changed','expected_pre_worker_failure'])
def test_no_reduced_cohort_resources_or_other_failure_recovery(field):
    c=s.registration()
    if field=='jobs':c[field].pop()
    elif field=='settings':c[field]['minimum_commit_headroom_bytes']=1
    elif field=='numerical_worker_body_changed':c[field]=True
    else:c[field]='other failure'
    with pytest.raises(ValueError):s.verify(c)


def test_numeric_worker_mapping_and_metrics_bytecode_are_identical():
    private=s.private_module()
    for name in ['worker','mapped_labels','protocol_diagnostics']:
        before=getattr(s.original,name).__code__;after=getattr(private,name).__code__
        assert before.co_code==after.co_code and before.co_names==after.co_names
    source=ast.parse(Path(s.original.__file__).read_text(encoding='utf-8'))
    controller=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    assert sum(isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='install_facades' for n in controller.body)==1


def test_pure_completed_result_is_added_only_to_new_local_facade(tmp_path,monkeypatch):
    facade=types.ModuleType('fixture_only_facade');monkeypatch.setitem(sys.modules,'scripts.flow_matching.run_tsad_queue',facade)
    s.install_metadata_complete_result();job=dict(id='fixture_only',output_directory='output')
    assert facade.complete_result(tmp_path,job) is False
    out=tmp_path/'output';out.mkdir();(out/'model.pt').write_bytes(b'fixture_not_model');(out/'scores.npz').write_bytes(b'fixture_not_arrays')
    result=dict(status='completed',experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
        scores_sha256=s.sha(out/'scores.npz'),checkpoint_sha256=s.sha(out/'model.pt'))
    (out/'result.json').write_text(json.dumps(result))
    assert facade.complete_result(tmp_path,job)
    (out/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):facade.complete_result(tmp_path,job)
    with pytest.raises(ValueError):s.install_metadata_complete_result()


def fixture(tmp_path):
    old=dict(state_root=str(tmp_path/'old_state'),artifact_root=str(tmp_path/'old_output'))
    path=tmp_path/'old.json';path.write_text(json.dumps(old));(tmp_path/'old_state').mkdir()
    config=dict(predecessor_runtime=str(path));job=dict(id='fixture_only')
    state=tmp_path/'old_state'/'status.json';return config,job,state,old


def test_exact_metadata_only_failure_with_no_original_worker_may_enter(tmp_path):
    c,j,state,old=fixture(tmp_path);state.write_text(json.dumps(dict(failed_preserved={j['id']:s.EXPECTED_ERROR})))
    p=s.metadata_failure(c,j)
    assert p['no_original_worker_or_output_created'] and p['case_failure']==s.EXPECTED_ERROR


@pytest.mark.parametrize('fault',['other_failure','worker_attempt','numeric_output'])
def test_any_numeric_attempt_or_different_error_cannot_be_retried(tmp_path,fault):
    c,j,state,old=fixture(tmp_path);error=s.EXPECTED_ERROR
    if fault=='other_failure':error='FloatingPointError'
    state.write_text(json.dumps(dict(failed_preserved={j['id']:error})))
    if fault=='worker_attempt':(Path(old['state_root'])/'attempts'/j['id']).mkdir(parents=True)
    if fault=='numeric_output':(Path(old['artifact_root'])/j['id']).mkdir(parents=True)
    with pytest.raises(ValueError):s.metadata_failure(c,j)


def test_unobserved_predecessor_case_waits_instead_of_retrying(tmp_path):
    c,j,state,old=fixture(tmp_path)
    assert s.metadata_failure(c,j) is None
    state.write_text(json.dumps(dict(failed_preserved={})))
    assert s.metadata_failure(c,j) is None


def test_cold_recovery_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v2 as s;import sys;c=s.registration();s.verify(c);s.private_module();assert "numpy" not in sys.modules and "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
