from copy import deepcopy
import hashlib
import json
import subprocess
import sys
import pytest
from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as s
from tests.test_fm_timesnet_constant_fit_complete_v1 import fixture as original_fixture


def fixture(seed=1104):
    c,q,j,r,m,g,t=deepcopy(original_fixture())
    j.update(id=s.identity(seed),seed=seed,parameter_overrides=dict(seed=seed,device='cpu'))
    q['jobs'][0]=j;c.update(job_id=j['id'],seed=seed);g['job_id']=j['id']
    r.update(job=j,id=j['id'],experiment_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest())
    return c,q,j,r,m,g,t


@pytest.mark.parametrize('seed',s.SEEDS)
def test_full_actual_seed_identity_retains_original_all_checks(seed):
    values=fixture(seed);before=deepcopy(values)
    s.validate_identity(*values)
    assert values==before


@pytest.mark.parametrize('change',['scope','queue','seed','override','result_id','guard_id','fingerprint',
    'window','width','epochs','batch','template_seed','guard','threads','aborted','exit','capacity','author_claim'])
def test_reduced_or_substituted_model_identity_is_rejected(change):
    c,q,j,r,m,g,t=fixture()
    if change=='scope':c['original_scope']['papers']=44
    elif change=='queue':q['jobs'].pop()
    elif change=='seed':j['seed']=1105
    elif change=='override':j['parameter_overrides']['seed']=1105
    elif change=='result_id':r['id']='substitute'
    elif change=='guard_id':g['job_id']='substitute'
    elif change=='fingerprint':r['experiment_sha256']='substitute'
    elif change=='window':j['window']=50
    elif change=='width':m['parameters']['parameters']['d_model']=64
    elif change=='epochs':m['parameters']['epochs']=1
    elif change=='batch':m['parameters']['parameters']['batch_size']=64
    elif change=='template_seed':m['parameters']['seed']=1104
    elif change=='guard':c['settings']['minimum_commit_headroom_bytes']=16
    elif change=='threads':c['cpu_threads']=1
    elif change=='aborted':g['resource_guard_aborted']=True
    elif change=='exit':g['exit_code']=1
    elif change=='capacity':r['parameter_count']-=1
    else:r['paper_equivalence']=True
    with pytest.raises(ValueError):s.validate_identity(c,q,j,r,m,g,t)


def test_real_queue_projection_preserves_35_distinct_registered_jobs():
    c,q,j,r,m,g,t=fixture()
    q['jobs'][1]['id']=s.original.CANONICAL
    before=deepcopy((c,q,j,r,m,g,t));s.validate_identity(c,q,j,r,m,g,t)
    assert before==(c,q,j,r,m,g,t)


def test_original_worker_bytecode_and_shared_globals_are_unchanged():
    old=s.original.worker.__globals__.copy();worker=s.numerical_worker()
    assert worker.__code__ is s.original.worker.__code__
    assert worker.__globals__ is not s.original.worker.__globals__
    assert worker.__globals__['binding'] is s.binding
    assert worker.__globals__['restore_model'] is s.original.restore_model
    assert worker.__globals__['require_exact_scores'] is s.original.require_exact_scores
    assert old==s.original.worker.__globals__


def test_registration_covers_all_remaining_stages_without_replaying_closed1103():
    c=s.registration();jobs=s.verify(c)
    assert len(jobs)==4 and {j['seed'] for j in jobs.values()}==set(s.SEEDS)
    assert c['required_new_audit_stages']==8 and c['modes']==['point','native']
    assert c['required_original_seed_cohort']==[1103,1104,1105,1106,1107]
    original=s.read(s.ROOT/s.GUARD)
    assert c['settings']==original['settings'] and c['resource_lock']==original['resource_lock']
    for job in jobs.values():
        for mode in c['modes']:
            stage=s.stage_config(c,job,mode)
            assert stage['seed']==job['seed'] and stage['job_id']==job['id']
            assert stage['output_directory'].startswith('F:/aicoding/IIA_Data/')


@pytest.mark.parametrize('field',['jobs','settings','modes'])
def test_shortened_registration_or_relaxed_resources_rejected(field):
    c=s.registration()
    if field=='jobs':c[field].pop()
    elif field=='modes':c[field].pop()
    else:c[field]['minimum_commit_headroom_bytes']=1
    with pytest.raises(ValueError):s.verify(c)


def test_missing_original_training_rejected_before_numerical_worker(monkeypatch):
    c=s.registration();job=c['jobs'][0];stage=s.stage_config(c,job,'native')
    monkeypatch.setattr(s,'source_ready',lambda *a:False)
    monkeypatch.setattr(s,'numerical_worker',lambda:pytest.fail('NN body must not execute'))
    with pytest.raises(ValueError,match='before NN imports'):s.worker(stage,'fixture_only')


def test_cold_registration_has_no_numpy_or_torch_imports():
    code='from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    run=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
