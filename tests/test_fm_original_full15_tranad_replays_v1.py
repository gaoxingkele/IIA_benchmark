"""Replay/identity tests;small neural fixtures are not benchmark results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess,sys
import numpy as np
import pytest
from scripts.flow_matching import replay_original_full15_tranad_v1 as s


def test_all15new_fullraw_cases_and_all35views_keep_original_full_scope():
    c=s.registration();cases=s.verify(c);old=s.read(s.ROOT/s.scalar.CONFIG)
    assert cases==[x for x in old['cases'] if x['family']=='tranad64']
    assert {(x['dataset'],x['seed']) for x in cases}=={(d,v) for d in ['smap','msl','smd'] for v in range(1103,1108)}
    assert sum(len(x['required_label_policies']) for x in cases)==35
    assert c['original_all_family_cases']==873 and c['original_all_label_views']==2037
    guard=s.read(s.ROOT/s.POLICY)
    assert c['settings']==guard['settings'] and c['resource_lock']==guard['resource_lock']
    assert c['original_scope']==old['original_scope']


@pytest.mark.parametrize('change',['cases','required_family_native_cases','required_family_label_views','settings','resource_lock','model_scoring','native_comparison','original_all_family_cases'])
def test_reduced_or_precision_changed_native_scope_cannot_be_registered(change):
    c=s.registration()
    if change=='cases':c[change].pop()
    elif change=='settings':c[change]['minimum_commit_headroom_bytes']=1
    elif change=='resource_lock':c[change]='other.lock'
    elif change in ['model_scoring','native_comparison']:c[change]='approximate'
    else:c[change]=1
    with pytest.raises(ValueError):s.verify(c)


def model_fixture():
    c=s.registration();case=next(x for x in c['cases'] if x['seed']==1107 and x['dataset']=='smap')
    queue,job=s.scalar.resolve_case(case);model=s.read(s.ROOT/job['model_config'])
    return c,deepcopy(case),deepcopy(queue),deepcopy(job),deepcopy(model)


def test_nondefault_job_seed_restores_actual_override_without_metadata_projection():
    c,case,queue,job,model=model_fixture();s.validate_model(c,case,queue,job,model)
    assert model['parameters']['seed']==1103 and case['seed']==job['seed']==1107
    assert job['parameter_overrides']['seed']==1107


@pytest.mark.parametrize('change',['precision','method','epochs','batch','window','width','dropout','seed','device','threads'])
def test_any_source_precision_budget_or_job_identity_change_is_rejected(change):
    c,case,queue,job,model=model_fixture()
    if change=='precision':model['precision']='float32'
    elif change=='method':model['parameters']['method']='other'
    elif change=='epochs':model['parameters']['epochs']=1
    elif change=='batch':model['parameters']['parameters']['batch_size']=16
    elif change=='window':case['window']=100
    elif change=='width':model['parameters']['parameters']['d_ff']=8
    elif change=='dropout':model['parameters']['parameters']['dropout']=0.
    elif change=='seed':job['parameter_overrides']['seed']=42
    elif change=='device':job['parameter_overrides']['device']='cuda'
    else:queue['evaluation']['cpu_threads']=1
    with pytest.raises(ValueError):s.validate_model(c,case,queue,job,model)


@pytest.mark.parametrize('features',[2,25,38,55])
def test_every_tensor_restores_and_source_score_bytes_match_without_fit(features,monkeypatch):
    import torch
    from iia_benchmark.models.fm_tranad_reference_float64_v1 import ReferenceFloat64TranADDetector
    from scripts.flow_matching.audit_tranad_float64_constant_fit_complete_v1 import restore_model
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical
    torch.set_num_threads(2)
    _,case,_,job,model=model_fixture();parameters=model['parameters']
    original=ReferenceFloat64TranADDetector(**parameters['parameters'],epochs=5,seed=case['seed'],device='cpu')
    original._device=torch.device('cpu');original._model=original._build(features).double()
    checkpoint=dict(configuration=model,overrides=job['parameter_overrides'],window=10,adjacency=None,
        modules={'_model':{k:v.detach().clone() for k,v in original._model.state_dict().items()}})
    count=sum(p.numel() for p in original._model.parameters())
    def forbid(*args,**kwargs):raise AssertionError('No fresh fit allowed')
    monkeypatch.setattr(ReferenceFloat64TranADDetector,'fit',forbid)
    restored=restore_model(checkpoint,job,model,features,count);before=s.check_restored_weights(restored,checkpoint)
    windows=np.random.default_rng(13).normal(size=(3,10,features))
    require_identical(original.score(windows),restored.score(windows))
    assert s.check_restored_weights(restored,checkpoint)==before
    with torch.no_grad():next(restored._model.parameters()).add_(1)
    with pytest.raises(ValueError,match='tensor'):s.check_restored_weights(restored,checkpoint)


def native_fixture(tmp_path,monkeypatch):
    c=dict(state_root=str(tmp_path/'state'),artifact_root=str(tmp_path/'artifact'))
    case=dict(id='fixture_only_not_benchmark',worker_python=sys.executable)
    attempt,output=s.paths(c,case,'native');attempt.mkdir(parents=True);output.mkdir(parents=True)
    scores=output/'scores.bin';scores.write_bytes(b'fixture_only_no_real_predictions')
    proof=dict(passed=True,case=case,configuration_sha256='a'*64,
        all_array_dtype_shape_content_bytes_identical=True,all_model_tensors_identical_before_and_after=True,
        every_saved_native_array_serialized_and_reloaded=True,additional_fit_calls=0,additional_optimizer_updates=0,
        CUDA_context_initialized=False,scores_path=str(scores),scores_sha256=s.sha(scores))
    path=output/s.REPORTS['native'];path.write_text(json.dumps(proof),encoding='utf-8')
    g=dict(passed=True,exit_code=0,resource_guard_aborted=False,proof_sha256=s.sha(path),
        exact_child_handle=dict(command=s.worker_command(case,'native')))
    (attempt/'guard_receipt.json').write_text(json.dumps(g),encoding='utf-8')
    raw_sha=s.sha;monkeypatch.setattr(s,'sha',lambda p:'a'*64 if Path(p)==s.ROOT/s.CONFIG else raw_sha(p))
    return c,case,attempt,output,proof,g


def test_native_proof_requires_actual_successful_exact_worker_guard(tmp_path,monkeypatch):
    c,case,attempt,output,p,g=native_fixture(tmp_path,monkeypatch)
    path,proof=s.ready_native(c,case);assert proof['case']==case
    (attempt/'guard_receipt.json').unlink()
    with pytest.raises(ValueError,match='successful native guard'):s.ready_native(c,case)


@pytest.mark.parametrize('fault',['fit','update','CUDA','incomplete_arrays','changed_weights','reload','failed','abort','proof_hash','command','scores'])
def test_incomplete_changed_or_wrong_native_attempt_cannot_enter_review(tmp_path,monkeypatch,fault):
    c,case,attempt,output,p,g=native_fixture(tmp_path,monkeypatch)
    if fault=='fit':p['additional_fit_calls']=1
    elif fault=='update':p['additional_optimizer_updates']=1
    elif fault=='CUDA':p['CUDA_context_initialized']=True
    elif fault=='incomplete_arrays':p['all_array_dtype_shape_content_bytes_identical']=False
    elif fault=='changed_weights':p['all_model_tensors_identical_before_and_after']=False
    elif fault=='reload':p['every_saved_native_array_serialized_and_reloaded']=False
    elif fault=='failed':g['exit_code']=1
    elif fault=='abort':g['resource_guard_aborted']=True
    elif fault=='command':g['exact_child_handle']['command'][-1]='review'
    elif fault=='scores':Path(p['scores_path']).write_bytes(b'changed')
    path=output/s.REPORTS['native'];path.write_text(json.dumps(p),encoding='utf-8')
    g['proof_sha256']='wrong' if fault=='proof_hash' else s.sha(path)
    (attempt/'guard_receipt.json').write_text(json.dumps(g),encoding='utf-8')
    with pytest.raises(ValueError):s.ready_native(c,case)


def test_cold_registration_imports_no_numpy_or_torch():
    code='from scripts.flow_matching import replay_original_full15_tranad_v1 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
