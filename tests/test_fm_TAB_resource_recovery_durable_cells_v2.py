"""Actual official metric functions on tiny probes;not benchmark performance."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.flow_matching import recover_TAB_resource_cells_v2 as s


def test_all878scope_original_metrics_and_only_strengthened_admission():
    c=s.registration();s.verify(c);old=s.read(s.ROOT/s.original.CONFIG)
    assert c['entries']==old['entries'] and c['required_cases']==878 and c['required_label_views']==2057
    assert c['settings']['minimum_commit_headroom_bytes']==32*2**30
    assert old['settings']['minimum_commit_headroom_bytes']==28*2**30
    assert {k:v for k,v in c['settings'].items() if k!='minimum_commit_headroom_bytes'}=={
        k:v for k,v in old['settings'].items() if k!='minimum_commit_headroom_bytes'}
    assert c['resource_lock']==old['resource_lock'] and c['stable_admission_samples']==6 and c['stable_poll_seconds']==5
    assert c['maximum_new_attempts_per_entity']==1


@pytest.mark.parametrize('fault',['scope','metric','memory','emergency','cap','lock','stable','retry'])
def test_shortcut_or_automatic_retry_policy_rejected(fault):
    c=s.registration()
    if fault=='scope':c['entries'].pop()
    elif fault=='metric':c['label_metrics'].pop()
    elif fault=='memory':c['settings']['minimum_commit_headroom_bytes']=28*2**30
    elif fault=='emergency':c['settings']['emergency_commit_headroom_bytes']=1
    elif fault=='cap':c['settings']['maximum_worker_rss_bytes']*=2
    elif fault=='lock':c['resource_lock']='different.lock'
    elif fault=='stable':c['stable_admission_samples']=1
    else:c['maximum_new_attempts_per_entity']=2
    with pytest.raises(ValueError):s.verify(c)


def probe(c):
    scores=np.linspace(0,1,40);labels=np.zeros(40,np.uint8);labels[10:12]=1;labels[30:32]=1
    result=dict(metrics=dict(strict={str(r):dict(calibration=dict(threshold=.5)) for r in c['strict_ratios']},
        tab_core_diagnostic=dict(ratio_grid=[dict(ratio_percent=r,threshold=.4) for r in c['diagnostic_pooled_ratios']])))
    return scores,dict(first=labels),result


def test_actual_metric_program_resumes_all201cells_after_simulated_process_interruption(tmp_path):
    c=s.registration();e=c['entries'][0];scores,views,result=probe(c);metrics=s.original.load_metrics(c)
    line=dict(fixture_only=True);events=[]
    def interrupt(value):
        events.append(value)
        if len(events)==8:raise RuntimeError('simulated process interruption outside original metric functions')
    with pytest.raises(RuntimeError,match='simulated process interruption'):
        s.evaluate_checkpointed(c,e,'e000',scores,views,result,metrics,line,tmp_path/'bank',interrupt)
    assert len(list((tmp_path/'bank').glob('*.json')))==7
    restored,journal=s.evaluate_checkpointed(c,e,'e000',scores,views,result,metrics,line,tmp_path/'bank')
    assert journal['checkpoint_cells']==7 and journal['fresh_cells']==194 and len(journal['cell_records'])==201
    direct=s.original.evaluate_entity(c,scores,views,result,metrics)
    for name,outcome in direct['first'].items():
        actual=restored['first'][name]
        assert actual['status']==outcome['status'] and actual['value']==outcome['value']
        assert actual['warnings']==outcome['warnings'] and actual['original_stdout_stderr']==outcome['original_stdout_stderr']
    before={p.name:s.sha(p) for p in (tmp_path/'bank').glob('*.json')}
    again,all_cached=s.evaluate_checkpointed(c,e,'e000',scores,views,result,metrics,line,tmp_path/'bank')
    assert all_cached['checkpoint_cells']==201 and all_cached['fresh_cells']==0 and again==restored
    assert before=={p.name:s.sha(p) for p in (tmp_path/'bank').glob('*.json')}
    assert not list((tmp_path/'bank').glob('*.pending-*'))
    assert not list((tmp_path/'bank').glob('.pending-*'))


@pytest.mark.parametrize('fault',['score','label','source','function','outcome','identity'])
def test_changed_arguments_function_source_or_cached_outcome_cannot_be_reused(tmp_path,fault):
    c=s.registration();e=c['entries'][0];scores,views,result=probe(c);metrics=s.original.load_metrics(c);line=dict(fixture_only=True)
    def interrupt(value):
        if value['metric']=='score/auc_pr':raise RuntimeError('stop after first cell')
    with pytest.raises(RuntimeError):s.evaluate_checkpointed(c,e,'e000',scores,views,result,metrics,line,tmp_path,interrupt)
    files=list(tmp_path.glob('*.json'));assert len(files)==1
    if fault=='score':scores[0]+=.01
    elif fault=='label':views['first'][0]=1
    elif fault=='source':line['checkpoint_sha256']='changed'
    elif fault=='function':
        def alternate(a,b):return 0.
        metrics['score']['auc_roc']=alternate
    else:
        content=s.read(files[0])
        if fault=='outcome':content['outcome']['value']=.123
        else:content['identity_sha256']='changed'
        files[0].write_text(json.dumps(content))
    with pytest.raises(ValueError,match='checkpoint'):s.evaluate_checkpointed(c,e,'e000',scores,views,result,metrics,line,tmp_path)


def test_atomic_checkpoint_never_overwrites_or_silently_imputes_undefined_result(tmp_path):
    path=tmp_path/'cell.json';s.atomic_new(path,dict(status='nonfinite',value=None));before=path.read_bytes()
    with pytest.raises(FileExistsError):s.atomic_new(path,dict(status='finite',value=0))
    assert path.read_bytes()==before


def test_complete_worker_preserves_original21metric_body_and_writes_all804_durable_cells(tmp_path,monkeypatch):
    c=s.registration();e=next(e for e in c['entries'] if e['key']=='5eedb87987df7781d41c')
    scores,one,result=probe(c);metrics=s.original.load_metrics(c)
    source=tmp_path/'source';source.mkdir()
    labels=one['first'];np.savez_compressed(source/'scores.npz',e000_test=scores,e000_labels=labels)
    views={}
    for policy in e['case']['required_label_policies']:
        path=tmp_path/(policy+'.npz');np.savez_compressed(path,e000_labels=labels)
        views[policy]=dict(path=str(path),sha256=s.sha(path))
    dataset=dict(entities=[dict(entity_id='fixture_only',array_prefix='e000',test_points=len(scores))],
        labels_only_controls=views,primary_label_policy=e['case']['required_label_policies'][0])
    lineage=dict(fixture_only=True,native_scores=str(source/'scores.npz'))
    bound=dict(dataset=dataset,lineage=lineage,job=dict(output_directory='source'),result=result)
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s.original,'ROOT',tmp_path)
    monkeypatch.setattr(s,'CONFIG','fixture_control.json')
    monkeypatch.setattr(s,'resource_abort',lambda config,entry:dict(source_lineage=lineage,fixture_only=True))
    monkeypatch.setattr(s.original,'ready_case',lambda config,entry:bound)
    monkeypatch.setattr(s.original,'load_metrics',lambda config:metrics)
    s.write(tmp_path/s.original.CONFIG,dict(fixture_only=True))
    s.write(tmp_path/s.CONFIG,dict(fixture_only=True))
    c['artifact_root']=str(tmp_path/'artifacts');c['state_root']='state'
    s.worker(c,e,'e000')
    attempt,output=s.paths(c,e,'e000');p=s.read(output/s.REPORT)
    assert p['passed'] and p['entity']=='fixture_only' and p['metric_calls']==804
    assert p['durable_cell_journal']['fresh_cells']==804 and p['durable_cell_journal']['checkpoint_cells']==0
    assert len(p['durable_cell_journal']['cell_records'])==804
    assert p['all_original_metric_calls_freshly_executed_this_attempt']
    assert not p['neural_training_or_inference_performed'] and not p['original_metric_definition_or_grid_changed']
    assert not attempt.exists()  # No fake production/scientific resource guard.


def test_actual_old6_guarded_fragments_remain_reused_and_A7_resource_event_is_terminal():
    c=s.registration();e=next(e for e in c['entries'] if e['key']=='5eedb87987df7781d41c')
    evidence=s.resource_abort(c,e)
    assert evidence is not None and not evidence['failed_guard_is_algorithm_accuracy_failure']
    assert evidence['original_failed_guards'][0]['prefix']=='e006'
    for i in range(6):
        path,p,kind=s.checked_fragment(c,e,f'e{i:03d}')
        assert kind=='original_guarded_fragment' and p['metric_calls']==804
        assert s.sha(path)==s.original.sha(path)
    pending=next(e for e in c['entries'] if e['track']=='common105_DCD5')
    assert s.resource_abort(c,pending) is None


def test_stability_counter_resets_and_retains_both_original_resource_dimensions(monkeypatch):
    c=s.registration();memory=c['settings']['minimum_commit_headroom_bytes'];physical=c['settings']['minimum_available_memory_bytes']
    samples=[dict(physical_available_bytes=physical,commit_available_bytes=memory) for _ in range(2)]
    samples.append(dict(physical_available_bytes=physical,commit_available_bytes=memory-1))
    samples += [dict(physical_available_bytes=physical,commit_available_bytes=memory) for _ in range(6)]
    seen=[];sleeps=[];it=iter(samples)
    monkeypatch.setattr(s.time,'sleep',lambda t:sleeps.append(t))
    s.stable_headroom(c,dict(resource_snapshot=lambda:next(it)),seen.append)
    assert [v['stable_samples'] for v in seen]==[1,2,0,1,2,3,4,5,6]
    assert sleeps==[5]*8


def test_cold_registration_and_actual_abort_verification_import_no_numpy_or_torch():
    code='from scripts.flow_matching import recover_TAB_resource_cells_v2 as s;import sys;c=s.registration();s.verify(c);e=next(e for e in c["entries"] if e["key"]=="5eedb87987df7781d41c");assert s.resource_abort(c,e) is not None;assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
