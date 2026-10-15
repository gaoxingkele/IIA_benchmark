from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.flow_matching import run_original_full_raw873_experiments_v2 as s


def binding():
    return dict(manifest='F:/fixture_only_not_actual_data/manifest.json',manifest_sha256='fixture_manifest',records={
        d:dict(input_npz='F:/fixture_only_not_actual_data/'+d+'.npz',input_sha256='fixture_'+d,
            source_files=[dict(path='F:/fixture_only_not_actual_data/raw_'+d,sha256='raw_'+d)]) for d in s.DATASETS})


def allow_unregistered_config_digest(monkeypatch):
    original=s.sha
    monkeypatch.setattr(s,'sha',lambda path:'fixture_registration_sha' if Path(path)==s.ROOT/s.CONFIG else original(path))


def test_all873original_cases_across_all_seven_families_registered():
    c=s.registration();families=s.verify(c)
    assert {f['name']:f['required_cases'] for f in families}==dict(objectives=120,neural=510,baselines=105,entropic=45,reflow=60,tranad64=15,moment=18)
    assert c['required_cases']==873 and c['required_cases_per_dataset']==dict(smap=291,msl=291,smd=291)
    for kind,path in [('CPU',s.CPU_POLICY),('GPU',s.GPU_POLICY)]:
        original=s.read(s.ROOT/path)
        assert c[kind+'_settings']==original['settings'] and c[kind+'_resource_lock']==original['resource_lock']


@pytest.mark.parametrize('family',list(s.FAMILIES))
def test_source_models_budgets_precision_seeds_devices_and_gates_are_unchanged(family,monkeypatch):
    c=s.registration();allow_unregistered_config_digest(monkeypatch)
    spec=next(f for f in c['families'] if f['name']==family);original=s.read(s.ROOT/spec['source_queue'])
    before=deepcopy(original);queue=s.clone_queue(c,spec,binding());s.validate_clone(original,queue,spec,binding())
    assert original==before and queue['evaluation']==original['evaluation'] and queue['python']==original['python']
    assert queue.get('gpu_prerequisite_queues')==original.get('gpu_prerequisite_queues')
    for old,new in zip(s.selected(original),queue['jobs']):
        for key in ['model_config','window','seed','parameter_overrides']:
            assert old[key]==new[key]
        assert new['output_directory'].startswith('F:/aicoding/IIA_Data/')
        assert new['dataset_manifest_path']==binding()['manifest']


@pytest.mark.parametrize('fault',['case','seed','device','window','model','evaluation','python','phase_gate','old_pin','new_pin','new_array_pin','new_raw_pin','configuration_pin'])
def test_reduced_or_modified_scientific_identity_cannot_enter_fullraw_queue(fault,monkeypatch):
    c=s.registration();allow_unregistered_config_digest(monkeypatch)
    spec=next(f for f in c['families'] if f['name']=='neural');source=s.read(s.ROOT/spec['source_queue']);q=s.clone_queue(c,spec,binding())
    if fault=='case':q['jobs'].pop()
    elif fault=='seed':q['jobs'][0]['seed']=42
    elif fault=='device':q['jobs'][0]['parameter_overrides']['device']='cpu'
    elif fault=='window':q['jobs'][0]['window']=20
    elif fault=='model':q['jobs'][0]['model_config']='smaller_model.json'
    elif fault=='evaluation':q['evaluation']['bootstrap_resamples']=2
    elif fault=='python':q['python']='another_python'
    elif fault=='phase_gate':q['gpu_prerequisite_queues']=[]
    elif fault=='old_pin':
        target=s.selected(source)[0]['frozen_files'][0]['path']
        next(p for p in q['jobs'][0]['frozen_files'] if p['path']==target)['sha256']='changed'
    else:
        target={'new_pin':binding()['manifest'],'new_array_pin':binding()['records']['smap']['input_npz'],
            'new_raw_pin':binding()['records']['smap']['source_files'][0]['path'],'configuration_pin':s.CONFIG}[fault]
        q['jobs'][0]['frozen_files']=[p for p in q['jobs'][0]['frozen_files'] if p['path']!=target]
    with pytest.raises(ValueError):s.validate_clone(source,q,spec,binding())


def test_moment_contexts_keep_native512_and_TAB100_and_original_padding_contract(monkeypatch):
    c=s.registration();allow_unregistered_config_digest(monkeypatch);f=next(x for x in c['families'] if x['name']=='moment')
    q=s.clone_queue(c,f,binding())
    assert {j['window'] for j in q['jobs']}=={100,512} and {j['seed'] for j in q['jobs']}=={42}
    assert len(q['jobs'])==18 and f['controller_source'].endswith('run_moment_pretrained_queue.py')
    assert all(j['input_kind']=='standard' for j in q['jobs'])


def test_per_dataset_strong_baseline_configuration_is_not_replaced_by_SMAP_defaults(monkeypatch):
    c=s.registration();allow_unregistered_config_digest(monkeypatch);f=next(x for x in c['families'] if x['name']=='baselines')
    q=s.clone_queue(c,f,binding())
    for j in q['jobs']:assert '__'+j['dataset']+'__' in Path(j['model_config']).name
    assert len({j['model_config'] for j in q['jobs']})==21


def test_materialization_includes_every873case_but_claims_no_training_or_replay(tmp_path,monkeypatch):
    c=s.registration();allow_unregistered_config_digest(monkeypatch);c['artifact_root']=str(tmp_path/'fixture_queues_only')
    result=s.materialize(c,binding())
    assert result['required_cases']==873 and len(set(result['case_ids']))==873
    assert result['required_cases_per_dataset']==dict(smap=291,msl=291,smd=291)
    assert not result['trained_models_complete'] and not result['independent_native_replay_complete']
    assert not result['full_TAB_and_label_control_metrics_complete']
    for item in result['queues']:
        assert s.sha(item['path'])==item['sha256'] and len(s.read(item['path'])['jobs'])==item['jobs']
    with pytest.raises(FileExistsError):s.materialize(c,binding())


@pytest.mark.parametrize('field',['families','required_cases','CPU_settings','GPU_settings','datasets'])
def test_reduced_registration_or_relaxed_resources_cannot_enter(field):
    c=s.registration()
    if field in ['families','datasets']:c[field].pop()
    elif field=='required_cases':c[field]=120
    else:c[field]['minimum_commit_headroom_bytes']=1
    with pytest.raises(ValueError):s.verify(c)


def test_actual_data_is_required_before_materialized_family_execution(tmp_path):
    c=s.registration();c['artifact_root']=str(tmp_path/'not_created')
    with pytest.raises(ValueError,match='before NN imports'):s.verify_materialization(c)
    assert not Path(c['artifact_root']).exists()


def test_failed_data_guard_cannot_be_hidden_as_an_indefinite_wait(tmp_path,monkeypatch):
    c=s.registration();original=s.read
    source=dict(artifact_root=str(tmp_path/'raw'),state_root=str(tmp_path/'guard'))
    guard=tmp_path/'guard'/'guard_receipt.json';guard.parent.mkdir()
    guard.write_text(json.dumps(dict(passed=False,exit_code=1,resource_guard_aborted=False)))
    monkeypatch.setattr(s,'read',lambda p:source if Path(p)==s.ROOT/c['full_entity_preparation'] else original(p))
    with pytest.raises(ValueError,match='preparation failed'):s.ready_data(c)


def test_cold_registration_and_missing_data_gate_import_no_numpy_or_torch():
    code='from scripts.flow_matching import run_original_full_raw873_experiments_v2 as s;import sys;c=s.registration();s.verify(c);assert "numpy" not in sys.modules and "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr


def test_v2_binds_only_corrected_data_and_keeps_all873_scientific_cases():
    from scripts.flow_matching import run_original_full_raw873_experiments_v1 as previous
    old,new=previous.registration(),s.registration()
    for key in ['families','required_cases','required_cases_per_dataset','datasets','CPU_settings','CPU_resource_lock','GPU_settings','GPU_resource_lock','controller_python','test_label_controls','original_scope']:
        assert old[key]==new[key]
    assert new['full_entity_preparation']==s.preparation.CONFIG and new['full_entity_preparation']!=old['full_entity_preparation']
    assert new['artifact_root']!=old['artifact_root'] and new['state_root']!=old['state_root']
