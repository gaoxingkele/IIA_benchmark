from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from scripts.flow_matching import run_full_smap_DCD105_min105_controls_v1 as s


def test_all5original_full54_SMAP_seeds_and_resource_policy_retained():
    c=s.registration();s.verify(c)
    _,q,jobs=s.source();policy=s.read(s.ROOT/s.original.CPU_POLICY)
    assert c['settings']==policy['settings'] and c['resource_lock']==policy['resource_lock']
    assert c['original_evaluation']==q['evaluation'] and q['evaluation']['cpu_threads']==2
    assert len(jobs)==5 and {j['seed'] for j in jobs}==set(range(1103,1108))
    assert all(j['window']==105 and j['parameter_overrides']['device']=='cpu' for j in jobs)


def test_materialized_control_changes_only_data_identity_and_keeps_every_original_model_pin():
    c=s.registration();b=s.binding(c);q=s.clone_queue(c,b);s.validate_clone(c,q,b)
    _,old,jobs=s.source()
    for a,z in zip(jobs,q['jobs']):
        assert a['model_config']==z['model_config'] and a['parameter_overrides']==z['parameter_overrides']
        assert a['output_directory']!=z['output_directory'] and a['dataset_manifest_path']!=z['dataset_manifest_path']
        assert all(r in z['frozen_files'] for r in a['frozen_files'])
        p=s.read(s.ROOT/z['model_config'])['parameters']
        assert p['parameters']['window']==105 and p['epochs']==3


@pytest.mark.parametrize('fault',['drop_seed','shorten_window','lower_budget','remove_pin','old_data','evaluation'])
def test_missing_seed_changed_model_or_data_contract_rejected(fault):
    c=s.registration();b=s.binding(c);q=s.clone_queue(c,b)
    if fault=='drop_seed':q['jobs'].pop()
    elif fault=='shorten_window':q['jobs'][0]['window']=100
    elif fault=='lower_budget':q['jobs'][0]['parameter_overrides']['epochs']=1
    elif fault=='remove_pin':q['jobs'][0]['frozen_files'].pop()
    elif fault=='old_data':q['jobs'][0]['dataset_manifest_path']='old_manifest.json'
    else:q['evaluation']['cpu_threads']=1
    with pytest.raises(ValueError):s.validate_clone(c,q,b)


def test_every_actual_complete_SMAP_array_has_a_real105_window():
    from scripts.flow_matching import audit_full873_window_geometry_v1 as geometry
    c=s.registration();b=s.binding(c);record=b['records']['smap'];headers={}
    with zipfile.ZipFile(record['input_npz']) as archive:
        for entity in record['entities']:
            for role in ['train','validation','test']:
                key=entity['array_prefix']+'_'+role
                with archive.open(key+'.npy') as stream:headers[key]=geometry.array_header(stream)
    assert len(record['entities'])==54
    result=geometry.inspect_case(dict(window=105),record,headers)
    assert result['all_entities_pass_fixed_window_length_gate'] and not result['incompatible_role_inputs']
    entity=next(r for r in record['entities'] if r['entity_id']=='SMAP_D-12')
    assert entity['fit_interval']==[0,107] and entity['validation_interval']==[207,312]


def test_cold_full54_control_probe_does_not_import_or_run_the_models():
    code='from scripts.flow_matching import run_full_smap_DCD105_min105_controls_v1 as s;import sys;c=s.registration();s.verify(c);assert s.binding(c) is not None;assert "numpy" not in sys.modules and "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
