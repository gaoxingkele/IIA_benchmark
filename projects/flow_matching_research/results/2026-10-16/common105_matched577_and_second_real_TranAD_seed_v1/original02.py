"""All actual source jobs/data are checked;no synthetic benchmark is claimed."""
from collections import Counter
from copy import deepcopy
import subprocess
import sys
import zipfile
import pytest
from scripts.flow_matching import run_common105_matched577_v1 as s


def test_every_changed_input_global_model_and_original_resource_policy_retained():
    c=s.registration();s.verify(c);old=s.read(s.ROOT/s.original.CONFIG)
    assert {f['name']:f['required_cases'] for f in c['families']}==s.COUNTS
    assert sum(f['required_cases'] for f in c['families'])==577
    for k in ['CPU_settings','CPU_resource_lock','GPU_settings','GPU_resource_lock','original_scope']:
        assert c[k]==old[k]
    assert c['required_cohort_per_dataset']==dict(smap=291,msl=291,smd=291)


@pytest.mark.parametrize('fault',['family','case','memory','GPU_memory','lock','cohort','labels','SMD'])
def test_reduced_comparison_or_budget_rejected(fault):
    c=s.registration()
    if fault=='family':c['families'].pop()
    elif fault=='case':c['families'][0]['selected_source_ids'].pop()
    elif fault=='memory':c['CPU_settings']['minimum_commit_headroom_bytes']=1
    elif fault=='GPU_memory':c['GPU_settings']['minimum_gpu_free_bytes']=1
    elif fault=='lock':c['CPU_resource_lock']='different.lock'
    elif fault=='cohort':c['required_comparison_cases']=577
    elif fault=='labels':c['required_comparison_label_views']=100
    else:c['required_unchanged_SMD_references']=1
    with pytest.raises(ValueError):s.verify(c)


@pytest.fixture(scope='module')
def bound():
    c=s.registration();b=s.ready_data(c);assert b is not None
    return c,b


def test_all577_new_jobs_keep_all_parameters_scientific_fields_and_original_frozen_files(bound):
    c,b=bound;datasets=Counter();ids=[]
    for f in c['families']:
        original=s.read(f['source_queue']);q=s.clone_queue(c,f,b);s.validate_clone(original,q,f,b)
        for a,z in zip(s.selected(original,f),q['jobs']):
            assert a['model_config']==z['model_config'] and a['parameter_overrides']==z['parameter_overrides']
            assert a['seed']==z['seed'] and a['window']==z['window']
            assert all(p in z['frozen_files'] for p in a['frozen_files'])
            assert z['dataset_manifest_path']==b['manifest'] and z['output_directory']!=a['output_directory']
            assert len(str(s.ROOT/z['output_directory']))+30<260
            datasets[z['dataset']]+=1;ids.append(z['id'])
        assert q['evaluation']==original['evaluation'] and q['python']==original['python']
    assert dict(datasets)==dict(smap=286,msl=291) and len(set(ids))==577


@pytest.mark.parametrize('fault',['model','window','seed','budget','frozen','manifest','queue_hash','phase_gates'])
def test_any_method_window_budget_or_source_gate_change_is_rejected(bound,fault):
    c,b=bound;f=next(x for x in c['families'] if x['name']=='neural');source=s.read(f['source_queue']);q=s.clone_queue(c,f,b)
    if fault=='model':q['jobs'][0]['model_config']='different.json'
    elif fault=='window':q['jobs'][0]['window']=10
    elif fault=='seed':q['jobs'][0]['seed']=42
    elif fault=='budget':q['jobs'][0]['parameter_overrides']['epochs']=1
    elif fault=='frozen':q['jobs'][0]['frozen_files'].pop()
    elif fault=='manifest':q['jobs'][0]['dataset_manifest_path']='old_manifest.json'
    elif fault=='queue_hash':q['full_raw_data_manifest_sha256']='old_hash'
    else:q['gpu_prerequisite_queues']=[]
    with pytest.raises(ValueError):s.validate_clone(source,q,f,b)


@pytest.mark.parametrize('fault',['changed','bytes','labels','normalizer','group','old_binding','dropped'])
def test_SMD_references_require_actual_all28_array_label_and_split_equivalence(bound,fault):
    c,new=bound;old=deepcopy(new['original_full_binding'])
    cross=s.read(new['cross_version_report'])['cross_version'];new=deepcopy(new);cross=deepcopy(cross)
    first=next(x for x in cross['cases'] if x['entity_id'].startswith('SMD_'))
    if fault=='changed':first['split_changed']=True
    elif fault=='bytes':first['all_unaffected_coordinates_byte_equal']=False
    elif fault=='labels':first['test_labels_byte_equal']=False
    elif fault=='normalizer':new['records']['smd']['entities'][0]['mean'][0]+=1
    elif fault=='group':new['records']['smd']['entities'][0]['bootstrap_group']='joined_entity'
    elif fault=='old_binding':old['manifest_sha256']='changed'
    else:new['records']['smd']['entities'].pop()
    with pytest.raises(ValueError):s.SMD_equivalence(new,old,cross)


def test_all_actual_SMAP_MSL_roles_support_windows_except_specialized_MOMENT_fill(bound):
    from scripts.flow_matching import audit_full873_window_geometry_v1 as geometry
    from scripts.flow_matching import audit_full873_window_geometry_routing_v2 as routing
    c,b=bound;headers={}
    for dataset in ['smap','msl']:
        d=b['records'][dataset];headers[dataset]={}
        with zipfile.ZipFile(d['input_npz']) as archive:
            for e in d['entities']:
                for role in ['train','validation','test']:
                    key=e['array_prefix']+'_'+role
                    with archive.open(key+'.npy') as stream:headers[dataset][key]=geometry.array_header(stream)
    count=0
    for f in c['families']:
        for j in s.selected(s.read(f['source_queue']),f):
            row=geometry.inspect_case(dict(family=f['name'],window=j['window']),b['records'][j['dataset']],headers[j['dataset']])
            actual=routing.classify(row,dict(actual_specialized_worker_retained=True))
            assert not actual['actual_worker_predicted_incompatible'];count+=1
    assert count==577


def test_materialized_cohort_keeps_all873_separate_lineages_without_duplicate_new_runs(bound,tmp_path):
    c,b=bound;queues=[]
    for f in c['families']:
        path=tmp_path/(f['name']+'.json');s.write(path,s.clone_queue(c,f,b))
        queues.append(dict(family=f['name'],path=str(path),sha256=s.sha(path)))
    cohort=s.cohort(c,b,queues)
    assert len(cohort)==873 and len({x['canonical_source_id'] for x in cohort})==873
    assert Counter(r['mode'] for r in cohort)==dict(new_common105_execution=577,existing_common105_DCD_execution=5,unchanged_SMD_original_execution=291)
    assert sum(len(r['required_label_policies']) for r in cohort)==2037
    assert all(not r['original_result_rebound'] for r in cohort)
    assert all(r['dataset']=='smd' for r in cohort if r['mode']=='unchanged_SMD_original_execution')
    assert all(r['dataset']!='smd' for r in cohort if r['mode']=='new_common105_execution')


def test_cold_registration_and_actual_data_binding_do_not_import_torch_or_numpy():
    code='from scripts.flow_matching import run_common105_matched577_v1 as s;import sys;c=s.registration();s.verify(c);assert s.ready_data(c) is not None;assert "torch" not in sys.modules and "numpy" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
