import importlib.metadata
from pathlib import Path
import platform
import pytest
from scripts.flow_matching import bind_moment_icm_full_training_inputs_v1 as p


def test_full_registry_keeps_all_original_and_changed_sources_and_exact_epoch_scope():
    c=p.registration();original,sources=p.verify(c)
    assert len(sources)==2086 and c['original_anomaly_sources_requiring_raw_target_audit']==1627
    assert sum(s['family']=='anomaly_finite_control' for s in sources)==322
    assert original['count_audit']['total_items']==80850189


@pytest.mark.parametrize('change',['drop_sources','omit_raw_targets','raise_mapping_cap','relax_guard','bypass_lock','change_env','replace_closure'])
def test_incomplete_data_or_relaxed_budget_cannot_be_declared_fully_bound(change):
    c=p.registration()
    if change=='drop_sources':c['expected_sources']=1764
    elif change=='omit_raw_targets':c['original_anomaly_sources_requiring_raw_target_audit']=0
    elif change=='raise_mapping_cap':c['header_and_mapping_cache_bytes']=3*2**30
    elif change=='relax_guard':c['settings']['minimum_commit_headroom_bytes']=1
    elif change=='bypass_lock':c['resource_lock']='another.lock'
    elif change=='change_env':c['python']='another-python.exe'
    else:c['finite_control_closure_configuration']='another-closure.json'
    with pytest.raises(ValueError):p.verify(c)


def actual_registries():
    return {
        'classification':{e['dataset']:e for e in p.read(p.ROOT/p.epoch.CLASS)['entries']},
        'long_forecast':{e['dataset']:e for e in p.read(p.ROOT/p.epoch.LONG)['entries']
            if e['track']=='paired_full_multivariate' and e['split']=='train'},
        'monash':{e['dataset']:e for e in p.read(p.ROOT/p.epoch.MONASH)['entries']},
        'anomaly_original':{e['id']:e for e in p.read(p.ROOT/p.epoch.ANOMALY)['entries']}}


@pytest.mark.parametrize('family',['classification','long_forecast','monash','anomaly_original'])
def test_real_registry_resolves_every_selected_source_input_file(family):
    _,sources=p.verify(p.registration());registries=actual_registries()
    chosen=[s for s in sources if s['family']==family]
    for source in chosen:
        files=p.expected_files(source,registries,{})
        assert len(files)==(6 if family=='monash' else 3)
        assert all(Path(file).is_file() and len(digest)==64 for file,digest in files.items())
    # Rehash a complete actual source; this is data plumbing, not a fit/result.
    assert all(p.sha(file)==digest for file,digest in p.expected_files(chosen[0],registries,{}).items())


def test_finite_control_file_registry_requires_exact_directory_and_all_seven_names(tmp_path):
    directory=tmp_path/'source'/'train'
    source=dict(directory=str(directory),family='anomaly_finite_control',dataset_key='anomaly:fixture')
    entry=dict(cache_directory=str(directory.parent),splits={'train':{'files_sha256':{key:'a'*64 for key in p.finite.NAMES}}})
    files=p.expected_files(source,{},dict(fixture=entry))
    assert {Path(f).name for f in files}==set(p.finite.NAMES.values())
    entry['cache_directory']=str(tmp_path/'other')
    with pytest.raises(ValueError,match='Exact repaired cache directory'):p.expected_files(source,{},dict(fixture=entry))


def test_real_worker_path_binds_incremental_raw_targets_and_refuses_partial_scope(tmp_path,monkeypatch):
    c=p.registration();original,sources=p.verify(c)
    # Exercise the actual worker data path with one real closed original source.
    # Stub prerequisite/env checks for this fixture only; the mandatory final
    # 2086/1627 gate must reject it and no full readiness report may be written.
    source=next(s for s in sources if s['family']=='anomaly_original')
    c['artifact_root']=str(tmp_path/'partial_worker_fixture')
    c['environment']=dict(python=platform.python_version(),**{
        name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']})
    monkeypatch.setattr(p,'verify',lambda config:(original,[source]))
    monkeypatch.setattr(p,'prerequisites',lambda config:True)
    monkeypatch.setattr(p,'passed_upstream',lambda *args:dict(records=[]))
    with pytest.raises(ValueError,match='Every source and original raw-observation profile required'):p.worker(c)
    output=Path(c['artifact_root'])
    progress=p.read(output/'progress.json')
    assert progress['bound_sources']==progress['raw_anomaly_target_profiles']==1
    assert (output/'raw_target_bindings'/source['dataset_key'].split(':',1)[1]/'target_binding.json').is_file()
    assert not (output/p.REPORT).exists()
