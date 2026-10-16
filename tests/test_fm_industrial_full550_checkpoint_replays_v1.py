"""Checkpoint and full-array contracts; tiny fixtures are not performance."""
from copy import deepcopy
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from scripts.flow_matching import replay_industrial_full550_checkpoints_v1 as s
from scripts.flow_matching.model_registry import build_configured_method
from scripts.flow_matching.run_tsad_experiment import align_scores,covering_windows


@pytest.mark.parametrize('length',[100,101,199,200,201,351])
@pytest.mark.parametrize('scalar',[False,True])
def test_full_original_overlap_and_tail_alignment_for_both_score_layouts(length,scalar):
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import native_windows
    values=np.random.default_rng(42).normal(size=(length,3)).astype(np.float32)
    windows,starts=native_windows(values,100);old,positions=covering_windows(values,100)
    assert np.array_equal(windows,old) and np.array_equal(starts,positions)
    scores=np.random.default_rng(43).normal(size=(len(starts),) if scalar else (len(starts),100)).astype(np.float32)
    assert np.array_equal(s.native_align(scores,starts,length,100),align_scores(scores,positions,length,100))


@pytest.mark.parametrize('fault',['missing_tail','nonfinite','wrong_width','wrong_window_count'])
def test_never_pad_drop_tail_or_accept_nonfinite_scores(fault):
    starts=[0,100,101];scores=np.ones((3,100))
    if fault=='missing_tail':starts=starts[:-1];scores=scores[:-1]
    elif fault=='nonfinite':scores[0,0]=np.nan
    elif fault=='wrong_width':scores=scores[:,:99]
    else:scores=scores[:-1]
    with pytest.raises(ValueError):s.native_align(scores,starts,201,100)


def checkpoint(source,configuration):
    p=configuration['parameters'];window=p['parameters']['window'] if 'parameters' in p else p['window']
    return dict(configuration=configuration,overrides=dict(seed=1103,device='cpu'),window=window,adjacency=None,
        modules={name:deepcopy(value.state_dict()) for name,value in vars(source).items() if isinstance(value,torch.nn.Module)})


@pytest.mark.parametrize('family',['independent','target','vp','sf2m_auxiliary','reflow2'])
def test_full_source_score_matches_restored_model_and_observed_formulas(family):
    torch.set_num_threads(2)
    stable=family=='sf2m_auxiliary';staged=family=='reflow2'
    entry=('fm_reflow:ReflowCFMDetector' if staged else 'fm_objective_stable:StableCFMObjectiveDetector' if stable
        else 'fm_objective_detectors:CFMObjectiveDetector')
    params=dict(objective='sf2m' if stable else 'rectified' if staged else family,window=100,hidden_size=8,
        epochs=1,batch_size=64,seed=1103,device='cpu',monte_carlo_samples=1,score_times=[.5],sigma=0.1)
    if stable:params.update(score_weight=1.0,sigma=1.)
    if staged:params.update(sigma=0.,reflow_stages=2,reflow_epochs=1,reflow_ode_steps=2,reflow_solver='heun')
    config=dict(entrypoint='iia_benchmark.models.'+entry,parameters=params)
    source=build_configured_method(config);values=np.random.default_rng(1103).normal(size=(3,100,2)).astype(np.float32)
    source.fit(values);saved=checkpoint(source,config);job=dict(window=100,parameter_overrides=saved['overrides'])
    model,score,obs=s.restore(saved,job,config,2);before=s.check_weights(model,saved)
    actual=score(values);expected=source.score(values)
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
    s.validate_observation([dict(windows=3,features=2)],obs,config)
    assert s.check_weights(model,saved)==before
    with pytest.raises(RuntimeError):model.fit(values)
    if staged:assert '_reflow_evidence' in before
    if stable:assert model._model[-1].out_features==400


@pytest.mark.parametrize('name',[
    'fm_strict_anomaly_transformer__psm__registered','fm_strict_dcdetector__psm__registered',
    'fm_strict_itransformer__psm__registered','fm_strict_timesnet__psm__registered',
    'fm_strict_tranad__psm__registered','fm_strict_usad__psm__registered','fm_strict_usad__psm__signed_paper_loss'])
def test_all_seven_legacy_architectures_restore_without_fit_and_preserve_source_score(name):
    torch.set_num_threads(2)
    config=deepcopy(s.read(s.ROOT/'configs/models/fm_strict_baselines'/(name+'.json')))
    # Only a unit fixture: original registered550 configurations stay frozen.
    config['parameters']['epochs']=1
    window=config['parameters']['parameters']['window']
    source=build_configured_method(config,dict(seed=1103,device='cpu'))
    data=np.random.default_rng(1103).normal(size=(3,window,2)).astype(np.float32);source.fit(data)
    saved=checkpoint(source,config);job=dict(window=window,parameter_overrides=saved['overrides'])
    model,score,obs=s.restore(saved,job,config,2);before=s.check_weights(model,saved)
    actual=score(data);expected=source.score(data)
    assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
    s.validate_observation([dict(windows=3,features=2)],obs,config)
    assert s.check_weights(model,saved)==before
    with pytest.raises(RuntimeError):model.fit(data)
    with pytest.raises(RuntimeError):model._delegate.fit(data)


def test_corrupted_checkpoint_dtype_cannot_be_silently_cast_and_certified():
    config=dict(entrypoint='iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector',parameters=dict(
        objective='independent',window=100,hidden_size=8,epochs=1,device='cpu',seed=1103))
    torch.set_num_threads(2);source=build_configured_method(config);source.fit(np.zeros((2,100,2),dtype=np.float32))
    saved=checkpoint(source,config);key=next(iter(saved['modules']['_model']));saved['modules']['_model'][key]=saved['modules']['_model'][key].double()
    with pytest.raises(ValueError):
        model,_,_=s.restore(saved,dict(window=100,parameter_overrides=saved['overrides']),config,2)
        s.check_weights(model,saved)


def test_missing_source_is_pending_and_does_not_create_artifacts(tmp_path):
    assert s.source_binding({},dict(case={}),dict(case=dict(output_directory=str(tmp_path/'missing'))),'case') is None
    assert not (tmp_path/'missing').exists()


class Arrays(dict):
    @property
    def files(self):return list(self)


def grouped_fixture(window):
    ds=dict(training_groups=[],validation_groups=[],entities=[]);data=Arrays();saved=Arrays()
    rng=np.random.default_rng(42)
    # An analytic score independent of either alignment implementation.
    def scoring(windows):return np.square(windows.astype(np.float64)).mean(-1)
    for i,(role,key,n) in enumerate([('train','training_groups',2*window+11),('train','training_groups',window+17),
            ('validation','validation_groups',window+23),('test','entities',2*window+31)]):
        prefix='g'+str(i);g=dict(group_id=prefix,array_prefix=prefix,points=n,features=2,entity_id=prefix)
        ds[key].append(g);values=rng.normal(size=(n,2)).astype(np.float32)
        data[prefix+('_test' if role=='test' else '_values')]=values
        windows,starts=covering_windows(values,window);saved[prefix+'_'+role]=align_scores(scoring(windows),starts,n,window)
        for suffix,value in [('_timestamps',np.arange(n,dtype=np.int64)),('_source_rows',np.arange(100,100+n,dtype=np.int64)),
                ('_fault_labels',np.array(['Normal']*n)),('_labels',np.zeros(n,dtype=np.uint8))]:
            data[prefix+suffix]=value;saved[prefix+suffix]=value.copy()
    return ds,data,saved,scoring


@pytest.mark.parametrize('window',[10,60,100])
def test_all_train_segments_validation_test_metadata_and_tails_roundtrip(window):
    ds,data,saved,score=grouped_fixture(window)
    payload,bindings,coverage=s.replay_groups(ds,window,score,data,saved)
    assert set(payload)==set(saved) and len(bindings)==20 and len(coverage)==4
    assert [r['role'] for r in coverage]==['train','train','validation','test']
    for key,value in payload.items():assert value.dtype==saved[key].dtype and value.tobytes()==saved[key].tobytes()


@pytest.mark.parametrize('fault',['score_tail','timestamp','dtype','missing_test_labels','extra_array','cross_role_group'])
def test_full_group_roundtrip_refuses_source_or_role_corruption(fault):
    ds,data,saved,score=grouped_fixture(100)
    if fault=='score_tail':saved['g3_test'][-1]+=0.1
    elif fault=='timestamp':saved['g3_timestamps'][-1]+=1
    elif fault=='dtype':saved['g3_test']=saved['g3_test'].astype(np.float32)
    elif fault=='missing_test_labels':del data['g3_labels']
    elif fault=='extra_array':saved['untracked']=np.ones(1)
    else:ds['validation_groups'][0]['group_id']=ds['training_groups'][0]['group_id']
    with pytest.raises(ValueError):s.replay_groups(ds,100,score,data,saved)


def test_only_explicit_native_and_review_phases_are_callable():
    for phase in ['native','review']:
        command=s.command('original_case',phase)
        assert command[-4:]==['--case','original_case','--phase',phase]
        assert '--fit' not in command
    with pytest.raises(ValueError):s.command('original_case','train')


def test_cold_full_scope_registration_retains_original_resources_and_has_no_numeric_imports():
    code=('import sys;from scripts.flow_matching import replay_industrial_full550_checkpoints_v1 as s;'
        'c=s.registration();q,m,j,e=s.verify(c);assert len(j)==550 and len(e)==550;'
        'assert c["source_cpu_threads"]==2 and c["settings"]["minimum_commit_headroom_bytes"]==32*2**30;'
        'assert c["resource_lock"]=="experiments/runs/fm_bounded_strict_baseline_cpu.lock";'
        'assert c["new_optimizer_updates"]==0 and not c["independent_historical_training_updates"];'
        'assert not any(n in sys.modules for n in ["torch","numpy","sklearn","scipy","pandas"])')
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=s.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
