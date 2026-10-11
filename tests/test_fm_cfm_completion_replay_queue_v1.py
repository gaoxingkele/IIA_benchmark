import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.flow_matching import run_cfm_completion_replay_queue_v1 as m


def test_recovery_may_change_execution_directories_but_no_scientific_field():
    original=dict(id='canonical',seed=1103,epochs=400,expected_optimizer_updates=400,
        test_points_per_trajectory=500,training_items=258,batch_size=258,model_overrides={'grid_points':160},
        output_directory='old',artifact_directory='old-artifacts')
    recovery=dict(original,id='canonical_retry',recovery_original_job_id='canonical',
        output_directory='new',artifact_directory='new-artifacts')
    m.check_binding(original,recovery)
    for field in ['seed','epochs','expected_optimizer_updates','test_points_per_trajectory','training_items','batch_size']:
        bad=copy.deepcopy(recovery);bad[field]+=1
        with pytest.raises(ValueError):m.check_binding(original,bad)
    bad=copy.deepcopy(recovery);bad['model_overrides']['grid_points']=159
    with pytest.raises(ValueError):m.check_binding(original,bad)


def terminal_fixture():
    item=dict(original_canonical_id='canonical',result_path='result',result_sha256='abc',job={'id':'retry'})
    config=dict(full_results=[item],cpu_threads=2)
    row=dict(original_canonical_id='canonical',result_path='result',result_sha256='abc',execution_id='retry')
    report=dict(full_checkpoint_contents_and_re_inference_passed=True,prediction_tolerance=m.TOLERANCE.copy(),
        cuda_context_initialized=False,additional_training_seed_slots=0,cpu_threads=2,full_job_count=1,records=[row])
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_replay_passed=True)
    return config,report,receipt


@pytest.mark.parametrize('changed',['exit','guard','passed','tolerance','cuda','seed','threads','missing','duplicate','execution','result'])
def test_actual_terminal_full_scope_and_immutable_native_binding_required(changed):
    config,report,receipt=terminal_fixture();m.check_native_terminal(config,report,receipt)
    if changed=='exit':receipt['exit_code']=1
    elif changed=='guard':receipt['resource_guard_aborted']=True
    elif changed=='passed':receipt['full_replay_passed']=False
    elif changed=='tolerance':report['prediction_tolerance']['atol']=1e-6
    elif changed=='cuda':report['cuda_context_initialized']=True
    elif changed=='seed':report['additional_training_seed_slots']=1
    elif changed=='threads':report['cpu_threads']=1
    elif changed=='missing':report['records']=[]
    elif changed=='duplicate':report['records']*=2
    elif changed=='execution':report['records'][0]['execution_id']='another_seed'
    elif changed=='result':report['records'][0]['result_sha256']='other'
    with pytest.raises(ValueError):m.check_native_terminal(config,report,receipt)


def array_fixture(tmp_path,monkeypatch):
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import prediction_metrics
    monkeypatch.setattr(m,'ROOT',tmp_path)
    job=dict(id='canonical',seed=1103,epochs=2,expected_optimizer_updates=2,training_items=1,
        test_trajectories=2,test_points_per_trajectory=500,observed_dimensions=1,
        data_path='data.npz',model_config='model.json')
    ids=np.array([0,1]);times=np.tile(np.linspace(.01,5.,500),(2,1));initial=np.array([[.1],[.2]])
    truth=np.sin(times)[...,None];prediction=truth+.03
    metrics,per=prediction_metrics(prediction,truth)
    np.savez(tmp_path/'data.npz',train_ids=np.array([0]),test_ids=ids,test_states=truth,test_times=times,initial_states=initial)
    (tmp_path/'model.json').write_text('{}');(tmp_path/'model.pt').write_bytes(b'fixture-checkpoint')
    job.update(data_sha256=m.sha(tmp_path/'data.npz'),model_config_sha256=m.sha(tmp_path/'model.json'))
    saved=dict(predictions=prediction,truth=truth,times=times,test_ids=ids,initial_states=initial,per_trajectory_MSE=per)
    np.savez(tmp_path/'predictions.npz',**saved)
    reproduced={k:saved[k].copy() for k in ['predictions','truth','times','test_ids']}
    np.savez(tmp_path/'replay.npz',**reproduced)
    result=dict(status='completed',job=job,epochs_completed=2,optimizer_updates=2,training_items=1,
        training_losses=[1.,.5],training_trajectory_ids=[0],test_trajectory_ids=[0,1],metrics=metrics,
        artifacts=[dict(path=str(tmp_path/name),sha256=m.sha(tmp_path/name)) for name in ['model.pt','predictions.npz']])
    (tmp_path/'result.json').write_text(json.dumps(result))
    canonical=dict(original_canonical_id='canonical',result_path='result.json',result_sha256=m.sha(tmp_path/'result.json'),job=job)
    replay=dict(original_canonical_id='canonical',result_path='result.json',result_sha256=canonical['result_sha256'],
        checkpoint_path=str(tmp_path/'model.pt'),checkpoint_sha256=m.sha(tmp_path/'model.pt'),
        replay_arrays=str(tmp_path/'replay.npz'),replay_arrays_sha256=m.sha(tmp_path/'replay.npz'),
        original_full_data_sha256=job['data_sha256'],epochs=2,optimizer_updates=2,execution_id='canonical',seed=1103,
        checkpoint_contents_and_full_budget_verified=True,original_full_grid_replayed=True,
        full_replay_shape=[2,500,1],bitwise_identical=True,max_absolute_difference=0.,replayed_metrics=metrics)
    return canonical,replay,reproduced


@pytest.mark.parametrize('changed',[None,'truncated','last_point','truth','times','ids','nan','metric','budget','checkpoint','tolerance','max_difference'])
def test_independent_review_checks_every_original_coordinate_and_its_provenance(tmp_path,monkeypatch,changed):
    canonical,replay,arrays=array_fixture(tmp_path,monkeypatch);config=dict(prediction_tolerance=m.TOLERANCE.copy())
    if changed is None:
        proof=m.verify_record(config,canonical,replay)
        assert proof['all_original_grid_coordinates_independently_checked'];return
    if changed=='truncated':arrays['predictions']=arrays['predictions'][:,:-1]
    elif changed=='last_point':arrays['predictions'][-1,-1,0]+=.01
    elif changed=='truth':arrays['truth'][-1,-1,0]+=.01
    elif changed=='times':arrays['times'][-1,-1]+=.01
    elif changed=='ids':arrays['test_ids']=arrays['test_ids'][::-1]
    elif changed=='nan':arrays['predictions'][-1,-1,0]=np.nan
    elif changed=='metric':replay['replayed_metrics']['MSE']+=.01
    elif changed=='budget':replay['optimizer_updates']=1
    elif changed=='checkpoint':(tmp_path/'model.pt').write_bytes(b'another-checkpoint')
    elif changed=='tolerance':config['prediction_tolerance']['atol']=1.
    elif changed=='max_difference':replay['max_absolute_difference']=1.
    if changed in ['truncated','last_point','truth','times','ids','nan']:
        np.savez(tmp_path/'replay.npz',**arrays);replay['replay_arrays_sha256']=m.sha(tmp_path/'replay.npz')
    with pytest.raises(ValueError):m.verify_record(config,canonical,replay)


def test_exclusive_proofs_are_never_overwritten(tmp_path):
    path=tmp_path/'proof.json';m.exclusive(path,dict(passed=False))
    with pytest.raises(FileExistsError):m.exclusive(path,dict(passed=True))
    assert json.loads(path.read_text())==dict(passed=False)


def test_registration_never_imports_numerical_stack_or_runs_training():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    top_imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('numpy' in ast.unparse(n) or 'torch' in ast.unparse(n) for n in top_imports)
    dispatch=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller'))
    assert 'run_cfm_ts_original_job' not in dispatch
    assert 'audit_cfm_checkpoint_replay_v1' in dispatch
