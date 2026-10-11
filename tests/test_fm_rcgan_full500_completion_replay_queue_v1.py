import copy
from pathlib import Path
import numpy as np
import pytest
from scripts.flow_matching import run_rcgan_full500_completion_replay_queue_v1 as replay


def fixture_routes():
    queue=replay.read(Path(__file__).resolve().parents[1]/'configs/experiments/fm_rcgan_native_original_mnist500_queue.v2.json')
    config=dict(routes=[dict(canonical_job=copy.deepcopy(j),execution_job=copy.deepcopy(j),kind='original',
        recovery_configuration=None,result_path=(Path(j['output_directory'])/'result.json').as_posix()) for j in queue['jobs']],
        python=queue['python'],settings=copy.deepcopy(queue['settings']),resource_lock=queue['resource_lock'],
        required_epochs_per_job=500,required_generated_points_per_epoch=4984,required_generator_calls_per_epoch=178,
        full_fit_points=996,full_MMD_points=3988,equivalence_gate='bitwise',cpu_threads=2)
    return queue,config


def test_all_five_original_full_native_obligations_are_retained():
    queue,config=fixture_routes()
    assert len(replay.check_routes(queue,config))==5


@pytest.mark.parametrize('change',['missing_seed','duplicate','seed','epochs','samples','calls','fit','MMD','tolerance','threads','guards','output'])
def test_partial_changed_science_or_seed_inventory_cannot_become_completion(change):
    queue,config=fixture_routes()
    if change=='missing_seed':config['routes'].pop()
    elif change=='duplicate':config['routes'][1]=copy.deepcopy(config['routes'][0])
    elif change=='seed':config['routes'][0]['execution_job']['seed']=42
    elif change=='epochs':config['required_epochs_per_job']=1
    elif change=='samples':config['required_generated_points_per_epoch']=28
    elif change=='calls':config['required_generator_calls_per_epoch']=1
    elif change=='fit':config['full_fit_points']=1
    elif change=='MMD':config['full_MMD_points']=1
    elif change=='tolerance':config['equivalence_gate']='allclose'
    elif change=='threads':config['cpu_threads']=1
    elif change=='guards':config['settings']['emergency_commit_headroom_bytes']=0
    else:config['routes'][0]['result_path']='different.json'
    with pytest.raises(ValueError):replay.check_routes(queue,config)


def fixture_complete():
    queue,config=fixture_routes();route=config['routes'][0]
    route['successful_native_receipt_field']='full_native_result_verified'
    result=dict(job=route['execution_job'],full_training_completed=True,full_original_evaluators_completed=True,
        full_original500_epoch_loop_completed=True,independent_training_seed_slots=1,author_equivalence_certified=False,
        strict_TAB_result=False,epochs=[dict(epoch=e) for e in range(500)],
        evaluators=[dict(epoch=e,sigma_optimizer_updates=1,stopping_delta=0.,complete_generated_points=4984,
            full_fit_points=996,full_test_points=3988,native_mmd2=.1,native_ratio=2.) for e in range(500)],
        checkpoints=[dict(epoch=e,optimizer_updates=dict(D=(e+1)*1284,G=(e+1)*1284,sigma=e+1)) for e in range(500)],
        optimizer_updates=dict(D=642000,G=642000,sigma=500))
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_result_verified=True)
    return route,result,receipt


def test_complete_500epoch_metadata_is_a_prerequisite_only():
    route,result,receipt=fixture_complete();replay.complete_metadata(route,result,receipt)
    assert 'full_replay_passed' not in result


@pytest.mark.parametrize('change',['aborted','exit','unverified','epochs','evaluators','checkpoints','D','G','sigma','stopping','MMD','order','extra_seed'])
def test_failed_native_guard_partial_budget_or_evaluator_cannot_trigger_replay(change):
    route,result,receipt=fixture_complete()
    if change=='aborted':receipt['resource_guard_aborted']=True
    elif change=='exit':receipt['exit_code']=1
    elif change=='unverified':receipt['full_native_result_verified']=False
    elif change in ('epochs','evaluators','checkpoints'):result[change].pop()
    elif change in ('D','G','sigma'):result['optimizer_updates'][change]=1
    elif change=='stopping':result['evaluators'][30]['stopping_delta']=.01
    elif change=='MMD':result['evaluators'][40]['full_test_points']=1
    elif change=='order':result['checkpoints'][490]['epoch']=1
    else:result['independent_training_seed_slots']=2
    with pytest.raises(ValueError):replay.complete_metadata(route,result,receipt)


def test_one_ulp_native_mmd_difference_is_not_hidden_by_numeric_tolerance():
    replay.bitwise_float32(np.float32(1.),np.float32(1.),np)
    with pytest.raises(ValueError):replay.bitwise_float32(np.float32(1.),np.nextafter(np.float32(1.),np.float32(2.)),np)
    with pytest.raises(ValueError):replay.bitwise_float32(float('nan'),1.,np)


@pytest.mark.parametrize('change',['missing_weights','missing_RNG','multiple_meta','mixed_directory'])
def test_full_checkpoint_cannot_be_replaced_with_parameter_only_artifacts(change):
    names=['model.meta','model.index','model.data-00000-of-00001','numpy_random_state.npz','python_random_state.json']
    record=dict(artifacts=[dict(path='original/epoch_000/'+name) for name in names])
    assert replay.checkpoint_directory(record)==Path('original/epoch_000')
    if change=='missing_weights':record['artifacts'].pop(2)
    elif change=='missing_RNG':record['artifacts'].pop(3)
    elif change=='multiple_meta':record['artifacts'].append(dict(path='other/model.meta'))
    else:record['artifacts'][1]['path']='other/model.index'
    with pytest.raises(ValueError):replay.checkpoint_directory(record)


def fixture_report():
    queue,config=fixture_routes();route=config['routes'][0]
    binding=dict(binding_sha256='binding',result_sha256='result')
    report=dict(passed=True,canonical_job=route['canonical_job'],execution_job=route['execution_job'],
        binding_sha256='binding',result_sha256='result',complete_replayed_epochs=500,
        full_generator_sample_count=2492000,full_native_generator_calls=89000,complete_original_MMD_calls=500,
        additional_training_seed_slots=0,training_executed=False,
        epochs=[dict(epoch=e,complete_generated_points=4984,full_MMD_points=3988,
            all_native_model_and_optimizer_variables_restored_bitwise=True,
            all4984_generated_samples_replayed_bitwise=True,complete_original_MMD_and_ratio_replayed_bitwise=True,
            all_real_evaluator_images_belong_to_original_validation_group=True) for e in range(500)])
    receipt=dict(exit_code=0,resource_guard_aborted=False,complete_full500_native_replay_passed=True)
    return route,binding,report,receipt


def test_actual_full_scope_report_contract():
    replay.check_report(*fixture_report())


@pytest.mark.parametrize('change',['one_checkpoint','one_epoch','samples','calls','MMD','abort','different_result','missing_variable','generator','metric','group','training'])
def test_narrow_replays_and_unverified_native_arrays_never_count_as500epoch_completion(change):
    route,binding,report,receipt=fixture_report()
    if change=='one_checkpoint':report['complete_replayed_epochs']=1
    elif change=='one_epoch':report['epochs']=report['epochs'][:1]
    elif change=='samples':report['full_generator_sample_count']=4984
    elif change=='calls':report['full_native_generator_calls']=178
    elif change=='MMD':report['complete_original_MMD_calls']=1
    elif change=='abort':receipt['resource_guard_aborted']=True
    elif change=='different_result':report['result_sha256']='different'
    elif change=='missing_variable':report['epochs'][499]['all_native_model_and_optimizer_variables_restored_bitwise']=False
    elif change=='generator':report['epochs'][499]['all4984_generated_samples_replayed_bitwise']=False
    elif change=='metric':report['epochs'][499]['complete_original_MMD_and_ratio_replayed_bitwise']=False
    elif change=='group':report['epochs'][499]['all_real_evaluator_images_belong_to_original_validation_group']=False
    else:report['training_executed']=True
    with pytest.raises(ValueError):replay.check_report(route,binding,report,receipt)
