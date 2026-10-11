from copy import deepcopy

import pytest

from scripts.flow_matching.verify_saits_completed_full_replay_batch_v1 import check_terminal,verify_record_identity


def terminal():
    ids=[f'original_model_{i}' for i in range(11)]
    producer=dict(completed_jobs=[dict(id=i) for i in ids],full_model_count=11)
    replay=dict(records=[dict(id=i) for i in ids],full_models_replayed=11,
        full_checkpoint_contents_and_re_inference_passed=True,cuda_context_initialized=False,additional_training_seed_slots=0)
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_native_model_replay_passed=True)
    return replay,receipt,producer


def test_all_eleven_original_models_can_finish_in_any_execution_order():
    replay,receipt,producer=terminal()
    replay['records'].reverse()
    check_terminal(replay,receipt,producer)


@pytest.mark.parametrize('change',['missing','duplicate','extra','count','cuda','failed_report','additional_seed',
    'exit_code','resource_abort','failed_receipt','duplicate_registration'])
def test_no_partial_retry_or_aborted_batch_can_satisfy_original_scope(change):
    replay,receipt,producer=terminal()
    if change=='missing':replay['records'].pop()
    elif change=='duplicate':replay['records'][-1]=deepcopy(replay['records'][0])
    elif change=='extra':replay['records'].append(dict(id='extra_retry'))
    elif change=='count':replay['full_models_replayed']=10
    elif change=='cuda':replay['cuda_context_initialized']=True
    elif change=='failed_report':replay['full_checkpoint_contents_and_re_inference_passed']=False
    elif change=='additional_seed':replay['additional_training_seed_slots']=1
    elif change=='exit_code':receipt['exit_code']=1
    elif change=='resource_abort':receipt['resource_guard_aborted']=True
    elif change=='failed_receipt':receipt['full_native_model_replay_passed']=False
    elif change=='duplicate_registration':producer['completed_jobs'][-1]=deepcopy(producer['completed_jobs'][0])
    with pytest.raises(ValueError):check_terminal(replay,receipt,producer)


def identities():
    native=dict(id='original',seed=26,bootstrap_group='month',diagnostic=False,dataset_sha256='original_data',
        method='saits',dataset='electricity',missing_ratio=.2)
    registered=dict(id='original',config='original_config')
    result=dict(config=deepcopy(native),status='completed',parameters=11511000,target_count=1944331,validation_best=dict(epoch=386))
    record=dict(id='original',seed=26,config='original_config',model_parameters=11511000,
        held_out_coordinates=1944331,original_full_dataset_sha256='original_data',validation_best=dict(epoch=386))
    return record,native,result,registered


def test_original_full_selected_model_identity_is_preserved():
    verify_record_identity(*identities())


@pytest.mark.parametrize('change',['seed','parameter_count','coordinates','selected_epoch','data','config_path',
    'method','missing_ratio','diagnostic','shortened','month_group'])
def test_model_replacement_or_shortened_test_grid_cannot_satisfy_full_original_identity(change):
    record,native,result,registered=identities()
    if change=='seed':record['seed']=27
    elif change=='parameter_count':record['model_parameters']=100
    elif change=='coordinates':record['held_out_coordinates']=100
    elif change=='selected_epoch':record['validation_best']=dict(epoch=1)
    elif change=='data':record['original_full_dataset_sha256']='changed'
    elif change=='config_path':record['config']='changed'
    elif change=='method':native['method']='substitute'
    elif change=='missing_ratio':native['missing_ratio']=.1
    elif change=='diagnostic':native['diagnostic']=True;result['config']=deepcopy(native)
    elif change=='shortened':native['limit_test_cases']=5;result['config']=deepcopy(native)
    elif change=='month_group':native['bootstrap_group']='window';result['config']=deepcopy(native)
    with pytest.raises(ValueError):verify_record_identity(record,native,result,registered)
