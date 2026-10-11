from copy import deepcopy

import pytest

from scripts.flow_matching import verify_grasp_completed_full_replay_v1 as audit


def fixture(monkeypatch,parameters):
    original=dict(status='completed',epochs_completed=1500,optimizer_updates=93000,
        parameter_count=parameters,job=dict(expected_optimizer_updates=93000),metrics=[{}])
    monkeypatch.setattr(audit.base,'read',lambda _: original)
    config=dict(full_job_count=1,full_results=[dict(canonical_id='original',result_path='result.json',result_sha256='sha')])
    record=dict(canonical_id='original',epochs_completed=1500,optimizer_updates=93000,
        parameter_count=parameters,full_checkpoint_restoration_and_reinference_passed=True,result_sha256='sha',recipes=[{}])
    report=dict(records=[record],full_checkpoint_contents_and_re_inference_passed=True,full_job_count=1,
        additional_training_seed_slots=0,cuda_context_initialized=False)
    receipt=dict(exit_code=0,resource_guard_aborted=False,full_replay_passed=True)
    return report,receipt,config


@pytest.mark.parametrize('parameters',[69244,125372])
def test_completed_replay_binds_actual_original_model_not_lin_parameter_constant(monkeypatch,parameters):
    audit.check_terminal(*fixture(monkeypatch,parameters))


@pytest.mark.parametrize('key,value',[('epochs_completed',1499),('optimizer_updates',92999),
    ('parameter_count',1),('result_sha256','different')])
def test_partial_or_wrong_model_replay_cannot_certify_completion(monkeypatch,key,value):
    report,receipt,config=fixture(monkeypatch,125372)
    report['records'][0][key]=value
    with pytest.raises(ValueError,match='Each full original model'):
        audit.check_terminal(report,receipt,config)


def test_duplicate_native_models_or_resource_abort_cannot_certify_completion(monkeypatch):
    report,receipt,config=fixture(monkeypatch,69244)
    report['records'].append(deepcopy(report['records'][0]))
    with pytest.raises(ValueError):
        audit.check_terminal(report,receipt,config)
    report,receipt,config=fixture(monkeypatch,69244)
    receipt['resource_guard_aborted']=True
    with pytest.raises(ValueError):
        audit.check_terminal(report,receipt,config)
