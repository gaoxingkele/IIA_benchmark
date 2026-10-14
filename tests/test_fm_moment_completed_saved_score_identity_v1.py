import copy
import hashlib
import json

import pytest

from scripts.flow_matching.audit_moment_completed_saved_scores_v1 import validate_identity


def inputs():
    job = dict(id='original', window=512)
    load = dict(parameter_count=109641608, checkpoint_sha256='weights', config_sha256='config',
        configured_sequence_length=512, task='reconstruction', all_keys_strictly_loaded=True,
        random_reinitialized_head=False, hub_downloads=False, fine_tuning_epochs=0, fit_points_used_for_parameter_estimation=0)
    result = dict(id='original', job=copy.deepcopy(job), status='completed', purpose='full_frozen_pretrained_zero_shot_tsad',
        experiment_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        training_seconds=0, training_losses=[], parameter_count=109641608, checkpoint_load_audit=load)
    guard = dict(job_id='original', exit_code=0, resource_guard_aborted=False, full_registered_budget_retained=True)
    return job, result, guard, dict(checkpoint_sha256='weights', config_sha256='config')


@pytest.mark.parametrize('fault', ['job', 'status', 'weights', 'context', 'random_head', 'fitting', 'guard', 'budget'])
def test_partial_or_changed_pretrained_runs_cannot_be_reported_complete(fault):
    job, result, guard, parameters = inputs()
    validate_identity(job, result, guard, parameters)
    if fault == 'job': result['job']['window'] = 256
    elif fault == 'status': result['status'] = 'partial'
    elif fault == 'weights': result['checkpoint_load_audit']['checkpoint_sha256'] = 'other'
    elif fault == 'context': result['checkpoint_load_audit']['configured_sequence_length'] = 256
    elif fault == 'random_head': result['checkpoint_load_audit']['random_reinitialized_head'] = True
    elif fault == 'fitting': result['checkpoint_load_audit']['fine_tuning_epochs'] = 1
    elif fault == 'guard': guard['resource_guard_aborted'] = True
    elif fault == 'budget': guard['full_registered_budget_retained'] = False
    with pytest.raises(ValueError): validate_identity(job, result, guard, parameters)
