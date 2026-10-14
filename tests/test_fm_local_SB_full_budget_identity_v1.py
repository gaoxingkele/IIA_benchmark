import copy
import hashlib
import json

import pytest

from scripts.flow_matching.audit_local_SB_full_checkpoint_scores_v1 import validate_budget


@pytest.mark.parametrize('fault', ['epochs', 'partial_losses', 'nonfinite_loss', 'job', 'MC', 'grid', 'guard', 'budget'])
def test_reject_shortened_or_changed_original_model_scoring_obligation(fault):
    job = dict(id='original')
    config = dict(parameters=dict(objective='sb', epochs=20, monte_carlo_samples=4, score_times=[.1,.3,.5,.7,.9]))
    result = dict(id='original', job=copy.deepcopy(job), status='completed', purpose='full_local_tsad_benchmark',
        experiment_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(), training_losses=[1.] * 20)
    guard = dict(job_id='original', exit_code=0, resource_guard_aborted=False, full_registered_budget_retained=True)
    validate_budget(job, result, config, guard)
    if fault == 'epochs': config['parameters']['epochs'] = 1
    elif fault == 'partial_losses': result['training_losses'].pop()
    elif fault == 'nonfinite_loss': result['training_losses'][0] = float('nan')
    elif fault == 'job': result['job']['id'] = 'other'
    elif fault == 'MC': config['parameters']['monte_carlo_samples'] = 1
    elif fault == 'grid': config['parameters']['score_times'] = [.5]
    elif fault == 'guard': guard['resource_guard_aborted'] = True
    elif fault == 'budget': guard['full_registered_budget_retained'] = False
    with pytest.raises(ValueError): validate_budget(job, result, config, guard)
