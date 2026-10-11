from copy import deepcopy
import statistics

import pytest

from scripts.flow_matching.publish_spectral_stocks_four_seed_listing_v46 import validate_native, training_statistics, check_delta, METRICS
from scripts.flow_matching.run_spectral_original_job import fingerprint
from tests.test_fm_spectral_stocks_publication_v28 import full_native
from tests.test_fm_spectral_completed_generation_v1 import original
from scripts.flow_matching.audit_spectral_completed_generation_v1 import evaluator_statistics


def fixture(seed=42):
    proof, result, receipt = full_native()
    job = dict(id=f'spectral_original__diffusion_ts__stocks__released_author_data__seed{seed}', seed=seed)
    result.update(job=job, job_sha256=fingerprint(job), artifacts=[], metrics=original())
    proof.update(canonical_id=job['id'], seed=seed, metrics=evaluator_statistics(result['metrics']),
        strict_TAB_result=False, cuda_context_initialized=False, artifacts=[])
    receipt['id'] = job['id']
    return proof, result, receipt, job


@pytest.mark.parametrize('fault', ['seed', 'budget', 'shape', 'identity', 'fingerprint', 'abort', 'partial_evaluators', 'fake_seeds', 'artifact', 'paper_equivalence'])
def test_actual_full_native_original_identity_budget_and_evaluator_units_required(fault):
    proof, result, receipt, job = fixture(45)
    validate_native(proof, result, receipt, job)
    if fault == 'seed': proof['seed'] = 44
    elif fault == 'budget': proof['original_optimizer_updates'] = 9999
    elif fault == 'shape': proof['complete_generated_shape'] = [2001, 24, 6]
    elif fault == 'identity': result['job'] = dict(job, seed=43)
    elif fault == 'fingerprint': result['job_sha256'] = 'wrong'
    elif fault == 'abort': receipt['resource_guard_aborted'] = True
    elif fault == 'partial_evaluators': proof['metrics']['context_fid']['raw_repeats'].pop()
    elif fault == 'fake_seeds': proof['metrics']['context_fid']['independent_training_models'] = 5
    elif fault == 'artifact': proof['artifacts'] = [{'role': 'different'}]
    else: proof['author_equivalence_certified'] = True
    with pytest.raises(ValueError): validate_native(proof, result, receipt, job)


def test_uncertainty_uses_four_training_means_never_twenty_evaluator_repetitions():
    proofs = [fixture(seed)[0] for seed in range(42, 46)]
    for index, proof in enumerate(proofs):
        for metric in METRICS: proof['metrics'][metric]['raw_repeats'] = [index + x / 10 for x in range(5)]
    rows = training_statistics(proofs)
    for item in rows:
        assert item['n'] == 4
        assert item['mean'] == pytest.approx(statistics.mean([.2, 1.2, 2.2, 3.2]))
        assert item['std'] == pytest.approx(statistics.stdev([.2, 1.2, 2.2, 3.2]))
    with pytest.raises(ValueError): training_statistics(proofs + [proofs[0]])
    with pytest.raises(ValueError): training_statistics(proofs[:3])
    wrong = deepcopy(proofs); wrong[3]['seed'] = 44
    with pytest.raises(ValueError): training_statistics(wrong)


def test_every_other_row_timestamp_and_original_protocol_is_preserved():
    before = [dict(task='unconditional_time_series_generation', algorithm='Spectral-author/diffusion_ts', dataset='stocks',
        protocol='released_author_data', metric=metric, n='1', required_repeats='5', mean='.1', paper_mean='.2',
        captured_utc='old', evaluator_repeats='5') for metric in METRICS]
    before.append(dict(task='anomaly_detection', algorithm='FM', dataset='MSL', protocol='original',
        metric='f1', n='1', required_repeats='5', mean='.5', captured_utc='old-other'))
    after = deepcopy(before)
    for row in after[:4]: row.update(n='4', mean='.12', captured_utc='new')
    assert check_delta(before, after) == [0, 1, 2, 3]
    for fault in ['other_timestamp', 'paper_value', 'n20', 'required4', 'protocol', 'lost_row']:
        bad = deepcopy(after)
        if fault == 'other_timestamp': bad[-1]['captured_utc'] = 'new'
        elif fault == 'paper_value': bad[0]['paper_mean'] = '.1'
        elif fault == 'n20': bad[0]['n'] = '20'
        elif fault == 'required4': bad[0]['required_repeats'] = '4'
        elif fault == 'protocol': bad[0]['protocol'] = 'strictTAB'
        else: bad.pop()
        with pytest.raises(ValueError): check_delta(before, bad)
