import copy
import statistics

import pytest

from scripts.flow_matching.audit_spectral_completed_generation_v1 import METRICS
from scripts.flow_matching.summarize_spectral_full_cohorts_v1 import summarize_cohort


def proofs():
    return [dict(seed=seed, model='diffusion_ts', dataset='energy', data_track='released_author_data',
        passed=True, actual_full_native_model_count=1, cuda_context_initialized=False,
        metrics={key: dict(raw_repeats=[seed + k for k in range(5)], mean=seed + 2) for key in METRICS})
        for seed in range(42, 47)]


def test_training_model_uncertainty_uses_five_models_not_twenty_five_evaluator_runs():
    summary = summarize_cohort(proofs())
    for metric in summary['metrics'].values():
        assert metric['independent_training_models'] == 5 and metric['evaluator_runs'] == 25
        assert metric['degrees_of_freedom'] == 4 and metric['full_cohort_mean'] == 46
        assert metric['between_training_model_sample_STD'] == statistics.stdev([44, 45, 46, 47, 48])


def test_four_models_do_not_produce_surviving_seed_full_cohort_statistics():
    summary = summarize_cohort([p for p in proofs() if p['seed'] != 43])
    assert summary['missing_original_training_seeds'] == [43]
    assert not summary['full_five_seed_cohort_complete']
    for metric in summary['metrics'].values():
        assert len(metric['per_training_model']) == 4
        assert metric['full_cohort_mean'] is None
        assert metric['between_training_model_sample_STD'] is None
        assert metric['between_training_model_CI95'] is None


@pytest.mark.parametrize('fault', ['duplicate', 'mixed', 'failed', 'gpu', 'partial', 'nonfinite', 'mean', 'missing_metric'])
def test_mixed_or_incomplete_native_evidence_cannot_enter_summary(fault):
    rows = copy.deepcopy(proofs())
    if fault == 'duplicate': rows[1]['seed'] = rows[0]['seed']
    elif fault == 'mixed': rows[0]['dataset'] = 'etth'
    elif fault == 'failed': rows[0]['passed'] = False
    elif fault == 'gpu': rows[0]['cuda_context_initialized'] = True
    elif fault == 'missing_metric': rows[0]['metrics'].pop('context_fid')
    elif fault == 'partial': rows[0]['metrics']['context_fid']['raw_repeats'].pop()
    elif fault == 'nonfinite': rows[0]['metrics']['context_fid']['raw_repeats'][0] = float('inf')
    elif fault == 'mean': rows[0]['metrics']['context_fid']['mean'] += 1
    with pytest.raises(ValueError):
        summarize_cohort(rows)
