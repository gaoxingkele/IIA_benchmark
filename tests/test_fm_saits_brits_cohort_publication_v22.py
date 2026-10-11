import copy

import pytest

from scripts.flow_matching.publish_saits_brits_full_listing_v22 import validated_cohorts, audited_updates


def proofs():
    rows = []
    for method, seeds in [('saits', range(26, 31)), ('brits', [26])]:
        for seed in seeds:
            metric = dict(mae=1., rmse=2., mre=.5)
            rows.append(dict(id='saits_paper_%s_electricity_author_missing0.1_seed%d' % (method, seed),
                seed=seed, model_parameters=11511000 if method == 'saits' else 6993328,
                full_test_shape=[290, 100, 370], held_out_coordinates=1020954,
                full_original_checkpoint_and_data_identity_verified=True,
                full_test_grid_and_grouped_statistics_replayed=True,
                full_training_completion=dict(full_training_finished=True, benchmark_smoke=False),
                original_metrics=metric, replayed_metrics=metric, independently_recomputed_metrics=metric,
                original_full_dataset_sha256='same-frozen-dataset'))
    return [dict(full_checkpoint_contents_and_re_inference_passed=True, full_models_replayed=6,
        additional_training_seed_slots=0, cuda_context_initialized=False, records=rows)]


@pytest.mark.parametrize('corruption', ['duplicate', 'partial_test', 'unfinished', 'different_data', 'nonfinite', 'wrong_model'])
def test_incomplete_or_inconsistent_original_cohorts_cannot_publish(corruption):
    data = copy.deepcopy(proofs())
    row = data[0]['records'][-1]
    if corruption == 'duplicate':
        data[0]['records'].append(copy.deepcopy(row)); data[0]['full_models_replayed'] += 1
    elif corruption == 'partial_test':
        row['full_test_shape'][0] -= 1
    elif corruption == 'unfinished':
        row['full_training_completion']['full_training_finished'] = False
    elif corruption == 'different_data':
        row['original_full_dataset_sha256'] = 'another-dataset'
    elif corruption == 'nonfinite':
        row['original_metrics']['mae'] = float('nan')
    elif corruption == 'wrong_model':
        row['model_parameters'] = 11511000
    with pytest.raises(ValueError):
        validated_cohorts(data)


def test_single_original_brits_seed_has_no_invented_seed_uncertainty():
    original = [dict(algorithm=method, dataset='electricity', task='imputation', protocol='author',
        pattern='paper_mask', missing_ratio='0.1', metric=metric, required_repeats='5')
        for method in ['saits', 'brits'] for metric in ['mae', 'rmse', 'mre']]
    updates = audited_updates(original, proofs(), 'proof.json', 'frozen-time')
    for row in updates:
        assert row['n'] == (5 if row['algorithm'] == 'saits' else 1)
        assert row['author_equivalence_certified'] is False
        assert row['strict_TAB_result'] is False
        if row['algorithm'] == 'brits':
            assert all(row[key] is None for key in ['std', 'se', 'ci95_low', 'ci95_high'])
