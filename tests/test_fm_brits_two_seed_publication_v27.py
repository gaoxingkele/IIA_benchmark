import copy

import pytest

from scripts.flow_matching.publish_brits_two_seed_listing_v27 import check_increment, validated_cohort


def proof():
    records = []
    for seed in [26, 27]:
        metrics = dict(mae=.8, rmse=1.2, mre=.4)
        records.append(dict(id='saits_paper_brits_electricity_author_missing0.1_seed%d' % seed,
            seed=seed, model_parameters=6993328, full_test_shape=[290, 100, 370],
            held_out_coordinates=1020954, full_original_checkpoint_and_data_identity_verified=True,
            full_test_grid_and_grouped_statistics_replayed=True,
            full_training_completion=dict(full_training_finished=True, benchmark_smoke=False),
            original_metrics=metrics, replayed_metrics=dict(metrics),
            independently_recomputed_metrics=dict(metrics), original_full_dataset_sha256='frozen-full-dataset'))
    return dict(full_checkpoint_contents_and_re_inference_passed=True, full_models_replayed=2,
        additional_training_seed_slots=0, cuda_context_initialized=False, records=records)


@pytest.mark.parametrize('fault', ['duplicate', 'partial', 'new_seed', 'capacity', 'tail', 'nonfinite', 'dataset', 'cuda'])
def test_only_two_actual_original_complete_model_seeds_are_counted(fault):
    p = proof()
    assert [r['seed'] for r in validated_cohort([p], [26, 27])] == [26, 27]
    r = p['records'][1]
    if fault == 'duplicate':
        p['records'][1] = copy.deepcopy(p['records'][0])
    elif fault == 'partial':
        r['full_training_completion']['full_training_finished'] = False
    elif fault == 'new_seed':
        r['seed'] = 28
    elif fault == 'capacity':
        r['model_parameters'] = 1000
    elif fault == 'tail':
        r['full_test_shape'][0] = 289
    elif fault == 'nonfinite':
        r['replayed_metrics']['mae'] = float('nan')
    elif fault == 'dataset':
        r['original_full_dataset_sha256'] = 'different'
    else:
        p['cuda_context_initialized'] = True
    with pytest.raises(ValueError):
        validated_cohort([p], [26, 27])


def test_unrelated_metric_change_or_lost_row_is_rejected():
    base = [dict(task='imputation', algorithm='brits', dataset='electricity', protocol='author',
        metric=metric, mean='1') for metric in ['mae', 'rmse', 'mre']]
    base.append(dict(task='anomaly_detection', algorithm='original-other', dataset='SMD',
        protocol='validation', metric='f1', mean='.3'))
    new = copy.deepcopy(base)
    for row in new[:3]:
        row['mean'] = '.8'
    assert check_increment(base, new, new[:3]) == [0, 1, 2]
    bad = copy.deepcopy(new)
    bad[3]['mean'] = '.9'
    with pytest.raises(ValueError):
        check_increment(base, bad, new[:3])
    with pytest.raises(ValueError):
        check_increment(base, new[:-1], new[:3])
