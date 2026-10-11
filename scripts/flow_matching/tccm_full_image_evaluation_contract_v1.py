"""Full50-epoch image-domain TCCM identity; no original best-domain claim."""
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts
from scripts.flow_matching.tccm_full_image_training_contract_v1 import validate_tccm_job

SCORE_PARAMETERS = dict(test_time=1., channel_residual='unsquared_L2', top_fraction=.01)


def validate_queue(queue):
    protocol = read(ROOT / queue['original_protocol'])
    rows = protocol['Table2_same_backbone_FM_comparison']
    index = next(i for i, row in enumerate(rows) if row['method'] == 'TCCM')
    row = rows[index]
    expected = set(range(1103,1108))
    if (len(queue['cases']) != 5 or {x['seed'] for x in queue['cases']} != expected
            or len({x['id'] for x in queue['cases']}) != 5):
        raise ValueError('All five full TCCM controlled-image seeds required')
    for case in queue['cases']:
        if (case['paper_table_key'] != 'Table2_same_backbone_FM_comparison'
                or case['paper_row_index'] != index or case['paper_reported'] != row['paper_reported']
                or case['dataset'] != 'mvtec' or case['original_dataset_name'] != row['dataset']
                or case['variant'] != 'conditional' or case['checkpoint_epoch'] != 50
                or case['score_parameters'] != SCORE_PARAMETERS):
            raise ValueError('Original Eq20/time1/50epoch full TCCM recipe differs')
        job = read(ROOT / case['training_configuration'])
        validate_tccm_job(job, read(ROOT/job['model_configuration']),
            read(ROOT/job['optimizer_configuration']), read(ROOT/job['dataset_configuration']))
        if (job['seed'] != case['seed']
                or case['checkpoint'] != job['output_directory'] + '/epoch_0050.pt'):
            raise ValueError('Full native TCCM checkpoint identity differs')
    return True


def complete(case):
    path = ROOT / case['output_directory'] / 'fully_verified_score_result.json'
    if not path.exists():
        return False
    location = read(path)
    if sha(location['report']) != location['report_sha256'] or location.get('passed') is not True:
        raise ValueError('Complete immutable independent TCCM audit required')
    report = read(location['report'])
    required = ('passed', 'every_native_image_model_replayed',
        'all_native_heatmaps_and_image_scores_equal', 'original_full_state_dictionary_loaded_strictly',
        'independent_all_category_metrics_passed', 'native_performance_measured')
    if (any(report.get(k) is not True for k in required) or report['case'] != case
            or report['complete_original_test_images'] != 1725
            or report['complete_original_categories'] != 15
            or report['original_author_equivalence_certified']
            or report['original_TCCM_best_domain_setting_reproduced'] or report['strict_TAB_result']
            or sha(ROOT/case['checkpoint']) != report['full_original_checkpoint_sha256']
            or report['queue_sha256'] != sha(ROOT/report['queue_configuration'])):
        raise ValueError('Every native original image/category/checkpoint/audit proof required')
    candidate_path = report['native_score_candidate']
    if sha(candidate_path) != report['native_score_candidate_sha256']:
        raise ValueError('Full native TCCM score candidate changed')
    candidate = read(candidate_path)
    if (candidate['case'] != case or candidate['complete_original_test_images'] != 1725
            or candidate['training_binding'] != report['training_binding']
            or candidate['original_checkpoint_sha256'] != report['full_original_checkpoint_sha256']):
        raise ValueError('Full native TCCM score/checkpoint binding differs')
    verify_receipts(candidate['output_receipts']); verify_receipts(candidate['native_truth_array_receipts'])
    return True
