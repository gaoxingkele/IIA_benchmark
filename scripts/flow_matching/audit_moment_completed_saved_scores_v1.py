"""Guarded producer-environment audit of complete frozen MOMENT score arrays."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def validate_identity(job, result, guard, parameters):
    load = result['checkpoint_load_audit']
    if (result['id'] != job['id'] or result['job'] != job or result['status'] != 'completed'
            or result['purpose'] != 'full_frozen_pretrained_zero_shot_tsad'
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            or result['training_seconds'] != 0 or result['training_losses'] != []
            or result['parameter_count'] != load['parameter_count']
            or load['checkpoint_sha256'] != parameters['checkpoint_sha256']
            or load['config_sha256'] != parameters['config_sha256']
            or load['configured_sequence_length'] != job['window']
            or load['task'] != 'reconstruction' or not load['all_keys_strictly_loaded']
            or load['random_reinitialized_head'] or load['hub_downloads']
            or load['fine_tuning_epochs'] != 0 or load['fit_points_used_for_parameter_estimation'] != 0
            or guard['job_id'] != job['id'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or not guard['full_registered_budget_retained']):
        raise ValueError('Complete original fixed pretrained weights, context and successful guarded execution required')


def worker(config, config_path):
    import platform
    import numpy as np
    import sklearn
    import torch
    from scripts.flow_matching.run_tsad_queue import complete_result
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
    from scripts.flow_matching.publish_full_range_increment_v25 import entity_buffers, verify_macros
    from iia_benchmark.evaluation.mtsad_metrics import point_adjust, pointwise_metrics
    queue = read(ROOT / config['queue'])
    job = next(item for item in queue['jobs'] if item['id'] == config['job_id'])
    if not complete_result(ROOT, job):
        raise ValueError('Full original checkpoint/result/score hashes must verify')
    for pin in config['source_receipts'] + job['frozen_files']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen source, asset or data changed: ' + pin['path'])
    source = ROOT / job['output_directory'] / 'result.json'
    result = read(source)
    model_config = read(ROOT / job['model_config'])
    guard = read(ROOT / config['terminal_receipt'])
    validate_identity(job, result, guard, model_config['parameters'])
    environment = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device='cpu')
    if environment != result['environment']:
        raise ValueError('Use the actual producer Python/NumPy/Torch environment without widening tolerances')
    checkpoint = torch.load(source.parent / 'model.pt', map_location='cpu', weights_only=True)
    if (checkpoint['checkpoint_load_audit'] != result['checkpoint_load_audit']
            or checkpoint['configuration'] != model_config
            or checkpoint['overrides'] != job['parameter_overrides']):
        raise ValueError('Full frozen weight reference and recorded load identity must agree')
    asset = ROOT / model_config['parameters']['checkpoint_directory']
    weight_files = [pin for pin in job['frozen_files'] if pin['sha256'] == model_config['parameters']['checkpoint_sha256']]
    if len(weight_files) != 1:
        raise ValueError('Exactly one frozen full pretrained weight file required')
    if (sha(ROOT / weight_files[0]['path']) != model_config['parameters']['checkpoint_sha256']
            or sha(asset / 'config.json') != model_config['parameters']['config_sha256']):
        raise ValueError('Actual complete official pretrained weights/config must remain unchanged')
    manifest = read(ROOT / job['dataset_manifest_path'])
    dataset = next(item for item in manifest['datasets'] if item['id'] == job['dataset'])
    if sha(ROOT / dataset['input_npz']) != dataset['input_sha256'] or not dataset['grouped_splits_verified']:
        raise ValueError('Complete frozen grouped dataset required')
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as inputs, np.load(source.parent / 'scores.npz', allow_pickle=False) as scores:
        entities, lengths, validation = check_arrays(inputs, scores, dataset)
        training = np.concatenate([scores[entry['array_prefix'] + '_train'] for entry in dataset['entities']])
        coverage = {(row['group'], row['role']): row for row in result['native_coverage']}
        if len(coverage) != len(dataset['entities']) * 3:
            raise ValueError('Every original train/validation/test entity segment required')
        for entry in dataset['entities']:
            for split in ('train', 'validation', 'test'):
                row = coverage[(entry['entity_id'], split)]
                if row['native_points'] != len(inputs[entry['array_prefix'] + '_' + split]) or not row['all_context_tokens_observed']:
                    raise ValueError('Every native timestamp and full observed context required')
        settings = dict(queue['evaluation'], bootstrap_block_points=max(queue['evaluation']['bootstrap_block_points'], job['window']))
        actual = evaluate_saved(entities, validation, training, dataset['input_sha256'], result['checkpoint_sha256'], settings, job['seed'])
        if actual != result['metrics']:
            raise ValueError('Every primary/core diagnostic metric and grouped uncertainty must recompute exactly')
        truth = np.concatenate([pair[0] for pair in entities.values()])
        native_scores = np.concatenate([pair[1] for pair in entities.values()])
        threshold = actual['strict']['1']['calibration']['threshold']
        adjusted = np.concatenate([point_adjust(labels, scores > threshold) for labels, scores in entities.values()])
        controls = dict(validation_threshold=threshold,
            validation_only_no_PA=pointwise_metrics(truth, native_scores > threshold),
            validation_only_same_threshold_with_PA=pointwise_metrics(truth, adjusted),
            test_pool_calibration_ratio1=next(row for row in actual['tab_core_diagnostic']['ratio_grid'] if row['ratio_percent'] == 1),
            best_test_label_selected_plain_grid_F1=actual['tab_core_diagnostic']['best_grid_f_score'],
            best_test_label_selected_PA_grid_F1=actual['tab_core_diagnostic']['best_grid_adjust_f_score'],
            all_controls_use_identical_frozen_score_arrays=True,
            primary_threshold_selected_using_test_features_or_labels=False,
            PA_control_uses_test_ground_truth_events=True)
        range_path = ROOT / config['range_evaluation']
        ranged = read(range_path)
        science = read(ROOT / config['range_config'])
        science_hash = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
        if (ranged['status'] != 'completed_range_evaluation' or ranged['id'] != job['id']
                or ranged['model_result_sha256'] != sha(source) or ranged['scores_sha256'] != result['scores_sha256']
                or ranged['evaluation_config_sha256'] != science_hash or ranged['source_receipts'] != science['source_receipts']
                or ranged['evaluation_identity'] != hashlib.sha256((sha(source) + science_hash).encode()).hexdigest()
                or range_path.with_name('evaluation.sha256').read_text(encoding='ascii').strip() != sha(range_path)
                or set(ranged['VUS_per_entity']) != set(entities)
                or set(ranged['strict_affiliation_per_entity']) != set(entities)):
            raise ValueError('Full original range evaluation must bind exact arrays and scientific configuration')
        for pin in science['source_receipts']:
            if sha(ROOT / pin['path']) != pin['sha256']:
                raise ValueError('Original complete TAB range source changed')
        range_guard = read(ROOT / config['range_resource_receipt'])
        range_state = read(ROOT / config['range_controller_state'])
        if range_guard['resource_guard_aborted'] or range_state['completed'][job['id']]['evaluation_identity'] != ranged['evaluation_identity']:
            raise ValueError('Actual completed guarded full range evaluation required')
        for entity, (labels, _) in entities.items():
            entity_buffers(labels, ranged['VUS_per_entity'][entity])
        verify_macros(ranged)
    if torch.cuda.is_initialized():
        raise ValueError('Saved score audit must remain CPU-only')
    target = ROOT / config['output_directory']
    target.mkdir(parents=True, exist_ok=False)
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True, canonical_id=job['id'],
        full_original_entities=len(lengths), full_original_test_points=sum(lengths.values()), full_lengths=lengths,
        all_frozen_weights_source_data_and_saved_score_hashes_verified=True,
        all_original_train_validation_test_scores_finite_and_labels_exact=True,
        all_three_strict_calibrations_event_metrics_bootstraps_and_ten_TAB_core_grid_metrics_recomputed_exact=True,
        actual_producer_environment=dict(**environment, sklearn=sklearn.__version__),
        primary_metrics=actual['strict']['1'], tab_core_diagnostic=actual['tab_core_diagnostic'],
        full_saved_score_protocol_controls=controls,
        full_original_range_source_identity_buffer_grid_scope_and_macros_verified=True,
        VUS_entity_macro=ranged['VUS_entity_macro'], strict_affiliation_entity_macro=ranged['strict_affiliation_entity_macro'],
        numerical_VUS_and_affiliation_integrals_independently_rerun=False,
        model_reinference_performed=False, pretrained_weight_historical_paper_equivalence_certified=False,
        strict_TAB_training_harness_reproduction_certified=False, all_original_experiments_complete=False,
        original_result_sha256=sha(source), original_scores_sha256=result['scores_sha256'],
        range_evaluation_sha256=sha(range_path), source_guard_sha256=sha(ROOT / config['terminal_receipt']),
        audit_configuration_sha256=sha(ROOT / config_path))
    write(target / 'independent_full_saved_score_verification.json', proof)
    print(json.dumps(dict(passed=True, full_test_points=sum(lengths.values()), primary_F1=actual['strict']['1']['micro']['f1'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    if args.worker:
        worker(config, args.config)
        return
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen audit source changed')
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    api = resource_api()
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None:
        raise RuntimeError('This original full audit is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_moment_completed_saved_scores_v1', '--config', args.config, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    creationflags=subprocess.CREATE_NO_WINDOW)
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
            passed = code == 0 and not usage['resource_guard_aborted'] and (ROOT / config['output_directory'] / 'independent_full_saved_score_verification.json').exists()
            write(base / 'receipt.json', dict(exit_code=code, full_saved_score_review_passed=passed, **usage))
            update(dict(state='completed' if passed else 'failed_preserved', active_pid=None))
            if not passed:
                raise RuntimeError('Preserve unsuccessful full saved score audit')
    finally:
        lane.close()


if __name__ == '__main__':
    main()
