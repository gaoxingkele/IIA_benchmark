"""Full cold replay of a frozen local SB objective adapter without fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import subprocess

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def validate_budget(job, result, model_config, guard):
    parameters = model_config['parameters']
    if (result['status'] != 'completed' or result['purpose'] != 'full_local_tsad_benchmark'
            or result['job'] != job or result['id'] != job['id']
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            or parameters['objective'] != 'sb' or parameters['epochs'] != 20
            or parameters['monte_carlo_samples'] != 4 or parameters['score_times'] != [.1, .3, .5, .7, .9]
            or len(result['training_losses']) != parameters['epochs']
            or not all(math.isfinite(loss) and loss >= 0 for loss in result['training_losses'])
            or guard['job_id'] != job['id'] or guard['exit_code'] != 0
            or guard['resource_guard_aborted'] or not guard['full_registered_budget_retained']):
        raise ValueError('Full original20-epoch SB objective, scoring grid and guarded successful source required')


def worker(config, path):
    import platform
    import numpy as np
    import torch
    from torch import nn
    from scripts.flow_matching.run_tsad_queue import complete_result
    from scripts.flow_matching.model_registry import build_configured_method
    from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores, evaluate_saved
    from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
    from scripts.flow_matching.publish_full_range_increment_v25 import entity_buffers, verify_macros
    queue = read(ROOT / config['queue'])
    job = next(item for item in queue['jobs'] if item['id'] == config['job_id'])
    result_path = ROOT / job['output_directory'] / 'result.json'
    result = read(result_path)
    model_config = read(ROOT / job['model_config'])
    guard = read(ROOT / config['terminal_receipt'])
    validate_budget(job, result, model_config, guard)
    if not complete_result(ROOT, job):
        raise ValueError('Full source checkpoint and arrays must verify')
    for pin in config['source_receipts'] + job['frozen_files']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen source/model/data changed: ' + pin['path'])
    environment = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        cuda=torch.version.cuda, device='cpu')
    if environment != result['environment']:
        raise ValueError('Exact original source environment required')
    torch.set_num_threads(queue['evaluation']['cpu_threads'])
    model = build_configured_method(model_config, job['parameter_overrides'])
    manifest = read(ROOT / job['dataset_manifest_path'])
    dataset = next(item for item in manifest['datasets'] if item['id'] == job['dataset'])
    if sha(ROOT / dataset['input_npz']) != dataset['input_sha256'] or not dataset['grouped_splits_verified']:
        raise ValueError('Complete frozen grouped input required')
    checkpoint = torch.load(ROOT / job['output_directory'] / 'model.pt', map_location='cpu', weights_only=True)
    if (checkpoint['configuration'] != model_config or checkpoint['overrides'] != job['parameter_overrides']
            or checkpoint['window'] != job['window'] or set(checkpoint['modules']) != {'_model'}):
        raise ValueError('Actual complete local SB checkpoint identity required')
    with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as inputs, np.load(ROOT / job['output_directory'] / 'scores.npz', allow_pickle=False) as saved:
        entities, lengths, validation = check_arrays(inputs, saved, dataset)
        features = inputs[dataset['entities'][0]['array_prefix'] + '_train'].shape[1]
        model._shape = (job['window'], features)
        model._device = torch.device('cpu')
        model._score_head = False
        dimensions = job['window'] * features
        hidden = model_config['parameters']['hidden_size']
        # Restore the exact architecture used by the frozen fit implementation.
        # No fit method, optimizer, coupling solver or training update is called.
        model._model = nn.Sequential(nn.Linear(dimensions + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dimensions))
        model._model.load_state_dict(checkpoint['modules']['_model'], strict=True)
        model._model.eval()
        if sum(parameter.numel() for parameter in model._model.parameters()) != result['parameter_count']:
            raise ValueError('Every full original parameter required')
        if any(not torch.equal(value, checkpoint['modules']['_model'][key]) or not torch.isfinite(value).all()
                for key, value in model._model.state_dict().items()):
            raise ValueError('Every complete original model tensor must restore bitwise and finite')
        coverage = []
        for entry in dataset['entities']:
            prefix = entry['array_prefix']
            for split in ('train', 'validation', 'test'):
                values = inputs[prefix + '_' + split]
                windows, starts = covering_windows(values, job['window'])
                predictions = model.score(windows)
                replayed = align_scores(predictions, starts, len(values), job['window'])
                if not np.array_equal(replayed, saved[prefix + '_' + split]):
                    raise ValueError('Complete native timestamp score replay must match bitwise: ' + prefix + '_' + split)
                coverage.append(dict(entity=entry['entity_id'], split=split, points=len(values), windows=len(windows), every_score_bitwise_equal=True))
                print(json.dumps(coverage[-1]), flush=True)
        training = np.concatenate([saved[entry['array_prefix'] + '_train'] for entry in dataset['entities']])
        actual = evaluate_saved(entities, validation, training, dataset['input_sha256'], result['checkpoint_sha256'], queue['evaluation'], job['seed'])
        if actual != result['metrics']:
            raise ValueError('All full original core/event/uncertainty metrics must recompute exactly')
        primary = actual['strict']['1']
        scores = np.concatenate([pair[1] for pair in entities.values()])
        labels = np.concatenate([pair[0] for pair in entities.values()])
        threshold = primary['calibration']['threshold']
        distributions = {}
        for name, array in [('train', training), ('validation', validation),
                ('test_normal_diagnostic_only', scores[labels == 0]), ('test_anomaly_diagnostic_only', scores[labels == 1])]:
            distributions[name] = dict(points=len(array), quantiles=np.quantile(array, [.5, .9, .95, .99]).tolist())
        ranges = read(ROOT / config['range_evaluation'])
        science = read(ROOT / config['range_config'])
        science_sha = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
        if (ranges['status'] != 'completed_range_evaluation' or ranges['id'] != job['id']
                or ranges['model_result_sha256'] != sha(result_path) or ranges['scores_sha256'] != result['scores_sha256']
                or ranges['evaluation_config_sha256'] != science_sha or ranges['source_receipts'] != science['source_receipts']
                or ranges['evaluation_identity'] != hashlib.sha256((sha(result_path) + science_sha).encode()).hexdigest()
                or sha(ROOT / config['range_evaluation']) != (ROOT / config['range_evaluation']).with_name('evaluation.sha256').read_text(encoding='ascii').strip()
                or set(ranges['VUS_per_entity']) != set(entities)):
            raise ValueError('Complete original range identity must bind full exact source scores')
        for pin in science['source_receipts']:
            if sha(ROOT / pin['path']) != pin['sha256']:
                raise ValueError('Frozen original range metric source changed')
        for entity, (truth, _) in entities.items():
            entity_buffers(truth, ranges['VUS_per_entity'][entity])
        verify_macros(ranges)
    if torch.cuda.is_initialized():
        raise ValueError('Cold replay must remain CPU-only')
    target = ROOT / config['output_directory']
    target.mkdir(parents=True, exist_ok=False)
    write(target / 'independent_full_model_score_verification.json', dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        canonical_id=job['id'], restored_full_parameters=result['parameter_count'], complete_checkpoint_tensors_restored_bitwise=True,
        original_epochs=20, additional_fit_calls=0, additional_optimizer_updates=0,
        original_train_validation_test_native_scores_reinferred_bitwise=True, full_native_split_coverage=coverage,
        full_native_entities=len(lengths), full_native_test_points=sum(lengths.values()), full_native_test_lengths=lengths,
        all_original_metrics_events_and_bootstrap_intervals_recomputed_exact=True, primary_metrics=primary,
        score_distribution_diagnostics=distributions, test_labels_used_for_primary_calibration=False,
        frozen_validation_threshold=threshold, measured_test_normal_false_alarm_fraction=float(np.mean(scores[labels == 0] > threshold)),
        tab_core_diagnostic=actual['tab_core_diagnostic'], full_range_source_buffer_scope_and_macros_verified=True,
        VUS_entity_macro=ranges['VUS_entity_macro'], strict_affiliation_entity_macro=ranges['strict_affiliation_entity_macro'],
        numerical_VUS_and_affiliation_integrals_independently_rerun=False, environment=environment,
        source_result_sha256=sha(result_path), source_checkpoint_sha256=result['checkpoint_sha256'], source_scores_sha256=result['scores_sha256'],
        range_evaluation_sha256=sha(ROOT / config['range_evaluation']), configuration_sha256=sha(ROOT / path),
        original_author_TSAD_experiment_equivalence=False, strict_TAB_training_harness_equivalence=False, all_original_experiments_complete=False))


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
            raise ValueError('Frozen replay configuration input changed')
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    api = resource_api()
    lane = api['exclusive_lock'](base / 'controller.lock')
    if lane is None:
        raise RuntimeError('This complete SB cold replay is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, state='starting')
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_local_SB_full_checkpoint_scores_v1', '--config', args.config, '--worker']
            with (base / 'stdout.log').open('xb') as out, (base / 'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                    creationflags=subprocess.CREATE_NO_WINDOW)
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
            proof = ROOT / config['output_directory'] / 'independent_full_model_score_verification.json'
            passed = code == 0 and not usage['resource_guard_aborted'] and proof.exists()
            write(base / 'receipt.json', dict(exit_code=code, full_cold_model_replay_passed=passed, **usage))
            update(dict(state='completed' if passed else 'failed_preserved', active_pid=None))
            if not passed:
                raise RuntimeError('Preserve unsuccessful full original model replay')
    finally:
        lane.close()


if __name__ == '__main__':
    main()
