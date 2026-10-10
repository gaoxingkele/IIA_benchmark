"""Restore full trained GRASP checkpoints and replay every saved score recipe."""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait, read, sha, write
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import ROOT, resource_api

SCIENTIFIC_FIELDS = ('profile', 'dataset', 'entity', 'seed', 'model_config',
                     'model_config_sha256', 'expected_optimizer_updates',
                     'test_points', 'validation_points', 'score_recipes')


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen checkpoint replay source differs: ' + item['path'])
    records = config['full_results']
    if len(records) != config['full_job_count'] or len({r['canonical_id'] for r in records}) != len(records):
        raise ValueError('All canonical completed slots must be unique and present')
    queue = read(ROOT / config['queue'])
    canonical = {j['id']: j for j in queue['jobs']}
    for record in records:
        if sha(ROOT / record['result_path']) != record['result_sha256']:
            raise ValueError('Registered complete result changed')
        result = read(ROOT / record['result_path'])
        job = canonical[record['canonical_id']]
        if any(result['job'][k] != job[k] for k in SCIENTIFIC_FIELDS):
            raise ValueError('Recovered job changed original scientific budget or seed')
    return queue


def verify_checkpoint_metadata(checkpoint, result, parameters, fingerprint):
    if (checkpoint['job_sha256'] != fingerprint(result['job'])
            or result['job_sha256'] != checkpoint['job_sha256']
            or checkpoint['seed'] != result['job']['seed']
            or checkpoint['parameters'] != parameters
            or checkpoint['entity_data_audit'] != result['data_audit']):
        raise ValueError('Checkpoint model, seed, data or job binding differs')
    if (checkpoint['epochs_completed'] != 1500 or result['epochs_completed'] != 1500
            or checkpoint['optimizer_updates'] != result['job']['expected_optimizer_updates']
            or checkpoint['optimizer_updates'] != result['optimizer_updates']):
        raise ValueError('Checkpoint full training budget differs')


def compare_full_scores(actual, saved, tolerance):
    import numpy as np
    if (actual.shape != saved.shape or actual.ndim != 1 or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(saved).all()):
        raise ValueError('Full finite point coverage required')
    difference = np.abs(actual.astype(np.float64) - saved.astype(np.float64))
    evidence = dict(max_absolute_difference=float(difference.max()),
                    max_relative_difference=float((difference / np.maximum(np.abs(saved), 1e-12)).max()),
                    bitwise_identical=bool(np.array_equal(actual, saved)))
    if not np.allclose(actual, saved, rtol=tolerance['rtol'], atol=tolerance['atol']):
        raise ValueError('Restored model scores exceed frozen tolerance: ' + str(evidence))
    return evidence


def compare_metrics(actual, saved, tolerance):
    differences = {}
    for protocol, metrics in saved.items():
        for name, value in metrics.items():
            if isinstance(value, bool) or value is None or isinstance(value, str):
                if actual[protocol][name] != value:
                    raise ValueError('Recomputed metric protocol differs')
            elif name in ('quantile',):
                if actual[protocol][name] != value:
                    raise ValueError('Calibration quantile changed')
            else:
                observed = actual[protocol][name]
                difference = abs(observed - value)
                differences[protocol + '/' + name] = difference
                bound = tolerance['threshold_atol'] + tolerance['threshold_rtol'] * abs(value) if name == 'threshold' else tolerance['metric_atol']
                if not math.isfinite(observed) or difference > bound:
                    raise ValueError('Recomputed metric exceeds frozen tolerance: ' + protocol + '/' + name)
    return differences


def equal_arrays(actual, saved, name):
    import numpy as np
    if not np.array_equal(actual, saved):
        raise ValueError('Saved checkpoint preprocessing differs: ' + name)


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    import importlib.util
    from scripts.flow_matching.grasp_protocol_runner import complete, environment_versions, fingerprint, metric_records, verify
    from iia_benchmark.models.grasp_protocol_v2 import GRASPProtocolDetector
    queue = verify_registration(config)
    verify(queue)
    if environment_versions() != read(ROOT / queue['environment_lock']):
        raise ValueError('Original checkpoint numerical environment differs')
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():
        raise ValueError('CPU replay may not allocate a CUDA context')
    target = ROOT / config['output_directory']
    artifact_root = Path(config['artifact_root'])
    if target.exists() or artifact_root.exists():
        raise FileExistsError('Preserve previous partial or complete replay')
    target.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('grasp_replay_native_metrics', ROOT / queue['native_metrics'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.basic_metricor()
    records = []
    for registered in config['full_results']:
        result_path = ROOT / registered['result_path']
        result = read(result_path)
        job = result['job']
        if not complete(job):
            raise ValueError('Full completed training and original artifacts required')
        model_path = ROOT / job['model_config']
        if sha(model_path) != job['model_config_sha256']:
            raise ValueError('Original configured model changed')
        parameters = read(model_path)['parameters']
        entity = result['data_audit']
        if sha(ROOT / entity['prepared_path']) != entity['prepared_sha256']:
            raise ValueError('Full prepared data changed')
        for source in entity['raw_sources']:
            if sha(ROOT / source['path']) != source['sha256']:
                raise ValueError('Original raw data changed')
        checkpoint_path = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'checkpoint.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        verify_checkpoint_metadata(checkpoint, result, parameters, fingerprint)
        detector = GRASPProtocolDetector(**dict(parameters, device='cpu'), seed=job['seed'])
        with np.load(ROOT / entity['prepared_path'], allow_pickle=False) as arrays:
            train, validation, test, labels = (arrays[k] for k in ('train', 'validation', 'test', 'labels'))
        detector.prepare(train, validation)
        for name in ('median', 'mean', 'scale', 'constant_channels'):
            equal_arrays(detector.scaler_[name], checkpoint['scaler'][name], name)
        for name in ('repairs', 'normalization'):
            if detector.scaler_[name] != checkpoint['scaler'][name]:
                raise ValueError('Saved scaler repair or normalization differs')
        equal_arrays(detector.adjacency_, checkpoint['adjacency'], 'adjacency')
        for name in ('distance', 'kernel'):
            equal_arrays(detector.graph_audit_[name], checkpoint['graph_audit'][name], name)
        # The actual saved spectral basis must be restored, including tied eigenspaces.
        detector._build()
        detector.model.load_state_dict(checkpoint['model'], strict=True)
        detector.path.load_state_dict(checkpoint['path'], strict=True)
        detector.model.eval()
        if sum(p.numel() for p in detector.model.parameters()) != result['parameter_count']:
            raise ValueError('Restored architecture parameter count differs')
        for name, state in (('model', detector.model.state_dict()), ('path', detector.path.state_dict())):
            for field, value in state.items():
                if not torch.isfinite(value).all() or not torch.equal(value, checkpoint[name][field]):
                    raise ValueError('Actual trained checkpoint was not restored exactly')
        job_records = []
        job_artifacts = artifact_root / registered['canonical_id']
        job_artifacts.mkdir()
        for original in result['metrics']:
            recipe = original['recipe']
            name = f"s{recipe['source_samples']}_t{recipe['flow_evaluations']}"
            phase = dict(canonical_id=registered['canonical_id'], execution_id=job['id'], recipe=recipe,
                         completed_full_jobs=len(records), required_full_jobs=config['full_job_count'])
            start = time.perf_counter()
            full_scores = {}
            counts = {}
            for tag, values, split_tag in (('validation', validation, 1), ('test', test, 2)):
                write(target / 'progress.json', dict(phase, phase='scoring_' + tag))
                calls = dict(count=0, windows=0)
                def observe(_model, inputs, _output):
                    calls['count'] += 1
                    calls['windows'] += len(inputs[1])
                    if calls['count'] % 500 == 0:
                        write(target / 'progress.json', dict(phase, phase='scoring_' + tag, forward_calls=calls['count'], window_evaluations=calls['windows']))
                hook = detector.model.register_forward_hook(observe)
                try:
                    # Identical source priors, time grid, stride, mapping and batch size.
                    full_scores[tag] = detector.score(values, split_tag=split_tag, **recipe)
                finally:
                    hook.remove()
                expected_windows = (len(values) - detector.window + 1) * recipe['source_samples'] * recipe['flow_evaluations']
                expected_calls = math.ceil((len(values) - detector.window + 1) / detector.batch_size) * recipe['source_samples'] * recipe['flow_evaluations']
                if calls != dict(count=expected_calls, windows=expected_windows):
                    raise ValueError('Incomplete full recipe evaluation budget')
                counts[tag] = calls
            score_path = job_artifacts / (name + '.npz')
            np.savez_compressed(score_path, test_scores=full_scores['test'], validation_scores=full_scores['validation'],
                                labels=labels, test_row_ids=np.arange(len(test)),
                                validation_raw_row_ids=np.arange(entity['split_cut'], entity['raw_train_points']))
            # Preserve the full replay before performing any agreement assertion.
            with np.load(original['scores_path'], allow_pickle=False) as saved:
                equal_arrays(saved['labels'], labels, 'test labels')
                equal_arrays(saved['test_row_ids'], np.arange(len(test)), 'full test row identity')
                equal_arrays(saved['validation_raw_row_ids'], np.arange(entity['split_cut'], entity['raw_train_points']), 'full validation row identity')
                evidence = {tag: compare_full_scores(full_scores[tag], saved[tag + '_scores'], config['score_tolerance']) for tag in ('validation', 'test')}
                threshold = float(np.quantile(full_scores['validation'], .99))
                saved_threshold = float(np.quantile(saved['validation_scores'], .99))
                disagreements = int(np.count_nonzero((full_scores['test'] > threshold) != (saved['test_scores'] > saved_threshold)))
                if disagreements / len(test) > config['metric_tolerance']['maximum_prediction_disagreement_rate']:
                    raise ValueError('Validation-calibrated classifications exceed frozen disagreement bound')
            metrics = metric_records(native, labels, full_scores['test'], full_scores['validation'])
            differences = compare_metrics(metrics, original['metrics'], config['metric_tolerance'])
            record = dict(recipe=recipe, complete_validation_points=len(validation), complete_test_points=len(test),
                          forward_evaluation_counts=counts, replay_score_path=str(score_path), replay_score_sha256=sha(score_path),
                          original_score_path=original['scores_path'], original_score_sha256=sha(original['scores_path']),
                          score_agreement=evidence, prediction_disagreements=disagreements,
                          replayed_metrics=metrics, metric_absolute_differences=differences,
                          elapsed_seconds=time.perf_counter() - start, full_recipe_reinference_passed=True)
            job_records.append(record)
            write(target / (registered['canonical_id'] + '.json'), dict(phase, completed_recipes=job_records))
            print(json.dumps(dict(id=job['id'], recipe=recipe, full_recipe_passed=True, elapsed_seconds=record['elapsed_seconds'], score_agreement=evidence)), flush=True)
        records.append(dict(canonical_id=registered['canonical_id'], execution_id=job['id'], seed=job['seed'],
                            checkpoint_path=checkpoint_path, checkpoint_sha256=sha(checkpoint_path),
                            result_path=registered['result_path'], result_sha256=sha(result_path),
                            epochs_completed=checkpoint['epochs_completed'], optimizer_updates=checkpoint['optimizer_updates'],
                            prepared_data_sha256=entity['prepared_sha256'], parameter_count=result['parameter_count'],
                            full_checkpoint_restoration_and_reinference_passed=True, recipes=job_records))
        write(target / 'progress.json', dict(completed_full_jobs=len(records), required_full_jobs=config['full_job_count'], phase='between_complete_models'))
        del detector, checkpoint, train, validation, test, labels, full_scores
        gc.collect()
    verify_registration(config)
    if torch.cuda.is_initialized() or len(records) != config['full_job_count']:
        raise ValueError('All full CPU checkpoint replays required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
                  configuration_sha256=sha(ROOT / config_path), full_job_count=len(records),
                  full_checkpoint_contents_and_re_inference_passed=True, records=records,
                  score_tolerance=config['score_tolerance'], metric_tolerance=config['metric_tolerance'],
                  cuda_context_initialized=False, additional_training_seed_slots=0,
                  all_original_experiments_complete=False, author_equivalence_certified=False,
                  boundary=config['boundary'])
    write(target / 'checkpoint_replay_audit.json', report)
    write(target / 'validation.json', dict(source_receipts=config['source_receipts'],
          outputs={p.name: sha(p) for p in target.glob('*.json') if p.name != 'validation.json'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify_registration(config)
    if args.probe_only:
        print(json.dumps(dict(full_jobs=config['full_job_count'], full_score_recipes=config['full_score_recipe_count'],
                              torch_imported='torch' in sys.modules, original_full_data_and_budget_required=True)))
        return
    if args.worker:
        worker(config, args.config)
        return
    api = resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Registered checkpoint replay already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
                 configuration_sha256=sha(ROOT / args.config), active_pid=None)
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_grasp_checkpoint_replay_v1', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
                             PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT / config['output_directory'] / 'checkpoint_replay_audit.json'
        passed = bool(code == 0 and proof.exists() and read(proof)['full_checkpoint_contents_and_re_inference_passed'])
        write(base / 'receipt.json', dict(exit_code=code, full_replay_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, full_replay_passed=passed))
        if not passed:
            raise RuntimeError('Preserve failed full checkpoint replay and its original scientific inputs')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
