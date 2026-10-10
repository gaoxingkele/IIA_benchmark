"""Replay all registered complete CFM-TS checkpoints on their full original grids."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.run_light_controller import ROOT
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def verify_checkpoint_metadata(checkpoint, result, job):
    import numpy as np
    fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if checkpoint['job'] != job or result['job'] != job or checkpoint['job_sha256'] != fingerprint or result['job_sha256'] != fingerprint:
        raise ValueError('Saved checkpoint belongs to another registered job/seed')
    if (checkpoint['optimizer_updates'] != job['expected_optimizer_updates']
            or result['optimizer_updates'] != job['expected_optimizer_updates']
            or result['epochs_completed'] != job['epochs']
            or len(checkpoint['training_losses']) != job['epochs']
            or not np.array_equal(checkpoint['training_losses'], result['training_losses'])
            or not np.isfinite(checkpoint['training_losses']).all()):
        raise ValueError('Saved full training budget/history differs')


def compare_full_predictions(actual, saved, tolerance):
    import numpy as np
    if actual.shape != saved.shape or not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError('Complete finite trained-model predictions required')
    difference = float(np.max(np.abs(actual.astype(np.float64)-saved.astype(np.float64))))
    if not np.allclose(actual, saved, rtol=tolerance['rtol'], atol=tolerance['atol']):
        raise ValueError('Checkpoint re-inference differs under frozen tolerance; max_abs='+str(difference))
    return dict(max_absolute_difference=difference, bitwise_identical=bool(np.array_equal(actual, saved)))


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen full checkpoint audit source differs: '+item['path'])
    if config['full_job_count'] != len(config['full_results']) or len({r['original_canonical_id'] for r in config['full_results']}) != len(config['full_results']):
        raise ValueError('All registered canonical completed slots required')


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from scripts.flow_matching.model_registry import build_configured_method
    from scripts.flow_matching.register_cfm_ts_original import environment
    from scripts.flow_matching.run_cfm_ts_original_job import complete
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, prediction_metrics
    from iia_benchmark.models.fm_cfm_ts_original import PaperTrajectoryField
    if environment() != read(ROOT/config['environment_lock']):
        raise ValueError('Original checkpoint inference numerical environment differs')
    torch.set_num_threads(config['cpu_threads'])
    if torch.cuda.is_initialized():
        raise ValueError('Read-only CPU replay must not allocate a CUDA context')
    target = ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve prior complete checkpoint replay')
    target.mkdir(parents=True)
    records = []
    for registered in config['full_results']:
        path = ROOT/registered['result_path']
        if sha(path) != registered['result_sha256']:
            raise ValueError('Frozen full result differs')
        result = read(path)
        job = registered['job']
        if not complete(job):
            raise ValueError('Incomplete source training/scoring result')
        if sha(ROOT/job['data_path']) != job['data_sha256'] or sha(ROOT/job['model_config']) != job['model_config_sha256']:
            raise ValueError('Original full registered data/model differs')
        checkpoint_path = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'model.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        verify_checkpoint_metadata(checkpoint, result, job)
        model = build_configured_method(ROOT/job['model_config'], job['model_overrides'])
        if checkpoint['network_parameters'] != model.network_parameters:
            raise ValueError('Saved network architecture differs from original configuration')
        model.model = PaperTrajectoryField(job['observed_dimensions'], **model.network_parameters)
        model.model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.model.eval()
        if (sum(p.numel() for p in model.model.parameters()) != result['parameter_count']
                or not all(torch.isfinite(p).all() for p in model.model.parameters())):
            raise ValueError('Saved full model parameter count/finite values differ')
        scores_path = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
        with np.load(ROOT/job['data_path'], allow_pickle=False) as data, np.load(scores_path, allow_pickle=False) as arrays:
            audit_arrays(job, result, arrays, data)
            predictions = np.asarray([model.simulate(data['initial_states'][index],
                np.concatenate([[0.], data['test_times'][index]]))[1:] for index in data['test_ids']])
            evidence = compare_full_predictions(predictions, arrays['predictions'], config['prediction_tolerance'])
            metrics, _ = prediction_metrics(predictions, arrays['truth'])
            arrays_path = target/(registered['original_canonical_id']+'_replayed.npz')
            np.savez_compressed(arrays_path, predictions=predictions, truth=arrays['truth'],
                times=arrays['times'], test_ids=arrays['test_ids'])
        record = dict(original_canonical_id=registered['original_canonical_id'], execution_id=job['id'],
            seed=job['seed'], epochs=job['epochs'], optimizer_updates=checkpoint['optimizer_updates'],
            checkpoint_path=checkpoint_path, checkpoint_sha256=sha(checkpoint_path),
            result_path=registered['result_path'], result_sha256=sha(path),
            original_full_data_sha256=job['data_sha256'], full_replay_shape=list(predictions.shape),
            replay_arrays=str(arrays_path.resolve()), replay_arrays_sha256=sha(arrays_path),
            checkpoint_contents_and_full_budget_verified=True, original_full_grid_replayed=True,
            replayed_metrics=metrics, **evidence)
        records.append(record)
        write(target/'progress.json', dict(completed_full_replays=len(records), required_full_replays=config['full_job_count'], latest=record))
        print(json.dumps(dict(completed=len(records), required=config['full_job_count'], id=job['id'], **evidence)), flush=True)
        del model, checkpoint, predictions
        gc.collect()
    if len(records) != config['full_job_count']:
        raise ValueError('All registered full trained-model replays required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=sha(ROOT/config_path), full_job_count=len(records),
        full_checkpoint_contents_and_re_inference_passed=True,
        prediction_tolerance=config['prediction_tolerance'], bitwise_identical_runs=sum(r['bitwise_identical'] for r in records),
        cpu_threads=torch.get_num_threads(), cuda_context_initialized=torch.cuda.is_initialized(),
        additional_training_seed_slots=0, all_original_experiments_complete=False, records=records,
        boundary='All frozen completed CFM-TS canonical slots; restore actual full checkpoints and replay every original held-out test grid. No retraining, shorter grid, modified solver, new seed or test-based attempt selection. Author equivalence and the other remaining original tasks are not certified.')
    write(target/'checkpoint_replay_audit.json', report)


def capped_wait(child, config, api, update):
    settings = config['settings']
    usage = dict(peak_tree_rss_bytes=0, peak_tree_private_bytes=0, resource_guard_aborted=False)
    while child.poll() is None:
        memory = api['resource_snapshot']()
        try:
            parent = psutil.Process(child.pid)
            infos = [p.memory_info() for p in [parent, *parent.children(recursive=True)]]
            rss = sum(i.rss for i in infos)
            private = sum(getattr(i, 'private', i.vms) for i in infos)
            usage['peak_tree_rss_bytes'] = max(usage['peak_tree_rss_bytes'], rss)
            usage['peak_tree_private_bytes'] = max(usage['peak_tree_private_bytes'], private)
            if (memory['commit_available_bytes'] < settings['emergency_commit_headroom_bytes']
                    or rss > settings['maximum_worker_rss_bytes'] or private > settings['maximum_worker_private_bytes']):
                usage.update(resource_guard_aborted=True, abort_snapshot=memory)
                api['terminate_tree'](child)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        update(dict(state='replaying_full_checkpoints', **memory, **usage))
        time.sleep(1)
    return child.wait(), usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT/args.config)
    verify_registration(config)
    if args.probe_only:
        print(json.dumps(dict(full_jobs=config['full_job_count'], torch_imported='torch' in sys.modules,
            numpy_imported='numpy' in sys.modules, full_original_grids_required=True)))
        return
    if args.worker:
        worker(config, args.config)
        return
    api = resource_api()
    base = ROOT/config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base/'controller.lock')
    if owner is None:
        raise RuntimeError('This registered checkpoint replay already has a controller')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting',
        configuration_sha256=sha(ROOT/args.config), active_pid=None)
    def update(values):
        state.update(values)
        write(base/'status.json', state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m', 'scripts.flow_matching.audit_cfm_checkpoint_replay_v1', '--config', args.config, '--worker']
            with (base/'worker.stdout.log').open('xb') as out, (base/'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT/config['output_directory']/'checkpoint_replay_audit.json'
        passed = bool(code==0 and proof.exists() and read(proof)['full_checkpoint_contents_and_re_inference_passed'])
        write(base/'receipt.json', dict(exit_code=code, full_replay_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None, full_replay_passed=passed))
        if not passed:
            raise RuntimeError('Full trained-model replay failed; preserve its immutable inputs/partial evidence')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
