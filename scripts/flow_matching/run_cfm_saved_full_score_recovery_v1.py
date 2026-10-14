"""Recover full scoring after an observed metadata-only failure; never fit.

Restore the completed original checkpoint and run the original scoring body,
solver and grids. Preserve failed outputs in place and publish a separate alias.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching.run_cfm_metadata_safe_original_job_v1 import safe_write
from scripts.flow_matching import queued_family_lock_v1 as queued

ROOT = cold.ROOT
IDENTITY = 'cfm_ts_original__main_text__gp_gaussian_conditional_e400__sine__seed1106'


def verify(config):
    read, sha = cold.original.read, cold.streaming_sha
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Preserved metadata failure or frozen source changed')
    primary = read(ROOT / config['primary_runtime'])
    original = next(r['job'] for r in primary['routes'] if r['original_canonical_id'] == IDENTITY)
    job = config['job']
    cold.original.check_binding(original, job)
    if (config['original_job'] != original or original['epochs'] != 400
            or original['expected_optimizer_updates'] != 62400 or original['test_trajectories'] != 20
            or original['test_points_per_trajectory'] != 500 or original['observed_dimensions'] != 1
            or job['id'] == IDENTITY or job['output_directory'] == original['output_directory']
            or job['artifact_directory'] == original['artifact_directory']):
        raise ValueError('Exact completed400/62400 checkpoint and all20 grids required')
    for key in ['python', 'cpu_threads', 'settings', 'resource_lock', 'environment_lock', 'original_scope']:
        if config[key] != primary[key]:
            raise ValueError('Original native scoring resource policy must remain: ' + key)
    failure = read(ROOT / config['failed_receipt'])
    progress = read(ROOT / config['failed_progress'])
    stderr = (ROOT / config['failed_stderr']).read_text(encoding='utf-8')
    if (failure['exit_code'] != 1 or failure['complete'] or failure['resource_observations']['resource_guard_aborted']
            or progress['phase'] != 'scoring' or progress['test_trajectories_completed'] != 6
            or progress['required_test_trajectories'] != 20 or 'PermissionError: [WinError 5]' not in stderr
            or "write(output/'progress.json'" not in stderr
            or (ROOT / original['output_directory'] / 'result.json').exists()):
        raise ValueError('Only the actual saved-model progress-write failure may be recovered')
    if sha(config['checkpoint_path']) != config['checkpoint_sha256']:
        raise ValueError('Original complete saved model changed')
    return primary


def restore(model, config):
    import numpy as np
    import torch
    from iia_benchmark.models.fm_cfm_ts_original import PaperTrajectoryField
    from scripts.flow_matching.run_cfm_ts_original_job import fingerprint
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=False)
    job = config['original_job']
    count = job['training_items']
    if (checkpoint['job'] != job or checkpoint['job_sha256'] != fingerprint(job)
            or checkpoint['network_parameters'] != model.network_parameters
            or checkpoint['optimizer_updates'] != job['expected_optimizer_updates']
            or len(checkpoint['training_losses']) != job['epochs']
            or not np.isfinite(checkpoint['training_losses']).all()
            or job['epochs'] * ((count + job['batch_size'] - 1) // job['batch_size']) != checkpoint['optimizer_updates']):
        raise ValueError('Exact full checkpoint budget, architecture and registration required')
    model.model = PaperTrajectoryField(job['observed_dimensions'], **model.network_parameters)
    model.model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.model.parameters()):
        raise ValueError('Nonfinite saved model')
    # The original fit returns an eval-mode network; no optimizer is created.
    model.model.eval()
    model.loss_history_ = checkpoint['training_losses']
    model.optimizer_updates_ = checkpoint['optimizer_updates']
    model.preprocessing_ = checkpoint['preprocessing']
    model.training_items_ = count
    if not all(torch.equal(v, model.model.state_dict()[k]) for k, v in checkpoint['state_dict'].items()):
        raise ValueError('Restored model weights must be bitwise identical')


def scoring_ast(original):
    tree = ast.parse(inspect.getsource(original.run))
    changed = 0
    for index, node in enumerate(tree.body[0].body):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'fit'):
            tree.body[0].body[index] = ast.parse('_restore(model, _recovery_config)').body[0]
            changed += 1
    if changed != 1:
        raise ValueError('Replace exactly the original fit call, keeping the full original scoring body')
    return ast.fix_missing_locations(tree)


def worker(config):
    from scripts.flow_matching.run_cfm_metadata_safe_original_job_v1 import namespace
    from scripts.flow_matching import run_cfm_ts_original_job as original
    verify(config)
    values = namespace()
    values.update(_restore=restore, _recovery_config=config)
    def write(path, data):
        if Path(path).name == 'result.json':
            data = dict(data, checkpoint_restore_seconds=data['training_seconds'], training_seconds=None,
                additional_optimizer_updates=0, recovery_original_checkpoint=config['checkpoint_path'],
                recovery_original_checkpoint_sha256=config['checkpoint_sha256'],
                recovery_original_job_sha256=original.fingerprint(config['original_job']),
                restored_original_weights_bitwise_identical=True,
                recovery_boundary='Original full400/62400 model; no fitting or target preprocessing. '
                'Every20x500 grid rescored with the original solver. Original training duration is unavailable.')
        safe_write(path, data)
    values['write'] = write
    exec(compile(scoring_ast(original), __file__, 'exec'), values)
    queue = cold.original.read(ROOT / config['original_queue'])
    values['run'](queue, config['job'])
    if cold.streaming_sha(config['checkpoint_path']) != config['checkpoint_sha256']:
        raise ValueError('Original checkpoint must remain untouched')


def controller(config, config_path):
    import psutil
    api = cold.cached_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Saved-model scoring already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=cold.streaming_sha(ROOT / config_path), additional_optimizer_updates=0,
        all_original_experiments_complete=False, state='starting', shared_lock_owned=False)
    def update(values):
        state.update(values)
        safe_write(base / 'status.json', state)
    try:
        verify(config)
        with queued.resource_slot(ROOT / config['family_lock'], config['settings'], api, update):
            with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
                command = [config['python'], '-X', 'utf8', '-u', '-m',
                    'scripts.flow_matching.run_cfm_saved_full_score_recovery_v1', '--config', config_path, '--worker']
                with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                        env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                            PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='-1'),
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    handle = psutil.Process(child.pid)
                    update(dict(active_pid=child.pid, active_create_time=handle.create_time(), actual_command=handle.cmdline()))
                    code, usage = cold.original.capped_wait(child, config, api, update)
        result = ROOT / config['job']['output_directory'] / 'result.json'
        passed = bool(code == 0 and not usage['resource_guard_aborted'] and result.is_file())
        safe_write(base / 'receipt.json', dict(exit_code=code, complete=passed,
            additional_optimizer_updates=0, resource_observations=usage))
        update(dict(active_pid=None, state='completed' if passed else 'failed_preserved', complete=passed))
        if not passed:
            raise RuntimeError('Full saved-model scoring failed; preserve evidence without automatic retry')
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = cold.original.read(ROOT / args.config)
    verify(config)
    if args.probe_only:
        print(json.dumps(dict(original_slots=285, saved_epochs=400, saved_updates=62400,
            full_score_shape=[20, 500, 1], additional_optimizer_updates=0,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules)))
    elif args.worker:
        worker(config)
    else:
        controller(config, args.config)


if __name__ == '__main__':
    main()
