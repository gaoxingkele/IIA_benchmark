"""Restore all413 untouched SHCL commands after a host admission query failed.

Unavailable GPU telemetry cannot admit a job. Only prelaunch capacity queries
change; active-worker resource guards, native CLI/data/budgets remain unchanged.
SHCL released VAE: Zhangzzbzzb/SHCL,022520104f2d0d3bda563557ebc99279e6812511.
Paper: https://doi.org/10.1016/j.ins.2025.122680
"""
import argparse
import ast
import copy
import inspect
import subprocess
import sys

from scripts.flow_matching import run_shcl_native_script_queue_v1 as original

NATIVE = original.observer.native
ROOT = NATIVE.ROOT


def admission_gpu_snapshot(settings):
    try:
        observed = original.gpu_snapshot(settings)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return dict(gpu_uuid=settings['gpu_uuid'], gpu_free_bytes=0,
            gpu_admission_observation_valid=False, gpu_admission_observation_error=type(error).__name__,
            gpu_admission_observation_returncode=getattr(error, 'returncode', None))
    return dict(observed, gpu_admission_observation_valid=True)


def host_queue(queue, config):
    if queue != NATIVE.read(ROOT / config['original_queue']):
        raise ValueError('Exact original queue contents required')
    return dict(queue, state_root=config['state_root'])


def main_ast():
    tree = ast.parse(inspect.getsource(original.main))
    function = tree.body[0]
    matches = 0
    for index, node in enumerate(function.body):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'verify'):
            function.body.insert(index + 1, ast.parse('queue = _host_queue(queue, _runtime)').body[0])
            matches += 1
            break
    reservations = [n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'reservation']
    if matches != 1 or len(reservations) != 1:
        raise ValueError('Only host state isolation and original reservation may change')
    gpu_calls = 0
    for node in ast.walk(reservations[0]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'gpu_snapshot':
            node.func.id = '_admission_gpu_snapshot'
            gpu_calls += 1
    if gpu_calls != 2:
        raise ValueError('Exactly the two prelaunch GPU checks must remain')
    return ast.fix_missing_locations(tree)


def retired_owner_absent(incident):
    import psutil
    old = incident['previous_owner']
    try:
        process = psutil.Process(old['pid'])
        same_time = abs(process.create_time() - old['create_time']) < .001
        same_command = process.cmdline() == old['command']
    except psutil.NoSuchProcess:
        return
    if same_time or same_command:
        raise RuntimeError('Original native413 dispatcher still exists')


def verify(config):
    for pin in config['source_receipts']:
        if NATIVE.sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen native recovery source changed: ' + pin['path'])
    queue = NATIVE.read(ROOT / config['original_queue'])
    original.verify(queue)
    if (NATIVE.sha(ROOT / config['original_queue']) != config['original_queue_sha256']
            or config['canonical_job_ids'] != [j['id'] for j in queue['jobs']]
            or config['required_training_commands'] != 49 or config['required_test_commands'] != 364
            or config['settings'] != queue['settings'] or config['resource_lock'] != queue['resource_lock']
            or config['original_scope'] != queue['original_scope'] or config['state_root'] == queue['state_root']):
        raise ValueError('All413 unchanged commands/guards with distinct host state required')
    incident = NATIVE.read(ROOT / config['preserved_incident'])
    if (incident['failure_returncode'] != 255 or incident['failed_before_native_child'] is not True
            or incident['original_jobs_dispatched'] != 0 or incident['scientific_worker_repeated'] is not False):
        raise ValueError('Actual original prelaunch telemetry failure required')
    return queue, incident


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = NATIVE.read(ROOT / args.config)
    queue, incident = verify(config)
    if args.probe_only:
        print(dict(original_commands=len(queue['jobs']), original_trainings=49, original_tests=364,
            original_eight_loader_workers=all(j['native_DataLoader_workers'] == 8 for j in queue['jobs']),
            native_math_and_training_budgets_unchanged=True, native_worker_guards_unchanged=True,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            original_experiments_started=False))
        return
    retired_owner_absent(incident)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], '--queue', config['original_queue']]
        namespace = dict(original.__dict__, _host_queue=host_queue, _runtime=config,
                         _admission_gpu_snapshot=admission_gpu_snapshot)
        exec(compile(main_ast(), __file__, 'exec'), namespace)
        namespace['main']()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
