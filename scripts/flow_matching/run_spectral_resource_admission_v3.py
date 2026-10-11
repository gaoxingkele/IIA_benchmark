"""Preserve all full original jobs while reserving measured Spectral sampling headroom."""
import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching import run_spectral_original_queue as original
from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha, write
from scripts.flow_matching.transition_light_controllers_v2 import console_helper


def admission_settings(queue, job, registered_settings):
    settings = copy.deepcopy(registered_settings)
    if job['model'] == 'spectral_flow':
        limit = queue['spectral_sampling_minimum_commit_headroom_bytes']
        if limit < registered_settings['minimum_commit_headroom_bytes']:
            raise ValueError('Sampling admission must not weaken an original guard')
        settings['minimum_commit_headroom_bytes'] = limit
    return settings


def selected_main_ast(source):
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'job'
        and ast.dump(node.iter) == ast.dump(ast.parse("queue['jobs']", mode='eval').body)]
    if len(matches) != 1:
        raise ValueError('Exactly one full original dispatch loop required')
    statement = ast.parse("queue['settings'] = admission_settings(queue, job, registered_settings)").body[0]
    matches[0].body.insert(0, statement)
    return ast.fix_missing_locations(tree)


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen scheduler registration changed: ' + item['path'])
    old = read(ROOT / config['previous_queue'])
    new = read(ROOT / config['queue'])
    retained = copy.deepcopy(new)
    limit = retained.pop('spectral_sampling_minimum_commit_headroom_bytes')
    if retained != old or len(new['jobs']) != 70 or limit != 48 * 1024**3:
        raise ValueError('All original queue fields and70full jobs must remain byte-equivalent')
    evidence = read(ROOT / config['measured_terminal_checkpoint_inspection'])
    peaks = evidence['actual_terminal_resource_receipt']
    if (evidence['original_job']['training_updates'] != 25000
            or not evidence['complete_optimizer_update_budget_verified']
            or peaks['resource_guard_aborted'] is not True
            or limit < peaks['peak_tree_private_bytes'] + old['settings']['emergency_commit_headroom_bytes'] + 6 * 1024**3):
        raise ValueError('Measured full native peak plus original emergency reserve and6GiB margin required')
    prior_proof = read(ROOT / config['previous_preflight'])
    proof = read(ROOT / config['preflight'])
    unchanged = copy.deepcopy(proof)
    unchanged['queue_sha256'] = prior_proof['queue_sha256']
    if unchanged != prior_proof or proof['queue_sha256'] != sha(ROOT / config['queue']):
        raise ValueError('Full14model diagnostic preflight must retain all original observations')
    original.verify(new)
    return new


def idle(parent, state, expected):
    return (parent.pid == expected['pid'] and parent.create_time() == expected['create_time']
        and parent.cmdline() == expected['command'] and state.get('pid') == parent.pid
        and state.get('state') in ('waiting_for_heavy_lock', 'waiting_for_memory', 'waiting_for_gpu_memory')
        and state.get('active_pid') is None and state.get('active_job') is None
        and not state.get('shared_lock_owned', False)
        and not any(not console_helper(child) for child in parent.children(recursive=True)))


def transition(config, path):
    queue = verify_registration(config)
    base = ROOT / config['handoff_root']
    if base.exists():
        raise FileExistsError('Preserve prior transition; never restart the same attempt')
    parent = psutil.Process(config['expected_idle_controller']['pid'])
    state_path = ROOT / queue['state_root'] / 'status.json'
    if not idle(parent, read(state_path), config['expected_idle_controller']):
        raise ValueError('No exact idle predecessor; preserve active workers')
    parent.suspend()
    terminated = False
    try:
        state = read(state_path)
        if not idle(parent, state, config['expected_idle_controller']):
            raise ValueError('Original worker started at boundary; retain it')
        base.mkdir(parents=True)
        write(base / 'preserved_original_idle_status.json', state)
        parent.terminate()
        parent.wait(10)
        terminated = True
    finally:
        if not terminated and parent.is_running():
            parent.resume()
    command = [sys.executable, '-X', 'utf8', '-u', '-m',
        'scripts.flow_matching.run_spectral_resource_admission_v3', '--config', path]
    environment = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    with (base / 'controller.stdout.log').open('xb') as out, (base / 'controller.stderr.log').open('xb') as err:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    receipt = dict(predecessor=config['expected_idle_controller'], terminated_only_verified_idle_controller=True,
        original_active_model_interrupted=False, successor_pid=child.pid,
        successor_create_time=psutil.Process(child.pid).create_time(), successor_command=command,
        original_full_jobs_retained=70, original_cohorts_retained=14,
        original_training_sampling_evaluator_source_and_budgets_unchanged=True,
        stronger_spectral_only_commit_admission_bytes=48 * 1024**3,
        diffusion_ts_resource_admission_and_all_emergency_guards_unchanged=True,
        complete_handoff=True, configuration=path, configuration_sha256=sha(ROOT / path))
    write(base / 'launch.json', receipt)
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--transition-idle', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    queue = verify_registration(config)
    if args.verify_only:
        print(dict(full_original_jobs=70, full_cohorts=14, stronger_spectral_commit_GiB=48,
            diffusion_settings_unchanged=True, all_scientific_fields_unchanged=True))
        return
    if args.transition_idle:
        return transition(config, args.config)
    namespace = dict(original.__dict__, admission_settings=admission_settings,
        registered_settings=copy.deepcopy(queue['settings']))
    tree = selected_main_ast(inspect.getsource(original.main))
    exec(compile(tree, '<original Spectral queue plus scheduler-only admission>', 'exec'), namespace)
    sys.argv = ['original-spectral-queue', '--queue', config['queue'], '--preflight', config['preflight']]
    namespace['main']()


if __name__ == '__main__':
    main()
