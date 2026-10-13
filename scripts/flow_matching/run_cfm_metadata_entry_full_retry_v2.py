"""Retry two proven zero-training entry failures with the full original recipe.

New outputs preserve every failed receipt. Queue on the same family byte lock;
all285 canonical slots, both400/6400 budgets and original resource guards stay.
"""
import argparse
import ast
import os
from pathlib import Path
import sys
from scripts.flow_matching import run_cfm_metadata_permission_full_retry_v1 as previous
from scripts.flow_matching import run_cfm_metadata_queued_family_handoff_v2 as queued
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT = cold.ROOT


def verify(config):
    import psutil
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen zero-training recovery evidence changed: ' + pin['path'])
    try:
        owner = psutil.Process(config['failed_parent']['pid'])
        if abs(owner.create_time() - config['failed_parent']['create_time']) < .001:
            raise ValueError('Previous producer must actually be terminal before new dispatch')
    except psutil.NoSuchProcess:
        pass
    old_runtime = cold.original.read(ROOT / config['failed_runtime'])
    old_jobs, _ = previous.verify_runtime(old_runtime)
    old_state = cold.original.read(ROOT / config['failed_state'])
    if old_state['state'] != 'completed_with_unresolved_jobs' or old_state['active_pid'] is not None or old_state['shared_lock_owned']:
        raise ValueError('Completed failed parent must own no active scientific child or lock')
    for job in old_jobs:
        base = ROOT / old_runtime['controllers']['recovery']['queue']
        queue = cold.original.read(base)
        receipt = cold.original.read(ROOT / queue['state_root'] / 'receipts' / (job['id'] + '.json'))
        stderr = (ROOT / queue['state_root'] / 'logs' / (job['id'] + '.stderr.log')).read_text(encoding='utf-8')
        if (receipt['exit_code'] != 1 or receipt['complete'] or receipt['resource_observations']['resource_guard_aborted']
                or "KeyError: 'main'" not in stderr or (ROOT / job['output_directory']).exists()
                or Path(job['artifact_directory']).exists()):
            raise ValueError('Only proven CLI-entry failures with no outputs or checkpoint may be restarted')
    runtime = cold.original.read(ROOT / config['runtime'])
    jobs, queue = previous.verify_runtime(runtime)
    if runtime['family_lock'] != old_runtime['family_lock'] or queue['resource_policy'] != cold.original.read(ROOT / old_runtime['controllers']['recovery']['queue'])['resource_policy']:
        raise ValueError('Original family lock and all resource safeguards must remain')
    return runtime, queue, jobs


def namespace(runtime, queue):
    values = queued.namespace(runtime, queue)
    tree = previous.main_ast()
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'scripts.flow_matching.run_cfm_metadata_safe_original_job_v1':
            node.value = 'scripts.flow_matching.run_cfm_metadata_safe_original_job_v2'
            changed += 1
    if changed != 1:
        raise ValueError('Only the isolated worker command entry may change')
    exec(compile(tree, __file__, 'exec'), values)
    return values


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = cold.original.read(ROOT / args.config)
    runtime, queue, jobs = verify(config)
    values = namespace(runtime, queue)
    if args.probe_only:
        print(dict(original_slots=len(queue['jobs']), complete_retries=len(jobs),
                   epochs=[j['epochs'] for j in jobs], updates=[j['expected_optimizer_updates'] for j in jobs],
                   old_failure_outputs_preserved=True, training_executed=False))
        return
    old_write = values['write']
    statepath = (ROOT / queue['state_root'] / 'status.json').resolve()
    def write(path, data):
        if Path(path).resolve() == statepath and isinstance(data, dict) and data.get('pid') == os.getpid():
            data = dict(data, create_time=psutil.Process().create_time(), entry_recovery_configuration=args.config)
        return old_write(path, data)
    values['write'] = write
    saved = sys.argv
    try:
        sys.argv = [saved[0], '--runtime-config', config['runtime'], '--controller', 'recovery']
        values['main']()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    main()
