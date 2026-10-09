"""Preserve failed v1 attempts and register the verified manifest-contract repair."""
from copy import deepcopy
import json
from pathlib import Path
import psutil

from scripts.flow_matching.native_exact_recovery import ROOT, sha
from scripts.flow_matching.native_exact_recovery_manifest_v2 import verified_manifest
from scripts.flow_matching.grasp_protocol_runner import write_json


def main():
    original_path = 'configs/experiments/fm_native_grasp_exact_recovery.v1.json'
    target = ROOT / 'configs/experiments/fm_native_grasp_exact_recovery.v2.json'
    if target.exists():
        raise FileExistsError('Registration already frozen')
    original = json.loads((ROOT / original_path).read_text(encoding='utf-8'))
    verified_manifest(original)
    commands = []
    for process in psutil.process_iter(['name']):
        if (process.info['name'] or '').lower() != 'python.exe':
            continue
        try:
            commands.append(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    queue = deepcopy(original)
    queue.update(state_root='experiments/runs/fm_native_grasp_exact_recovery_v2',
                 artifact_root='F:/aicoding/IIA_Data/experiments/native_exact_recovery_v2/grasp',
                 previous_attempt_queue=original_path, previous_attempt_failure_receipts=[],
                 verifier_contract='Return the verified original data manifest, not the queue. Frozen run() and train/evaluate remain unchanged.')
    for job in queue['jobs']:
        if any(job['id'] in c or job['recovery_original_job_id'] in c for c in commands):
            raise ValueError('A corresponding model is still live')
        failure = ROOT / original['state_root'] / 'logs' / (job['id'] + '.stderr.log')
        if "KeyError: 'datasets'" not in failure.read_text(encoding='utf-8'):
            raise ValueError('Expected terminal manifest-contract failure missing')
        queue['previous_attempt_failure_receipts'].append({'path': failure.relative_to(ROOT).as_posix(), 'sha256': sha(failure)})
        job['id'] = job['recovery_original_job_id'] + '__native_exact_recovery_v2'
        job['output_directory'] = queue['state_root'] + '/jobs/' + job['id']
        if (ROOT / job['output_directory']).exists():
            raise FileExistsError('Recovery output already exists')
    paths = [original_path, 'scripts/flow_matching/native_exact_recovery_manifest_v2.py',
             'scripts/flow_matching/run_native_grasp_recovery_v2.py',
             'scripts/flow_matching/register_native_grasp_recovery_v2.py',
             'tests/test_fm_native_recovery_manifest_v2.py']
    queue['source_receipts'] += [{'path': p, 'sha256': sha(ROOT / p)} for p in paths]
    queue['source_receipts'] += queue['previous_attempt_failure_receipts']
    verified_manifest(queue)
    write_json(target, queue)
    print(json.dumps({'registered': len(queue['jobs']), 'full_epochs': 1500,
                      'failure_attempts_preserved': True, 'training_semantics_unchanged': True}))


if __name__ == '__main__':
    main()
