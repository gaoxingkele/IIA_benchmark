"""Freeze scheduler-only successors; do not change algorithms or resource limits."""
import copy
import json
from pathlib import Path

from scripts.flow_matching.prepare_tsad_execution import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
COMMON = [
    'scripts/flow_matching/heavy_scheduler_v2.py',
    'scripts/flow_matching/register_heavy_scheduler_v2.py',
    'scripts/flow_matching/run_crossad_case_gated_queue_v2.py',
    'scripts/flow_matching/preflight_crossad_complete_v2.py',
    'scripts/flow_matching/preflight_crossad_complete.py',
    'scripts/flow_matching/run_resource_recovery_v2.py',
]


def successor(root, parent_path, target_path, worker, version, kind):
    parent = json.loads((root / parent_path).read_text(encoding='utf-8'))
    queue = copy.deepcopy(parent)
    queue['schema_version'] = version
    queue['worker_script'] = worker
    queue['settings']['queue_path'] = target_path
    frozen = [{'path': p, 'sha256': sha(root / p)} for p in COMMON]
    # Raw dataset files are shared across all cases. Hash every unique source
    # once, rather than allocating/reading the same multi-GB file 139 times.
    expected_files = {r['path']: r['sha256'] for c in queue['jobs']
                      for r in (c if kind == 'crossad' else c['job'])['frozen_files']}
    for name, expected in expected_files.items():
        assert sha(root / name) == expected, name
    for case in queue['jobs']:
        job = case if kind == 'crossad' else case['job']
        output = root / job['output_directory']
        if (output / 'workspace').exists() or (output / 'result.json').exists() or (output / 'failure.json').exists():
            raise ValueError(f'Cannot replace already attempted job: {job["id"]}')
        if kind == 'crossad':
            job['frozen_files'] += frozen
    if kind == 'recovery':
        queue['frozen_files'] += frozen
        # Child job identities stay exact, including their frozen source files.
        assert queue['jobs'] == parent['jobs']
    queue['scheduler_parent_queue'] = parent_path
    queue['scheduler_parent_sha256'] = sha(root / parent_path)
    queue['scheduler_boundary'] = ('Only lock scheduling changes. Physical/commit start gates and emergency guard remain unchanged. '
                                   'No waiting while owning the shared lock; resources rechecked after ownership. '
                                   'Original jobs, datasets, seeds, full budgets and existing native proofs retained.')
    assert len(queue['jobs']) == (139 if kind == 'crossad' else 11)
    assert [j['id'] if kind == 'crossad' else j['job']['id'] for j in queue['jobs']] == [
        j['id'] if kind == 'crossad' else j['job']['id'] for j in parent['jobs']]
    for key in ('minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'):
        assert queue['settings'][key] == parent['settings'][key]
    target = root / target_path
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != queue:
        raise ValueError('Successor queue already frozen with different content')
    write_json(target, queue)
    return {'queue_path': target_path, 'queue_sha256': sha(target), 'parent': parent_path,
            'parent_sha256': sha(root / parent_path), 'registered_jobs': len(queue['jobs']),
            'resource_limits_preserved': True, 'job_scope_preserved': True}


def main():
    crossad = successor(ROOT, 'configs/experiments/fm_crossad_complete_queue.v5.json',
                        'configs/experiments/fm_crossad_complete_queue.v6.json',
                        'run_crossad_case_gated_queue_v2.py', 6, 'crossad')
    recovery = successor(ROOT, 'configs/experiments/fm_resource_recovery_queue.v2.json',
                         'configs/experiments/fm_resource_recovery_queue.v3.json',
                         'run_resource_recovery_v2.py', 3, 'recovery')
    report = {'crossad': crossad, 'recovery': recovery,
              'preflight_parent_unchanged': 'configs/experiments/fm_crossad_complete_queue.v4.json',
              'new_preflight_controller': 'scripts.flow_matching.preflight_crossad_complete_v2',
              'old_controllers_must_be_verified_idle_before_transition': True,
              'all_experiments_complete': False}
    write_json(ROOT / 'docs/reports/fm_heavy_scheduler_registration_2026-10-09.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
