"""Register complete native experiments with a model-boundary fair handoff."""
import json
import psutil
from scripts.flow_matching.grasp_protocol_runner import ROOT, sha, verify, write_json


def main():
    original_path = 'configs/experiments/fm_grasp_protocol_queue.v3.json'
    target_path = 'configs/experiments/fm_grasp_protocol_queue.v4.json'
    handoff_path = 'configs/runtime/fm_grasp_boundary_handoff.v1.json'
    if (ROOT / target_path).exists() or (ROOT / handoff_path).exists():
        raise FileExistsError('Fair native registration already frozen')
    original = json.loads((ROOT / original_path).read_text(encoding='utf-8')); verify(original)
    paths = [original_path, 'scripts/flow_matching/run_grasp_fair_queue_v4.py',
             'scripts/flow_matching/transition_grasp_at_job_boundary.py',
             'scripts/flow_matching/register_grasp_fair_queue_v4.py',
             'scripts/flow_matching/transition_light_controllers_v2.py',
             'tests/test_fm_grasp_boundary_fairness.py']
    receipts = [{'path': path, 'sha256': sha(ROOT / path)} for path in paths]
    queue = {**original, 'state_root': 'experiments/runs/fm_grasp_protocol_v4',
             'controller_yield_seconds': 2, 'source_receipts': [*original['source_receipts'], *receipts]}
    if queue['jobs'] != original['jobs'] or queue['settings'] != original['settings']:
        raise ValueError('Full experimental jobs changed')
    verify(queue)
    state = json.loads((ROOT / original['state_root'] / 'status.json').read_text(encoding='utf-8'))
    parent = psutil.Process(state['pid'])
    command = parent.cmdline()
    if 'scripts.flow_matching.run_grasp_protocol_queue_v3' not in command:
        raise ValueError('Original full queue controller identity differs')
    if '--queue' not in command or (ROOT / command[command.index('--queue') + 1]).resolve() != (ROOT / original_path).resolve():
        raise ValueError('Original native queue binding differs')
    write_json(ROOT / target_path, queue)
    handoff = {'original_queue': original_path, 'original_queue_sha256': sha(ROOT / original_path),
               'successor_queue': target_path, 'successor_queue_sha256': sha(ROOT / target_path),
               'state_root': 'experiments/runs/fm_grasp_boundary_handoff_v1',
               'expected_controller': {'pid': parent.pid, 'create_time': parent.create_time(),
                    'module': 'scripts.flow_matching.run_grasp_protocol_queue_v3',
                    'queue_path': str((ROOT / original_path).resolve())},
               'source_receipts': receipts,
               'boundary': 'Original guard remains active while a model runs. Handoff only follows a verified complete model exit with no new model descendants; races resume and defer. All original 9120 entity jobs, 480 cohorts, budgets, data, seeds, outputs, score recipes and emergency guards unchanged. Successor releases native GPU mutex then yields two seconds to other native queues. Goal remains all known unfinished experiments.'}
    write_json(ROOT / handoff_path, handoff)
    print(json.dumps({'original_jobs_unchanged': len(queue['jobs']), 'original_cohorts': queue['cohort_count'],
                      'controller_yield_seconds': queue['controller_yield_seconds'], 'handoff_parent': parent.pid,
                      'successor_queue_sha256': sha(ROOT / target_path)}))


if __name__ == '__main__':
    main()
