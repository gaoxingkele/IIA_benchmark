"""Pin the original waiting TSAD GPU controller without unused data imports."""
import json
from pathlib import Path
import psutil
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


CONFIG = 'configs/runtime/fm_tsad_gpu_light_controller.v1.json'


def main():
    destination = ROOT / CONFIG
    if destination.exists():
        config = json.loads(destination.read_text(encoding='utf-8'))
        load_controller(ROOT, config, 'tsad_gpu')
        print(json.dumps({'registered': True, 'sha256': digest(destination), 'unchanged': True}))
        return
    original_path = ROOT / 'configs/runtime/fm_light_controllers.v1.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    state_path = 'experiments/runs/fm_tsad_execution_v1/gpu_status.json'
    state = json.loads((ROOT / state_path).read_text(encoding='utf-8'))
    process = psutil.Process(state['pid'])
    queue_path = 'configs/experiments/fm_tsad_gpu_queue.v1.json'
    binding = {'name': 'tsad_gpu', 'source_path': 'scripts/flow_matching/run_tsad_queue.py',
               'source_functions': ['main'], 'queue_path': queue_path,
               'queue_sha256': digest(ROOT / queue_path), 'status_path': state_path,
               'queue_argument': '--queue', 'old_worker_token': 'scripts/flow_matching/run_tsad_queue.py',
               'resource_scheduler_binding': False, 'expected_old_pid': process.pid,
               'expected_old_create_time': process.create_time()}
    sources = {r['path']: r['sha256'] for r in original['source_receipts']}
    for path in ['configs/runtime/fm_light_controllers.v1.json',
                 'scripts/flow_matching/register_tsad_gpu_light_controller.py',
                 'scripts/flow_matching/transition_tsad_gpu_light_controller.py',
                 'tests/test_fm_tsad_gpu_light_controller.py']:
        sources[path] = digest(ROOT / path)
    config = {'schema_version': 1, 'controllers': [binding],
              'source_receipts': [{'path': p, 'sha256': h} for p, h in sorted(sources.items())],
              'transition_archive': 'experiments/runs/fm_tsad_gpu_light_transition_20261009',
              'boundary': 'Exact original GPU TSAD orchestration, prerequisites, locks, dispatch and all job dictionaries preserved. Only unused top-level data-library imports omitted. Handoff requires the observed idle PID/create-time, exact queue, waiting_existing_imputation state and no descendants. No active model is stopped.'}
    load_controller(ROOT, config, 'tsad_gpu')
    from scripts.flow_matching.transition_tsad_gpu_light_controller import idle_observation
    if idle_observation(binding) is None:
        raise RuntimeError('Original TSAD GPU controller is no longer verified idle')
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'registered': True, 'sha256': digest(destination), 'original_pid': process.pid}))


if __name__ == '__main__':
    main()
