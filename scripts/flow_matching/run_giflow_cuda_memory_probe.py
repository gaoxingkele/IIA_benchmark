"""Resource-guarded diagnostic children; preserve every formal experiment slot."""
import argparse
import json
import os
import subprocess
import sys
import time

from scripts.flow_matching.giflow_cuda_memory_probe import ROOT, sha, verify_config, write_json
from scripts.flow_matching.run_grasp_protocol_queue import gpu_snapshot, guarded_gpu_wait
from scripts.flow_matching.run_light_controller import load_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    cli = parser.parse_args()
    path = ROOT / cli.config
    config = json.loads(path.read_text(encoding='utf-8'))
    queue = verify_config(config)
    runtime = json.loads((ROOT / queue['runtime_config']).read_text(encoding='utf-8'))
    _, api, _ = load_controller(ROOT, runtime, 'crossad')
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    lane = api['exclusive_lock'](base / 'worker.lock')
    if lane is None:
        raise RuntimeError('Memory preflight controller already owned')
    state = {'pid': os.getpid(), 'config_sha256': sha(path), 'diagnostic': True,
             'active_pid': None, 'profiles': {}}
    def update(values):
        state.update(values)
        write_json(base / 'status.json', state)
    for profile in config['profiles']:
        output = ROOT / profile['output_directory']
        if output.exists():
            state['profiles'][profile['id']] = 'existing_diagnostic_preserved'
            continue
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            while gpu_snapshot(config['settings'])['gpu_free_bytes'] < config['settings']['minimum_gpu_free_bytes']:
                update({'state': 'waiting_for_gpu_memory', 'active_pid': None})
                time.sleep(10)
            api['wait_for_headroom'](config['settings'], update)
            env = {k.upper(): v for k, v in os.environ.items()}
            env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='0',
                       OMP_NUM_THREADS=str(config['settings']['cpu_threads']),
                       MKL_NUM_THREADS=str(config['settings']['cpu_threads']), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (base / (profile['id'] + '.stdout.log')).open('x', encoding='utf-8') as out, \
                 (base / (profile['id'] + '.stderr.log')).open('x', encoding='utf-8') as err:
                child = subprocess.Popen([queue['python'], '-u', '-m', 'scripts.flow_matching.giflow_cuda_memory_probe',
                                          '--config', str(path), '--profile', profile['id']], cwd=ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                update({'state': 'running', 'active_profile': profile['id'], 'active_pid': child.pid})
                code, usage = guarded_gpu_wait(child, config['settings'], api, update)
            result_path = output / 'cuda_memory_result.json'
            passed = code == 0 and result_path.exists()
            if passed:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                passed = result['status'] == 'cuda_full_batch_memory_preflight_passed_not_benchmark'
            state['profiles'][profile['id']] = 'memory_preflight_passed' if passed else 'failed_diagnostic_preserved'
            write_json(base / (profile['id'] + '.resource_usage.json'), dict(usage, exit_code=code,
                       result_sha256=sha(result_path) if result_path.exists() else None))
            update({'active_pid': None, 'active_profile': None})
    update({'state': 'completed' if all(s == 'memory_preflight_passed' for s in state['profiles'].values())
            else 'completed_with_unresolved_diagnostics'})


if __name__ == '__main__':
    main()
