"""One resource-gated worker for all official and reconstructed CrossAD experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.heavy_resources import waited_lock, wait_for_headroom, guarded_wait
from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args(); queue = json.loads(args.queue.read_text()); settings = queue['settings']
    base = ROOT/settings['output_root']; base.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'active_job': None, 'active_pid': None, 'queue_sha256': sha(args.queue), 'state': 'starting'}
    def update(values):
        state.update(values); write_json(base/'status.json', state)
    with waited_lock(base/'worker.lock', update):
        frozen = {r['path']: r['sha256'] for j in queue['jobs'] for r in j['frozen_files']}
        for path, expected in frozen.items():
            if sha(ROOT/path) != expected:
                raise ValueError('CrossAD frozen source/data changed: '+path)
        state.update(frozen_files_verified=len(frozen)); write_json(base/'status.json', state)
        for job in queue['jobs']:
            output = ROOT/job['output_directory']; final = output/'result.json'
            if final.exists():
                receipt = json.loads(final.read_text())
                assert receipt['experiment_sha256'] == hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
                verify_artifacts(receipt); continue
            if (output/'workspace').exists() or (output/'failure.json').exists():
                continue
            # Full batches and original budgets retained. This gate delays jobs rather than shrinking them.
            with waited_lock(ROOT/'experiments/runs/fm_heavy_recovery_cpu.lock', update):
                start_resources = wait_for_headroom(settings, update)
                output.mkdir(parents=True, exist_ok=True)
                env = {k.upper(): v for k, v in os.environ.items()}
                env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg',
                           OMP_NUM_THREADS=str(settings['cpu_threads']), MKL_NUM_THREADS=str(settings['cpu_threads']), PYTHONHASHSEED=str(job['seed']))
                runner = job.get('runner', 'scripts/flow_matching/run_crossad_ablation_job.py' if 'ablation_row' in job else 'scripts/flow_matching/run_crossad_memory_job.py')
                command = [str(ROOT/settings['python']), '-u', str(ROOT/runner), '--queue', str(args.queue), '--job-id', job['id']]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                with (output/'stdout.log').open('a', encoding='utf-8') as out, (output/'stderr.log').open('a', encoding='utf-8') as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
                    state.update(state='running', active_job=job['id'], active_pid=child.pid, active_stage='full_author_or_reconstructed_pipeline', started_unix=time.time())
                    write_json(base/'status.json', state); code, resources = guarded_wait(child, settings, update)
                write_json(output/'resource_usage.json', dict(resources, start_resources=start_resources))
                if code != 0 or not final.exists():
                    write_json(output/'failure.json', {'status': 'failed_preserved', 'id': job['id'], 'exit_code': code, 'resources': resources})
                else:
                    verify_artifacts(json.loads(final.read_text()))
                state.update(active_job=None, active_pid=None, active_stage=None); write_json(base/'status.json', state)
        state['state'] = 'completed' if all((ROOT/j['output_directory']/'result.json').exists() for j in queue['jobs']) else 'completed_with_unresolved_jobs'
        write_json(base/'status.json', state)


if __name__ == '__main__':
    main()
