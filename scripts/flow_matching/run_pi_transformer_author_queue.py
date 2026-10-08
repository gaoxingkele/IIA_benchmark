"""Single CPU worker for full Pi pipelines; preserve partials and verify frozen inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_queue import exclusive_lock
from scripts.flow_matching.run_maelnet_author_queue import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding='utf-8'))
    settings = queue['settings']
    base = ROOT / settings['output_root']
    base.mkdir(parents=True, exist_ok=True)
    state_path = base / 'status.json'
    state = {'pid': os.getpid(), 'state': 'starting', 'queue_sha256': sha(args.queue), 'active_job': None, 'active_pid': None}
    with exclusive_lock(base / 'worker.lock'):
        verified = {}
        for job in queue['jobs']:
            for source in job['frozen_files']:
                if source['path'] in verified and verified[source['path']] != source['sha256']:
                    raise ValueError('Inconsistent Pi source identity')
                verified[source['path']] = source['sha256']
        for name, expected in verified.items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen Pi source changed: ' + name)
        state.update(frozen_files_verified=len(verified), state='running')
        write_json(state_path, state)
        for job in queue['jobs']:
            output = ROOT / job['output_directory']
            final = output / 'result.json'
            if final.exists():
                receipt = json.loads(final.read_text(encoding='utf-8'))
                if receipt['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
                    raise ValueError('Pi result identity changed')
                verify_artifacts(receipt)
                continue
            if (output / 'checkpoints').exists() or (output / 'failure.json').exists():
                continue
            output.mkdir(parents=True, exist_ok=True)
            env = {k.upper(): v for k, v in os.environ.items()}
            env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1',
                       OMP_NUM_THREADS=str(settings['cpu_threads']), MKL_NUM_THREADS=str(settings['cpu_threads']),
                       PYTHONHASHSEED=str(job['seed']))
            command = [str(ROOT / settings['python']), '-u', str(ROOT / 'scripts/flow_matching/run_pi_transformer_job.py'),
                       '--queue', str(args.queue), '--job-id', job['id']]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            with (output / 'stdout.log').open('a', encoding='utf-8') as out, (output / 'stderr.log').open('a', encoding='utf-8') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags)
                state.update(active_job=job['id'], active_pid=child.pid, active_stage='full_train_and_test', started_unix=time.time())
                write_json(state_path, state)
                code = child.wait()
            if code != 0 or not final.exists():
                write_json(output / 'failure.json', {'status': 'failed_preserved', 'exit_code': code, 'id': job['id'], 'logs_preserved': True})
            else:
                verify_artifacts(json.loads(final.read_text(encoding='utf-8')))
            state.update(active_job=None, active_pid=None, active_stage=None)
            write_json(state_path, state)
        state['state'] = 'completed_with_unresolved_jobs' if any(not (ROOT / j['output_directory'] / 'result.json').exists() for j in queue['jobs']) else 'completed'
        write_json(state_path, state)


if __name__ == '__main__':
    main()
