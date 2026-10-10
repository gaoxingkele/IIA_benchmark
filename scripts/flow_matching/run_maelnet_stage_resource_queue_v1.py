"""Run the unchanged MaelNet stage pipeline with shared per-stage CPU admission."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest, load_functions
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_resource_gated_light_controller_v1 import GuardedChild, write


FUNCTIONS = ['read', 'argument', 'replace_argument', 'effective_arguments',
    'verify_artifacts', 'training_rows', 'stage_artifacts', 'run_job', 'main']


def original_namespace():
    namespace = dict(globals())
    load_functions(ROOT, 'scripts/flow_matching/prepare_tsad_execution.py', ['sha', 'write_json'], namespace)
    load_functions(ROOT, 'scripts/flow_matching/run_tsad_queue.py', ['exclusive_lock'], namespace)
    load_functions(ROOT, 'scripts/flow_matching/run_maelnet_author_queue.py', FUNCTIONS, namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); path = ROOT / args.runtime_config
    registration = json.loads(path.read_text(encoding='utf-8'))
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen MaelNet admission source differs: ' + entry['path'])
    queue_path = ROOT / registration['queue']
    if digest(queue_path) != registration['queue_sha256']:
        raise ValueError('Original MaelNet queue differs')
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    namespace = original_namespace()
    if args.probe_only:
        print(json.dumps(dict(registered_jobs=len(queue['jobs']), unchanged_author_functions=FUNCTIONS,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            private_bytes=psutil.Process().memory_info().private)))
        return
    base = ROOT / registration['state_root']; api = resource_api()
    lane = api['exclusive_lock'](base / 'adapter.lock')
    if lane is None:
        raise RuntimeError('This exact MaelNet adapter is already live')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        runtime_sha256=digest(path), queue_sha256=digest(queue_path), state='starting', active_pid=None)
    def update(values):
        state.update(values); write(base / 'status.json', state)
    current = {'job': None, 'stage_number': 0}
    original_run = namespace['run_job']
    def unchanged_run(job, *arguments):
        current.update(job=job, stage_number=0)
        output = ROOT / job['output_directory']
        if output.exists() and not (output / 'result.json').exists():
            # Do not overwrite any incomplete receipt/failure from a reboot.
            for stage_dir in (output / 'stages').glob('*'):
                receipt = stage_dir / 'receipt.json'
                if not receipt.exists() or json.loads(receipt.read_text(encoding='utf-8')).get('status') != 'completed':
                    update(dict(state='partial_attempt_preserved', original_job=job['id']))
                    return 'failed_or_partial_preserved'
        return original_run(job, *arguments)
    def launch(command, *a, **kw):
        job = current['job']
        if job is None or Path(command[2]).name != 'run_anomaly.py':
            raise ValueError('Only the unchanged original MaelNet author stage may be launched')
        stage_id = Path(kw['stdout'].name).parent.name
        if stage_id not in {s['id'] for s in job['stages']}:
            raise ValueError('Author stage log path does not match registered pipeline')
        identity = job['id'] + '__' + stage_id
        receipt = base / 'receipts' / (identity + '.json')
        if receipt.exists():
            raise FileExistsError('Previous resource receipt must remain intact')
        update(dict(state='waiting_resource_slot', requested_job=job['id'], requested_command=command))
        reservation = api['resource_slot'](ROOT / registration['resource_lock'], registration['settings'], update)
        reservation.__enter__()
        try:
            process = subprocess.Popen(command, *a, **kw)
            handle = psutil.Process(process.pid)
            metadata = dict(original_job=job['id'], original_stage=stage_id, resource_stage_identity=identity,
                command=[str(x) for x in command], cwd=str(kw['cwd']), pid=process.pid,
                create_time=handle.create_time(), queue_sha256=digest(queue_path),
                full_registered_budget_retained=True)
            update(dict(state='running', active_pid=process.pid, active_create_time=handle.create_time(),
                active_job=job['id'], actual_command=handle.cmdline()))
            return GuardedChild(process, reservation, api, registration['settings'], update, receipt, metadata)
        except BaseException:
            reservation.__exit__(*sys.exc_info()); raise
    proxy = SimpleNamespace(**vars(subprocess)); proxy.Popen = launch
    namespace.update(subprocess=proxy, run_job=unchanged_run)
    sys.argv = ['run_maelnet_author_queue.py', '--queue', str(queue_path)]
    try:
        namespace['main']()
        update(dict(state='controller_terminal', active_pid=None))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
