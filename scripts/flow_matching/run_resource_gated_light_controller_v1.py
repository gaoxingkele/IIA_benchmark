"""Gate unchanged CPU model commands per job, without importing model libraries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class GuardedChild:
    """Keep original poll/wait behavior while owning the CPU slot until exit."""
    def __init__(self, process, reservation, api, settings, update, receipt, metadata):
        self.process, self.reservation, self.api = process, reservation, api
        self.settings, self.update, self.receipt, self.metadata = settings, update, receipt, metadata
        self.released = False
        self.peaks = dict(peak_tree_rss_bytes=0, peak_tree_private_bytes=0,
                          minimum_commit_available_bytes=None, resource_guard_aborted=False)

    def __getattr__(self, name):
        return getattr(self.process, name)

    def poll(self):
        code = self.process.poll()
        if code is None:
            memory = self.api['resource_snapshot']()
            available = memory['commit_available_bytes']
            prior = self.peaks['minimum_commit_available_bytes']
            self.peaks['minimum_commit_available_bytes'] = available if prior is None else min(prior, available)
            try:
                parent = psutil.Process(self.process.pid)
                entries = [parent, *parent.children(recursive=True)]
                values = [p.memory_info() for p in entries]
                self.peaks['peak_tree_rss_bytes'] = max(self.peaks['peak_tree_rss_bytes'], sum(v.rss for v in values))
                self.peaks['peak_tree_private_bytes'] = max(self.peaks['peak_tree_private_bytes'],
                    sum(getattr(v, 'private', v.vms) for v in values))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if available < self.settings['emergency_commit_headroom_bytes']:
                self.peaks.update(resource_guard_aborted=True, abort_resource_snapshot=memory)
                self.api['terminate_tree'](self.process)
                code = self.process.poll()
            self.update(dict(state='running' if code is None else 'terminal', **memory, **self.peaks))
        if code is not None and not self.released:
            write(self.receipt, dict(**self.metadata, **self.peaks, exit_code=code,
                completed_utc=datetime.now(timezone.utc).isoformat(), benchmark_result=False))
            self.reservation.__exit__(None, None, None)
            self.released = True
            self.update(dict(state='between_jobs', active_pid=None, exit_code=code, **self.peaks))
            time.sleep(self.settings.get('controller_yield_seconds', 2))
        return code

    def wait(self, timeout=None):
        start = time.monotonic()
        while True:
            code = self.poll()
            if code is not None:
                return code
            if timeout is not None and time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            time.sleep(min(self.settings.get('guard_poll_seconds', 2),
                max(.001, timeout - (time.monotonic() - start))) if timeout is not None
                else self.settings.get('guard_poll_seconds', 2))


def guarded_factory(api, settings, lock, state_root, update, queue_path, queue):
    ids = {j['id'] for j in queue['jobs']}
    def launch(command, *args, **kwargs):
        command = [str(x) for x in command]
        job_id = command[command.index('--job-id') + 1]
        supplied = Path(command[command.index('--queue') + 1])
        if job_id not in ids or (ROOT / supplied).resolve() != queue_path.resolve():
            raise ValueError('Only the unchanged registered model job may enter this CPU slot')
        receipt = state_root / 'receipts' / (job_id + '.json')
        if receipt.exists():
            raise FileExistsError('Preserve resource receipt for prior full attempt')
        update(dict(state='waiting_resource_slot', requested_job=job_id, requested_command=command))
        reservation = api['resource_slot'](lock, settings, update)
        reservation.__enter__()
        try:
            process = subprocess.Popen(command, *args, **kwargs)
            handle = psutil.Process(process.pid)
            metadata = dict(job_id=job_id, pid=process.pid, create_time=handle.create_time(),
                command=command, queue_sha256=digest(queue_path), full_registered_budget_retained=True)
            update(dict(state='running', active_pid=process.pid, active_create_time=handle.create_time(),
                actual_command=handle.cmdline(), active_job=job_id))
            return GuardedChild(process, reservation, api, settings, update, receipt, metadata)
        except BaseException:
            reservation.__exit__(*sys.exc_info())
            raise
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.runtime_config
    registration = json.loads(path.read_text(encoding='utf-8'))
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen resource adapter source differs: ' + entry['path'])
    original = json.loads((ROOT / registration['original_runtime']).read_text(encoding='utf-8'))
    controller, namespace, binding = load_controller(ROOT, original, args.controller)
    queue_path = ROOT / binding['queue_path']
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    if args.probe_only:
        print(json.dumps(dict(controller=args.controller, registered_jobs=len(queue['jobs']),
            queue_sha256=digest(queue_path), torch_imported='torch' in sys.modules,
            numpy_imported='numpy' in sys.modules, private_bytes=psutil.Process().memory_info().private)))
        return
    base = ROOT / registration['state_root'] / args.controller
    api = resource_api()
    lane = api['exclusive_lock'](base / 'adapter.lock')
    if lane is None:
        raise RuntimeError('This CPU adapter already has a live owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
        runtime_sha256=digest(path), queue_sha256=digest(queue_path), state='starting')
    def update(values):
        state.update(values); write(base / 'status.json', state)
    proxy = SimpleNamespace(**vars(subprocess))
    proxy.Popen = guarded_factory(api, registration['settings'], ROOT / registration['resource_lock'],
        base, update, queue_path, queue)
    namespace['subprocess'] = proxy
    sys.argv = [binding['source_path'], binding['queue_argument'], binding['queue_path']]
    try:
        result = controller.main()
        update(dict(state='controller_terminal', original_returncode=result, active_pid=None))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
