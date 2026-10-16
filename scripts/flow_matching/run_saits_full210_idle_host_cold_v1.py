"""Hand off only the exact idle full210 metadata owner, preserving all workers.

Compile unchanged scheduler functions without importing unused data libraries.
The saved-epoch special case retains its exact pending resume preconditions.
Original native commands, guards, per-job order and every210 config stay intact.
"""
import argparse
import ast
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write
from scripts.flow_matching import run_light_controller as loader
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api

MODULE = 'scripts.flow_matching.run_saits_full210_idle_host_cold_v1'
CONFIG = 'configs/runtime/fm_saits_full210_idle_host_cold.v1.json'
TEST = 'tests/test_fm_saits_full210_idle_host_cold_v1.py'
RUNTIME = 'configs/runtime/fm_saits_saved_epoch271_full210_recovery.v2.json'
ORIGINAL = 'scripts/flow_matching/run_saits_guarded_original_queue_v1.py'
RESUMER = 'scripts/flow_matching/run_saits_epoch271_full210_recovery_v2.py'
SELECTED = 'saits_paper_brits_electricity_author_missing0.2_seed27'
PREVIOUS = 'configs/runtime/fm_saits_guarded_original_resume.v1.json'
OWNER = dict(pid=2972, create_time=1791918817.7624688,
    command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe', '-X', 'utf8', '-u', '-m',
        'scripts.flow_matching.run_saits_epoch271_full210_recovery_v2', '--runtime-config', RUNTIME])


def selected_resume_binding(runtime):
    queue = read(ROOT / runtime['queue'])
    item = next(j for j in queue['jobs'] if read(ROOT / j['config'])['id'] == SELECTED)
    config = read(ROOT / item['config'])
    path = ROOT / config['output_root'] / 'result.json'
    if path.exists():
        result = read(path)
        if (result['status'] != 'completed' or result['config_sha256'] != item['config_sha256']
                or result['config'] != config):
            raise ValueError('Unexpected selected recovery output; never bypass its resume guard')
        return dict(id=SELECTED, state='completed', path=str(path), sha256=sha(path), original_config_sha256=item['config_sha256'])
    proof_path = ROOT / runtime['saved_epoch_proof']
    proof = read(proof_path)
    if (proof['canonical_id'] != SELECTED or proof['full_saved_epoch_boundary'] != 271
            or proof['saved_optimizer_updates'] != 1897
            or sha(proof['checkpoint_path']) != proof['checkpoint_sha256']
            or sha(proof['best_checkpoint_path']) != proof['best_checkpoint_sha256']):
        raise ValueError('Pending exact epoch271 checkpoint/optimizer/RNG binding must remain unchanged')
    return dict(id=SELECTED, state='pending_exact_saved_epoch271_resume', proof_path=str(proof_path),
        proof_sha256=sha(proof_path), saved_epoch=271, saved_optimizer_updates=1897,
        checkpoint_sha256=proof['checkpoint_sha256'], best_checkpoint_sha256=proof['best_checkpoint_sha256'],
        original_config_sha256=item['config_sha256'])


def registration():
    runtime = read(ROOT / RUNTIME)
    queue = read(ROOT / runtime['queue'])
    if len(queue['jobs']) != 210:
        raise ValueError('All210 original native jobs required')
    paths = [RUNTIME, ORIGINAL, RESUMER, MODULE.replace('.', '/') + '.py', TEST,
        'scripts/flow_matching/run_light_controller.py',
        'scripts/flow_matching/run_spectral_original_queue.py',
        'scripts/flow_matching/run_spectral_table2_guarded_queue_v2.py',
        'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
        'scripts/flow_matching/phase_gate.py']
    return dict(original_runtime=RUNTIME, previous_owner=OWNER,
        previous_state_path=runtime['state_root'] + '/status.json',
        handoff_root='experiments/runs/fm_saits_full210_idle_host_cold_v1',
        state_root='F:/aicoding/IIA_Data/experiments/saits_full210_idle_host_cold_v1',
        original_saved_epoch_condition=selected_resume_binding(runtime),
        required_original_jobs=210, queue_sha256=runtime['queue_sha256'],
        settings=runtime['settings'], selected_recovery_admission_settings=runtime['selected_recovery_admission_settings'],
        resource_lock=runtime['resource_lock'], original_queue_lock=runtime['original_queue_lock'],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        original_scope=dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273, paper_scalar_targets=702),
        numerical_source_or_commands_changed=False, all_original_experiments_complete=False)


def namespace(c):
    # This helper is also used before handoff for source/config/AST validation.
    ns = dict(globals(), digest=loader.digest, load_functions=loader.load_functions)
    for path, names in [
        ('scripts/flow_matching/phase_gate.py', ['check_queue']),
        ('scripts/flow_matching/run_resource_gated_light_controller_v1.py', ['write']),
        ('scripts/flow_matching/run_spectral_original_queue.py', ['guarded_api']),
        ('scripts/flow_matching/run_spectral_table2_guarded_queue_v2.py', ['complete_resource_api']),
        (ORIGINAL, ['read', 'verify_binding', 'main'])]:
        loader.load_functions(ROOT, path, names, ns)
    raw_read, original_main = ns['read'], ns['main']
    runtime = raw_read(ROOT / c['original_runtime'])
    ns['original'] = SimpleNamespace(main=original_main, read=raw_read, ROOT=ROOT,
        verify_binding=ns['verify_binding'], digest=loader.digest)
    loader.load_functions(ROOT, RESUMER, ['verify', 'predecessor_absent', 'saved_epoch_still_current',
        'job_admission', 'scheduler_jobs', 'controller_ast'], ns)
    ns['_job_admission'], ns['_scheduler_jobs'] = ns['job_admission'], ns['scheduler_jobs']
    def private_read(path):
        result = raw_read(path)
        if Path(path).resolve() == (ROOT / c['original_runtime']).resolve():
            return dict(result, state_root=c['state_root'])
        return result
    ns['read'] = private_read
    return ns, runtime


def verify(c):
    if c != registration():
        raise ValueError('Full210 exact idle handoff/source/data/guard contract changed')
    ns, runtime = namespace(c)
    queue, proof, incident = ns['verify'](runtime)
    ns['predecessor_absent'](incident)
    if c['original_saved_epoch_condition']['state'] == 'pending_exact_saved_epoch271_resume':
        ns['saved_epoch_still_current'](proof)
    ns['controller_ast']()
    if len(ns['scheduler_jobs'](queue, runtime)) != 210:
        raise ValueError('No completed or pending original seed/config may be omitted')
    return ns, runtime, queue


def exact_owner(process, expected):
    return process.pid == expected['pid'] and process.create_time() == expected['create_time'] and process.cmdline() == expected['command']


def idle_allowed(state, expected, children):
    if (state.get('pid') != expected['pid'] or state.get('create_time') != expected['create_time']
            or state.get('active_pid') is not None or state.get('shared_lock_owned') is not False
            or state.get('state') not in {'waiting_for_memory', 'waiting_for_heavy_lock', 'waiting_for_resource_lock', 'between_jobs'}):
        return False
    return all(p['name'].lower() == 'conhost.exe'
        and Path(p['exe']).as_posix().lower() == 'c:/windows/system32/conhost.exe' for p in children)


def descendants(process):
    return [dict(pid=p.pid, create_time=p.create_time(), name=p.name(), exe=p.exe(), command=p.cmdline())
            for p in process.children(recursive=True)]


def handoff(c, base):
    expected = c['previous_owner']
    while True:
        process = psutil.Process(expected['pid'])
        if not exact_owner(process, expected):
            raise RuntimeError('Old PID/creation time/full command changed; never terminate it')
        state = read(ROOT / c['previous_state_path'])
        children = descendants(process)
        if not idle_allowed(state, expected, children):
            write(base / 'handoff_status.json', dict(state='waiting_verified_idle_metadata_owner', previous_state=state,
                descendants=children, original_numeric_workers_stopped=False))
            time.sleep(5)
            continue
        process.suspend()
        stopped = False
        try:
            state = read(ROOT / c['previous_state_path'])
            children = descendants(process)
            if not exact_owner(process, expected) or not idle_allowed(state, expected, children):
                continue
            if selected_resume_binding(read(ROOT / c['original_runtime'])) != c['original_saved_epoch_condition']:
                raise ValueError('Actual selected saved-epoch recovery state changed')
            memory = process.memory_info()
            from scripts.flow_matching.collect_crossad_CUDA_full139_verified_v1 import exclusive
            exclusive(base / 'previous_status_at_verified_handoff.json', state)
            receipt = dict(previous_owner=expected, previous_descendants=children,
                previous_private_bytes=memory.private, previous_rss_bytes=memory.rss,
                no_numerical_descendants_verified=True, idle_heavy_lock_unowned=True,
                original_numeric_workers_suspended_or_stopped=False,
                native_worker_source_or_model_config_changed=False)
            process.terminate()
            process.wait(timeout=15)
            stopped = True
            exclusive(base / 'actual_handoff.json', receipt)
            write(ROOT / c['previous_state_path'], dict(state, state='stopped_for_verified_low_memory_handoff',
                successor_configuration=CONFIG, successor_state_root=c['state_root'],
                successor_pid=os.getpid(), original_numeric_workers_stopped=False))
            return
        finally:
            if not stopped and process.is_running():
                process.resume()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        from scripts.flow_matching.collect_crossad_CUDA_full139_verified_v1 import exclusive
        exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    _, runtime, queue = verify(c)
    if args.probe_only:
        print(json.dumps(dict(required_original_jobs=len(queue['jobs']),
            private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            native_worker_commands_unchanged=True)))
        return
    base = ROOT / c['handoff_root']
    base.mkdir(parents=True, exist_ok=False)
    handoff(c, base)
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    ns, runtime = namespace(c)
    tree = ns['controller_ast']()
    write(base / 'actual_source_binding.json', dict(original_runtime=c['original_runtime'],
        original_runtime_sha256=sha(ROOT / c['original_runtime']), successor_configuration_sha256=sha(ROOT / CONFIG),
        unchanged_numeric_settings=runtime['settings'], effective_state_root=c['state_root'],
        all210_jobs_retained=True, source_controller_AST_reuses_exact_existing_admission_and_order=True))
    exec(compile(tree, __file__, 'exec'), ns)
    sys.argv = ['unchanged_original_full210_controller', '--runtime-config', c['original_runtime']]
    ns['main']()


if __name__ == '__main__':
    main()
