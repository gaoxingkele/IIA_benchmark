"""Retire only the exact idle metadata collector; keep all full139 checks.

Compile the original file-lock function without importing its data-preparation
module. Original condition/cohort validators and immutable result files remain
unchanged. Original CUDA/CPU trainers are never suspended or stopped.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import psutil

from scripts.flow_matching import collect_crossad_CUDA_full139_verified_v1 as original
from scripts.flow_matching.run_light_controller import load_functions
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.collect_crossad_full139_cold_handoff_v1'
CONFIG = 'configs/runtime/fm_crossad_full139_cold_collector_handoff.v1.json'
EXPECTED_OWNER = dict(pid=37280, create_time=1792129142.1064477,
    command=['C:/Users/10175/AppData/Local/Programs/Python/Python312/python.exe', '-B', '-X',
             'utf8', '-u', '-m', 'scripts.flow_matching.collect_crossad_CUDA_full139_verified_v1'])


def registration():
    parent = read(ROOT / original.CONFIG)
    return dict(previous_configuration=original.CONFIG, previous_configuration_sha256=sha(ROOT / original.CONFIG),
        previous_owner=EXPECTED_OWNER, previous_state_path=parent['state_root'] + '/status.json',
        state_root='experiments/runs/fm_crossad_full139_cold_collector_handoff_v1',
        unchanged_artifact_root=parent['artifact_root'], required_conditions=139,
        original_scope=parent['original_scope'], expected_complete_primary_cohorts=22,
        individual_result_files_append_only=True, original_condition_cohort_validators_unchanged=True,
        original_CPU_completion_credit=0, all_original_experiments_complete=False,
        citation=parent['citation'],
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in [original.CONFIG,
            original.MODULE.replace('.', '/') + '.py', MODULE.replace('.', '/') + '.py',
            'scripts/flow_matching/run_light_controller.py', 'scripts/flow_matching/run_tsad_queue.py',
            'tests/test_fm_crossad_full139_cold_handoff_v1.py']])


def verify(c):
    if c != registration():
        raise ValueError('Full139 exact collector handoff registration changed')
    parent = read(ROOT / c['previous_configuration'])
    if parent != original.registration():
        raise ValueError('Original full139 collector contract changed')
    for item in parent['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Original full139 validator dependency changed')
    q = read(ROOT / parent['queue'])
    if len(q['jobs']) != 139 or len(original.cohorts(q['jobs'], {})['primary_cohorts']) != 22:
        raise ValueError('All139 original cases and all22 five-seed cohorts required')
    return parent, q


def exact_identity(process, expected):
    return process.pid == expected['pid'] and process.create_time() == expected['create_time'] and process.cmdline() == expected['command']


def safe_idle(state, expected, descendants):
    if (state.get('pid') != expected['pid'] or state.get('create_time') != expected['create_time']
            or state.get('active_pid') is not None
            or state.get('state') != 'waiting_actual_full_source_and_checkpoint_arrays'):
        return False
    return all(d['name'].lower() == 'conhost.exe'
               and Path(d['exe']).as_posix().lower() == 'c:/windows/system32/conhost.exe'
               for d in descendants)


def descendants(process):
    return [dict(pid=p.pid, create_time=p.create_time(), name=p.name(), exe=p.exe(), command=p.cmdline())
            for p in process.children(recursive=True)]


def checked_preserved_proofs(c, parent, q, previous):
    ids = previous['verified_original_CUDA_ids']
    if len(ids) != len(set(ids)) or len(ids) != previous['verified_count']:
        raise ValueError('Exact previous unique completed-case set required')
    jobs = {j['canonical_CPU_condition_reference']: j for j in q['jobs']}
    out = Path(c['unchanged_artifact_root'])
    files = {p.stem: p for p in (out / 'conditions').glob('*.json')}
    if set(files) != set(ids) or not set(previous['errors_preserved']).issubset(jobs):
        raise ValueError('No existing condition or unresolved error may be dropped')
    preserved = {}
    for identity in ids:
        proof = original.condition(parent, q, jobs[identity])
        if proof is None or proof != read(files[identity]):
            raise ValueError('Every preserved full checkpoint/array proof must independently pass again')
        preserved[identity] = dict(path=str(files[identity]), sha256=sha(files[identity]))
    return preserved


def handoff(c, parent, q, base):
    expected = c['previous_owner']
    process = psutil.Process(expected['pid'])
    if not exact_identity(process, expected):
        raise RuntimeError('Previous collector PID reused or exact command changed')
    previous = read(ROOT / c['previous_state_path'])
    if not safe_idle(previous, expected, descendants(process)):
        raise RuntimeError('Only the exact idle metadata collector with no numerical descendants may hand off')
    process.suspend()
    stopped = False
    try:
        if not exact_identity(process, expected):
            raise RuntimeError('Suspended collector identity changed')
        previous = read(ROOT / c['previous_state_path'])
        children = descendants(process)
        if not safe_idle(previous, expected, children):
            raise RuntimeError('Collector no longer in the verified idle state')
        preserved = checked_preserved_proofs(c, parent, q, previous)
        original.exclusive(base / 'previous_status_at_verified_handoff.json', previous)
        original.exclusive(base / 'previous_proof_hashes.json', preserved)
        memory = process.memory_info()
        receipt = dict(previous_owner=expected, previous_descendants=children,
            previous_private_bytes=memory.private, previous_rss_bytes=memory.rss,
            no_numerical_descendants_verified=True, stopped_process_role='idle_metadata_collector_only',
            native_workers_suspended_or_stopped=False, individual_result_files_overwritten=False,
            exact_previous_condition_ids=list(preserved))
        process.terminate()
        process.wait(timeout=15)
        stopped = True
        original.exclusive(base / 'actual_handoff.json', receipt)
        write(ROOT / c['previous_state_path'], dict(previous, state='stopped_for_verified_low_memory_handoff',
            successor_state_root=c['state_root'], successor_configuration=CONFIG,
            successor_pid=os.getpid(), native_workers_stopped=False))
        return previous, preserved
    finally:
        if not stopped and process.is_running():
            process.resume()


def cold_lock(path):
    # Compile exactly the old lock definition; avoid importing unused NumPy,
    # pandas and their native DLLs through run_tsad_queue's module imports.
    namespace = dict(sys=sys)
    function = load_functions(ROOT, 'scripts/flow_matching/run_tsad_queue.py', ['exclusive_lock'], namespace)['exclusive_lock']
    return function(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        original.exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    parent, q = verify(c)
    if args.probe_only:
        print(json.dumps(dict(required_conditions=len(q['jobs']),
            primary_cohorts=len(original.cohorts(q['jobs'], {})['primary_cohorts']),
            private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules, pandas_imported='pandas' in sys.modules,
            torch_imported='torch' in sys.modules, numerical_experiments_run=0)))
        return
    base = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=False)
    previous, hashes = handoff(c, parent, q, base)
    lock = cold_lock(ROOT / parent['state_root'] / 'controller.lock')
    if lock is None:
        raise RuntimeError('Retired collector lane still owned; never duplicate its successor')
    out = Path(c['unchanged_artifact_root'])
    verified = {identity: read(out / 'conditions' / (identity + '.json')) for identity in hashes}
    errors = dict(previous['errors_preserved'])
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_original_conditions=139,
        all_original_experiments_complete=False, original_CPU_completion_credit=0,
        previous_exact_owner=c['previous_owner'], original_condition_cohort_validators_unchanged=True)
    def update():
        summary = original.cohorts(q['jobs'], verified)
        write(out / 'complete_primary_cohorts.json', dict(summary, verified_original_CUDA_conditions=len(verified),
            new_training_seed_slots=0, original_CPU_completion_credit=0,
            strict_TAB_result=False, all_original_experiments_complete=False))
        write(base / 'status.json', dict(state, state='waiting_actual_full_source_and_checkpoint_arrays'
            if len(verified) < 139 else 'all139_original_CUDA_results_verified_other_experiments_pending',
            verified_original_CUDA_ids=list(verified), verified_count=len(verified),
            complete_primary_cohorts=summary['complete_primary_cohorts'], errors_preserved=errors,
            private_bytes=psutil.Process().memory_info().private, numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules, captured_utc=datetime.now(timezone.utc).isoformat()))
    try:
        update()
        while len(verified) < 139:
            for job in q['jobs']:
                identity = job['canonical_CPU_condition_reference']
                if identity in verified or identity in errors:
                    continue
                try:
                    proof = original.condition(parent, q, job)
                    if proof is None:
                        continue
                    original.exclusive(out / 'conditions' / (identity + '.json'), proof)
                    verified[identity] = proof
                    update()
                except Exception as error:
                    errors[identity] = repr(error)
            update()
            time.sleep(10)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
