"""Repeat original-guard review windows for every ready original CFM slot.

Only reserve the family lock between naturally completed models. Never stop a
model, change any scientific recipe, lower a guard, or wait for untrained slots
while holding the family. Preserve both full285 route maps and canonical IDs.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import queued_family_lock_v1 as queued
from scripts.flow_matching.reserve_cfm_complete_review_window_v1 import inspect_targets, window_action
from scripts.flow_matching.run_light_controller import load_functions

ROOT, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
_fingerprint_namespace = dict(hashlib=hashlib, json=json)
load_functions(ROOT, "scripts/flow_matching/run_cfm_ts_original_job.py", ["fingerprint"], _fingerprint_namespace)
fingerprint = _fingerprint_namespace["fingerprint"]


def ready_source(job, result, receipt):
    return (receipt.get('exit_code') == 0 and receipt.get('complete') is True
        and not receipt.get('resource_observations', {}).get('resource_guard_aborted', True)
        and result.get('status') == 'completed' and result.get('job_sha256') == fingerprint(job)
        and result.get('epochs_completed') == job['epochs']
        and result.get('optimizer_updates') == job['expected_optimizer_updates']
        and result.get('test_prediction_shape') == [job['test_trajectories'], job['test_points_per_trajectory'], job['observed_dimensions']]
        and result.get('diagnostic_smoke') is False and result.get('author_equivalence_certified') is False)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']: raise ValueError('Frozen recurring-window registration changed')
    runtimes = [read(ROOT / path) for path in config['review_runtimes']]
    source = read(ROOT / config['original_queue'])
    if (len(source['jobs']) != 285 or any(len(r['routes']) != 285 or r['settings'] != config['review_settings']
        or r['original_scope'] != config['original_scope'] for r in runtimes)
        or config['training_admission_settings'] != source['resource_policy']
        or config['family_lock'] != read(ROOT / config['original_training_runtime'])['family_lock']):
        raise ValueError('All285 routes, original family/source policy and identical full review guards required')
    if any({r['original_canonical_id'] for r in runtime['routes']} != {j['id'] for j in source['jobs']} for runtime in runtimes):
        raise ValueError('Every full canonical source slot must remain')
    return runtimes


def backlog(runtimes):
    records = {}
    routes = {}
    for runtime in runtimes:
        for record in read(ROOT / runtime['state_root'] / 'verified_full_model_records.json')['records']:
            identity = record['original_canonical_id']
            if identity in records and records[identity] != record: raise ValueError('Immutable canonical proof conflict')
            records[identity] = record
        for route in runtime['routes']:
            identity = route['original_canonical_id']
            if identity not in routes or route['job']['id'] != identity: routes[identity] = route
    ready = []
    for identity, route in routes.items():
        if identity in records: continue
        job = route['job'];resultpath = ROOT / route['result_path']
        receiptpath = ROOT / Path(job['output_directory']).parent.parent / 'receipts' / (job['id'] + '.json')
        if resultpath.exists() and receiptpath.exists() and ready_source(job, read(resultpath), read(receiptpath)):
            ready.append(identity)
    return sorted(ready), records


def live_original(config):
    import psutil
    expected = config['original_source_owner']
    try:
        process = psutil.Process(expected['pid'])
        if abs(process.create_time() - expected['create_time']) >= .001 or process.cmdline() != expected['command']:
            raise ValueError('Original source handle reused; no re-entry or stop permitted')
    except psutil.NoSuchProcess:
        return False, None
    state = read(ROOT / config['original_source_state'])
    return True, state.get('active_job')


def main():
    import os
    import psutil
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True);parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args();config = read(ROOT / args.config);runtimes = verify(config)
    ready, records = backlog(runtimes)
    if args.probe_only:
        print(dict(original_slots=285, ready_unclosed=ready, closed_union=len(records), training_executed=False));return
    base = ROOT / config['state_root'];base.mkdir(parents=True, exist_ok=True)
    api = cold.cached_api();owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('Recurring original-window owner already exists')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), configuration_sha256=sha(ROOT / args.config),
                 original_registered_models=285, windows_completed=0, numeric_workers_stopped_or_restarted=0,
                 original_guards_changed=False, shared_lock_owned=False, all_original_experiments_complete=False)
    def update(values):
        state.update(values);cold.original.write(base / 'status.json', state)
    try:
        while True:
            ready, records = backlog(runtimes);alive, active = live_original(config)
            if not ready and not active:
                update(dict(state='waiting_ready_sources_or_actual_original_model', ready_unclosed=ready, closed_union=len(records)))
                if not alive: break
                time.sleep(config['poll_seconds']);continue
            update(dict(state='queuing_next_natural_original_family_release', ready_unclosed=ready, original_active_job=active))
            with queued.resource_slot(ROOT / config['family_lock'], config['training_admission_settings'], api, update, 10):
                started = time.monotonic();targets = []
                update(dict(shared_lock_owned=True, state='selecting_only_complete_source_backlog'))
                # Source receipt is published just after the producer releases
                # its family lease. Give only that metadata a bounded grace.
                for _ in range(config['source_receipt_grace_polls']):
                    targets, records = backlog(runtimes)
                    if targets or not api['has_headroom'](api['resource_snapshot'](), config['review_settings']): break
                    time.sleep(config['poll_seconds'])
                while True:
                    pending, failed, workers, records = inspect_targets(runtimes, targets)
                    snapshot = api['resource_snapshot']();elapsed = time.monotonic() - started
                    action = window_action(pending, failed, api['has_headroom'](snapshot, config['review_settings']), bool(workers),
                        snapshot['commit_available_bytes'] >= config['review_settings']['emergency_commit_headroom_bytes'],
                        elapsed, config['maximum_window_seconds'])
                    update(dict(state=action, current_ready_targets=targets, target_pending=pending, failed_targets=failed,
                                actual_numeric_review_handles=workers, elapsed_window_seconds=elapsed,
                                closed_union=len(records), **snapshot))
                    if action != 'allow_original_full_reviews': break
                    time.sleep(config['poll_seconds'])
            state['windows_completed'] += 1
            cold.original.write(base / ('window_' + str(state['windows_completed']) + '_receipt.json'),
                dict(state, shared_lock_owned=False, original_next_training_allowed=True, captured_utc=datetime.now(timezone.utc).isoformat()))
            update(dict(shared_lock_owned=False, state='between_natural_release_windows'))
            if failed:
                update(dict(state='unresolved_review_failure_preserved'));break
            time.sleep(config['poll_seconds'])
    finally:
        if state['shared_lock_owned']: update(dict(shared_lock_owned=False, state='window_exception_lease_released'))
        owner.close()


if __name__ == '__main__': main()
