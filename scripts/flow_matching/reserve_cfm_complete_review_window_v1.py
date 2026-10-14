"""Reserve a bounded natural family-release window for unchanged full reviews.

Never stop/restart a model, change source/solver/grid/budget or lower a guard.
Queue with the original training-family policy; after grant require the exact
original review headroom. Release if unable to admit and no real review runs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold
from scripts.flow_matching import queued_family_lock_v1 as queued

ROOT, read, sha = cold.ROOT, cold.original.read, cold.streaming_sha
TARGETS = ['cfm_ts_original__appendix_d__gp_gaussian_conditional_e400__sine__seed' + str(s) for s in [1106, 1107]] + [
    'cfm_ts_original__main_text__bb_paper_literal_e400__sine__seed' + str(s) for s in [1104, 1106]]


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen review-window registration changed: ' + pin['path'])
    primary, supplement, original = [read(ROOT / config[key]) for key in ['primary_runtime', 'supplement_runtime', 'original_queue']]
    if (config['target_canonical_ids'] != TARGETS or len(primary['routes']) != 285 or len(supplement['routes']) != 285
            or primary['original_scope'] != config['original_scope'] or supplement['original_scope'] != config['original_scope']
            or config['training_admission_settings'] != original['resource_policy'] or config['review_settings'] != primary['settings']
            or supplement['settings'] != primary['settings'] or config['family_lock'] != read(ROOT / config['original_training_runtime'])['family_lock']):
        raise ValueError('Every285 route, original family policy and both unchanged review guard policies required')
    return primary, supplement


def window_action(pending, failed, headroom, numeric_live, emergency_ok, elapsed, limit):
    if failed: return 'release_unresolved_failure'
    if not pending: return 'release_all_target_proofs_complete'
    if not emergency_ok: return 'release_emergency_headroom'
    if elapsed >= limit: return 'release_window_deadline_only'
    if not headroom and not numeric_live: return 'release_no_original_review_headroom'
    return 'allow_original_full_reviews'


def inspect_targets(runtimes, targets):
    import psutil
    records, failed, workers = {}, [], []
    for runtime in runtimes:
        snapshot = ROOT / runtime['state_root'] / 'verified_full_model_records.json'
        for record in read(snapshot)['records']:
            identity = record['original_canonical_id']
            if identity in records and records[identity] != record: raise ValueError('Conflicting immutable canonical proof')
            records[identity] = record
        state = read(ROOT / runtime['state_root'] / 'status.json')
        failed.extend(identity for identity in targets if identity in state['failed_preserved'])
        for identity in targets:
            for suffix, flag in [('native', '--worker'), ('review', '--review-worker')]:
                configpath = ROOT / runtime['child_config_root'] / (identity + '.' + suffix + '.json')
                if not configpath.exists(): continue
                config = read(configpath)
                statuspath = ROOT / config['state_root'] / 'status.json'
                if not statuspath.exists(): continue
                status = read(statuspath)
                if not status.get('active_pid'): continue
                try:
                    process = psutil.Process(status['active_pid'])
                    command = process.cmdline()
                    if (abs(process.create_time() - status['active_create_time']) < .001
                            and command == status['actual_command'] and flag in command):
                        workers.append(dict(pid=process.pid, create_time=process.create_time(), actual_command=command))
                except psutil.NoSuchProcess: pass
    for identity in targets:
        routes = [route for runtime in runtimes for route in runtime['routes'] if route['original_canonical_id'] == identity]
        chosen = next((route for route in routes if route['job']['id'] != identity), routes[0])
        job = chosen['job']
        receiptpath = ROOT / Path(job['output_directory']).parent.parent / 'receipts' / (job['id'] + '.json')
        if receiptpath.exists():
            receipt = read(receiptpath)
            if receipt['exit_code'] != 0 or not receipt['complete']:
                failed.append(identity)
    return [i for i in targets if i not in records], sorted(set(failed)), workers, records


def main():
    import os
    import psutil
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = read(ROOT / args.config)
    runtimes = verify(config)
    pending, failed, workers, _ = inspect_targets(runtimes, config['target_canonical_ids'])
    if args.probe_only:
        print(dict(original_models=285, target_pending=pending, unchanged_review_guards=True, models_stopped=0)); return
    base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    api = cold.cached_api(); owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None: raise RuntimeError('A review-window owner already exists')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), configuration_sha256=sha(ROOT / args.config),
                 original_registered_models=285, models_stopped_or_restarted=0, original_guards_changed=False,
                 all_original_experiments_complete=False)
    def update(values):
        state.update(values); cold.original.write(base / 'status.json', state)
    try:
        update(dict(state='queuing_same_original_family_lock', target_pending=pending, shared_lock_owned=False))
        with queued.resource_slot(ROOT / config['family_lock'], config['training_admission_settings'], api, update, 10):
            started = time.monotonic()
            update(dict(state='original_family_release_acquired_for_complete_reviews', shared_lock_owned=True))
            while True:
                pending, failed, workers, records = inspect_targets(runtimes, config['target_canonical_ids'])
                snapshot = api['resource_snapshot']()
                elapsed = time.monotonic() - started
                action = window_action(pending, failed, api['has_headroom'](snapshot, config['review_settings']), bool(workers),
                    snapshot['commit_available_bytes'] >= config['review_settings']['emergency_commit_headroom_bytes'],
                    elapsed, config['maximum_window_seconds'])
                update(dict(state=action, target_pending=pending, failed_targets=failed, actual_numeric_review_handles=workers,
                            elapsed_window_seconds=elapsed, **snapshot))
                if action != 'allow_original_full_reviews':
                    if not pending and not failed:
                        for identity in config['target_canonical_ids']:
                            runtime = next(r for r in runtimes if any(route['original_canonical_id'] == identity
                                and route['job']['id'] == records[identity]['execution_id'] for route in r['routes']))
                            cold.validate_closed_primary_record(runtime, records[identity])
                        cold.original.write(base / 'complete_target_proof_union.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                            records=[records[i] for i in config['target_canonical_ids']], full_unique_canonical_union=len(records),
                            every_original_model_grid_and_guard_unchanged=True, all_original_experiments_complete=False))
                    break
                time.sleep(config['poll_seconds'])
        update(dict(state=action, shared_lock_owned=False, original_next_training_allowed=True))
    finally:
        owner.close()


if __name__ == '__main__': main()
