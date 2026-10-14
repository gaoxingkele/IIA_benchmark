"""Queue metadata coordination at the original training boundary; score at30GiB.

No numerical worker, scoring guard, model, solver, grid or budget changes.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

from scripts.flow_matching import run_cfm_saved_full_score_recovery_v1 as previous

ROOT = previous.ROOT


def verify(config):
    primary = previous.verify(config)
    baseline = previous.cold.original.read(ROOT / config['previous_scoring_configuration'])
    training = previous.cold.original.read(ROOT / config['original_training_queue'])
    for key in ['job', 'original_job', 'checkpoint_path', 'checkpoint_sha256', 'settings',
                'python', 'cpu_threads', 'environment_lock', 'resource_lock', 'family_lock']:
        if config[key] != baseline[key]:
            raise ValueError('Original saved model, full scoring and numerical limits must remain: ' + key)
    if config['family_coordination_settings'] != training['resource_policy']:
        raise ValueError('Metadata queue must use exactly the original training admission policy')
    if (config['family_coordination_settings']['minimum_commit_headroom_bytes'] != 22*1024**3
            or config['settings']['minimum_commit_headroom_bytes'] != 30*1024**3):
        raise ValueError('Original22GiB family queue and stronger30GiB scoring admission are separate')
    runtime = previous.cold.original.read(ROOT / config['complete_review_runtime'])
    selected = next(r for r in runtime['routes'] if r['original_canonical_id'] == previous.IDENTITY)
    if (len(runtime['routes']) != 285 or selected['job'] != config['job']
            or runtime['settings'] != config['settings'] or config['maximum_review_window_seconds'] != 900):
        raise ValueError('Full285 unchanged native/independent review routes and bounded window required')
    return primary


def drain_ready_reviews(config, api, update=None):
    from scripts.flow_matching import reserve_cfm_complete_review_window_v1 as window
    if not (ROOT / config['job']['output_directory'] / 'result.json').is_file():
        return
    runtime = previous.cold.original.read(ROOT / config['complete_review_runtime'])
    started = time.monotonic()
    while True:
        pending, failed, workers, records = window.inspect_targets([runtime], [previous.IDENTITY])
        memory = api['resource_snapshot']()
        elapsed = time.monotonic()-started
        action = window.window_action(pending, failed, api['has_headroom'](memory, config['settings']), bool(workers),
            memory['commit_available_bytes'] >= config['settings']['emergency_commit_headroom_bytes'],
            elapsed, config['maximum_review_window_seconds'])
        if update:
            update(dict(memory, state=action, target_pending=pending, failed_targets=failed,
                actual_numeric_review_handles=workers, elapsed_review_window_seconds=elapsed,
                numerical_workers_stopped_or_restarted=0))
        if action != 'allow_original_full_reviews':
            if not pending and not failed:
                previous.cold.validate_closed_primary_record(runtime, records[previous.IDENTITY])
            previous.safe_write(ROOT/config['state_root']/'post_score_full_review_window_receipt.json',
                dict(state=action, targets=[previous.IDENTITY], elapsed_seconds=elapsed,
                    numerical_workers_stopped_or_restarted=0, original_guards_changed=False,
                    records=[records[previous.IDENTITY]] if not pending and not failed else [],
                    all_original_experiments_complete=False))
            return
        time.sleep(2)


@contextmanager
def coordinated_slot(path, config, settings, api, update=None, poll_seconds=2, acquire_slot=None, sleep=time.sleep, finish=None):
    if Path(path).resolve() != (ROOT / config['family_lock']).resolve() or settings != config['settings']:
        raise ValueError('Only the frozen family coordination path may be intercepted')
    acquire_slot = acquire_slot or previous.queued.resource_slot
    while True:
        with acquire_slot(path, config['family_coordination_settings'], api, update) as snapshot:
            # Recheck the stronger original scoring guard after the lease grant.
            # If it is not met, release immediately; never park a numeric model.
            snapshot = api['resource_snapshot']()
            if api['has_headroom'](snapshot, settings):
                if update:
                    update(dict(snapshot, state='original_full_scoring_headroom_rechecked',
                        original_scoring_guards_changed=False, additional_optimizer_updates=0))
                yield snapshot
                (finish or drain_ready_reviews)(config, api, update)
                return
            if update:
                update(dict(snapshot, state='release_family_before_original_scoring_headroom',
                    original_scoring_guards_changed=False, active_pid=None))
        sleep(poll_seconds)


def namespace(config):
    def slot(path, settings, api, update=None, poll_seconds=10):
        return coordinated_slot(path, config, settings, api, update)
    values = dict(previous.__dict__, verify=verify, queued=SimpleNamespace(resource_slot=slot))
    return FunctionType(previous.controller.__code__, values)


def idle_allowed(role, state, output_exists, descendants):
    return (role == 'original' and state.get('active_pid') is None and not state.get('shared_lock_owned')
        and state.get('state') in {'waiting_for_memory', 'waiting_for_family_lock_kernel_queue'}
        and not output_exists and all(previous.cold.handoff.allowed_console_child(p['name'], p['exe']) for p in descendants))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = previous.cold.original.read(ROOT / args.config)
    verify(config)
    if args.probe_only:
        import sys, psutil
        print(json.dumps(dict(original_slots=285, saved_model_only=True, additional_optimizer_updates=0,
            family_metadata_queue_commit_bytes=config['family_coordination_settings']['minimum_commit_headroom_bytes'],
            actual_scoring_commit_bytes=config['settings']['minimum_commit_headroom_bytes'],
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            private_bytes=psutil.Process().memory_info().private)))
        return
    base = ROOT / config['handoff_state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = previous.cold.cached_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Exact saved-score coordination handoff already owned')
    try:
        fn = FunctionType(previous.cold.handoff.safe_handoff.__code__,
            dict(previous.cold.handoff.__dict__, handoff_phase_allowed=idle_allowed))
        route = dict(previous_owner=config['previous_owner'], previous_state_path=config['previous_state_path'],
            output_directory=config['job']['output_directory'], runtime_configuration=args.config)
        fn('original', route, base, lambda s: previous.safe_write(base / 'status.json', s), 2)
        verify(config)
        namespace(config)(config, args.config)
    finally:
        owner.close()


if __name__ == '__main__':
    main()
