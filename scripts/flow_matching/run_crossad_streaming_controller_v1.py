"""Cold handoff of an idle full139 CrossAD controller with bounded hash memory.

Original numerical children/recipes/guards remain frozen. SHA256 still consumes
every byte, and cached Windows telemetry retains the original resource formula.
Only an exact PID/create-time/command controller without native children stops.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import FunctionType

from scripts.flow_matching import run_light_controller as original
from scripts.flow_matching import run_timegan_corrected_observer_dispatch_v2 as handoff
from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot

ROOT = original.ROOT


def optimized_load(root, runtime, name):
    if name != 'crossad':
        raise ValueError('Only the exact original CrossAD controller may be handed off')
    controller, namespace, binding = original.load_controller(root, runtime, name)
    namespace['sha'] = original.digest
    namespace['resource_snapshot'] = resource_snapshot
    return controller, namespace, binding


def verify(config, verify_frozen_data=True):
    for pin in config['source_receipts']:
        if original.digest(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen cold controller binding changed: ' + pin['path'])
    runtime = json.loads((ROOT / config['original_runtime']).read_text(encoding='utf-8'))
    controller, namespace, binding = optimized_load(ROOT, runtime, 'crossad')
    queue = json.loads((ROOT / binding['queue_path']).read_text(encoding='utf-8'))
    if (binding['queue_path'] != config['original_queue'] or binding['status_path'] != config['previous_state_path']
            or original.digest(ROOT / binding['queue_path']) != config['original_queue_sha256']
            or len(queue['jobs']) != 139 or len({j['id'] for j in queue['jobs']}) != 139
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('All139 exact original jobs/runtime/science/guards required')
    frozen = {p['path']: p['sha256'] for j in queue['jobs'] for p in j['frozen_files']}
    if verify_frozen_data:
        for path, digest in frozen.items():
            if namespace['sha'](ROOT / path) != digest:
                raise ValueError('Full original data/source content changed: ' + path)
    return runtime, queue, binding, len(frozen)


def idle_allowed(role, state, output_exists, descendants):
    return (role == 'original' and state.get('state') in ('waiting_for_memory', 'waiting_for_heavy_lock',
                'waiting_for_remaining_native_preflight', 'waiting_training_results')
        and all(state.get(k) is None for k in ('active_pid', 'active_job', 'active_case', 'active_stage'))
        and state.get('shared_lock_owned') is not True
        and all(handoff.allowed_console_child(p['name'], p['exe']) for p in descendants))


def bound_main():
    return FunctionType(original.main.__code__, dict(original.main.__globals__, load_controller=optimized_load))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True); parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    runtime, queue, binding, count = verify(config)
    import psutil
    if args.probe_only:
        p = psutil.Process()
        print(json.dumps(dict(full_original_jobs=len(queue['jobs']), original_frozen_files_verified=count,
            private_bytes=getattr(p.memory_info(), 'private', p.memory_info().rss),
            original_controller_main_code_identical=bound_main().__code__ is original.main.__code__,
            every_data_byte_hashed=True, resource_formula_unchanged=True,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            training_executed=False, all_original_experiments_complete=False))); return
    _, api, _ = optimized_load(ROOT, runtime, 'crossad')
    base = ROOT / config['handoff_state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None or (base / 'handoff.json').exists():
        raise RuntimeError('Existing cold handoff owner/attempt must remain')
    state = dict(pid=psutil.Process().pid, create_time=psutil.Process().create_time(),
        configuration_sha256=original.digest(ROOT / args.config), all_original_experiments_complete=False)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        handoff.write(base / 'status.json', state)
    try:
        update(dict(state='verifying_idle_predecessor'))
        function = FunctionType(handoff.safe_handoff.__code__,
            dict(handoff.safe_handoff.__globals__, handoff_phase_allowed=idle_allowed))
        route = dict(previous_owner=config['previous_owner'], previous_state_path=config['previous_state_path'],
            runtime_configuration=args.config)
        function('original', route, base, update, 10)
        verify(config, verify_frozen_data=False)
        handoff.write(base / 'optimized_original_controller_dispatch.json', dict(
            original_runtime=config['original_runtime'], original_queue=config['original_queue'],
            original_queue_sha256=config['original_queue_sha256'], original_jobs=139,
            numerical_source_and_budget_fields_unchanged=True, raw_data_rewritten=False,
            native_training_children_stopped=False, hash_algorithm_and_complete_byte_coverage_unchanged=True,
            maximum_hash_read_block_bytes=1024*1024, resource_formula_and_limits_unchanged=True,
            original_controller_code_identical=True, additional_training_seed_slots=0,
            all_original_experiments_complete=False))
        sys.argv = ['bounded_original_crossad_controller', '--runtime-config', config['original_runtime'], '--controller', 'crossad']
        bound_main()()
    finally:
        owner.close()


if __name__ == '__main__':
    main()
