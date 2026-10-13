"""Cold handoff of idle monitors for all121 original corrected-SMAP CPU slots.

Only host resource_snapshot reuses the existing Windows ctypes definitions.
Every complete queue, model/evaluator child, cap, seed, thread count, evaluation
protocol and corrected-data receipt remains unchanged. The170 registered GPU
controls retain their original separate admission, outside this CPU handoff.
Literature/data citations remain in the unchanged original model/data configs.
"""
import argparse
from types import FunctionType
import sys

from scripts.flow_matching import run_tsad_constant_fit_cpu_controls_v2 as original
from scripts.flow_matching import run_cfm_streaming_idle_handoff_v3 as cold

ROOT = original.ROOT


def cached_api():
    api = original.resource_api()
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    api['resource_snapshot'] = resource_snapshot
    return api


def idle_allowed(role, state, output_exists, descendants):
    return (role == 'original' and state.get('active_pid') is None
            and state.get('state') in {'waiting_for_memory', 'waiting_for_resource_lock'}
            and not state.get('shared_lock_owned')
            and all(cold.handoff.allowed_console_child(p['name'], p['exe']) for p in descendants))


def rebound_main():
    return FunctionType(original.main.__code__, dict(original.__dict__, resource_api=cached_api))


def verify(config, controller):
    for pin in config['source_receipts']:
        if cold.streaming_sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen idle corrected CPU handoff changed: ' + pin['path'])
    registration = cold.original.read(ROOT / config['original_runtime'])
    if (registration['full_registered_corrected_CPU_slots'] != 121
            or registration['full_registered_corrected_GPU_slots_separate_original_gate_required'] != 170
            or sum(binding['jobs'] for binding in registration['controllers']) != 121
            or set(config['roles']) != {binding['original_lane'] for binding in registration['controllers']}
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Every6 original CPU lane and all121 original slots required')
    selected = next(b for b in registration['controllers'] if b['original_lane'] == controller)
    route = config['roles'][controller]
    if route['previous_state_path'] != registration['state_root'] + '/' + controller + '/status.json':
        raise ValueError('Existing source controller state and resource locks must remain')
    bound, namespace, binding, queue, factory = original.bind(registration, controller)
    if binding != selected or len(queue['jobs']) != selected['jobs']:
        raise ValueError('The whole exact original queue must survive')
    return registration, queue


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = cold.original.read(ROOT / args.config)
    registration, queue = verify(config, args.controller)
    if args.probe_only:
        import psutil
        print(dict(controller=args.controller, unchanged_complete_queue_jobs=len(queue['jobs']),
                   all_original_CPU_slots=121, separate_original_GPU_slots=170,
                   original_guard_settings=registration['settings'],
                   original_main_bytecode_unchanged=rebound_main().__code__ is original.main.__code__,
                   numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                   private_bytes=psutil.Process().memory_info().private))
        return
    base = ROOT / config['handoff_state_root'] / args.controller
    base.mkdir(parents=True, exist_ok=True)
    owner = cached_api()['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Corrected CPU idle handoff already owned')
    try:
        function = FunctionType(cold.handoff.safe_handoff.__code__,
                                dict(cold.handoff.__dict__, handoff_phase_allowed=idle_allowed))
        route = dict(config['roles'][args.controller], runtime_configuration=args.config)
        function('original', route, base, lambda state: cold.original.write(base / 'status.json', state), 10)
        verify(config, args.controller)
        sys.argv = ['unchanged_corrected_cpu_controller', '--runtime-config', config['original_runtime'], '--controller', args.controller]
        rebound_main()()
    finally:
        owner.close()


if __name__ == '__main__':
    main()
