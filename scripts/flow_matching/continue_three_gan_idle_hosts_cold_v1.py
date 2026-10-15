"""Continue exact TimeGAN45/energy and GTGAN4 hosts with cached telemetry.

Only completed metadata handoff callbacks are replaced, after a fresh verified
idle-owner handoff. Native entry commands, computations and guards stay frozen.
"""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as common
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha

MODULE = 'scripts.flow_matching.continue_three_gan_idle_hosts_cold_v1'
CONFIG = 'configs/runtime/fm_three_gan_idle_hosts_cold.v1.json'
TEST = 'tests/test_fm_three_gan_idle_hosts_cold_v1.py'
BASE = 'experiments/runs/fm_three_gan_idle_hosts_cold_v1'
PYTHON = 'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
ROLES = {
    'timegan_energy': dict(pid=22636, create_time=1791749831.9943528,
        entry='scripts.flow_matching.run_timegan_corrected_observer_dispatch_v2',
        configuration='configs/runtime/fm_timegan_corrected_observer_dispatch.v2.json',
        arguments=['--config', 'configs/runtime/fm_timegan_corrected_observer_dispatch.v2.json', '--role', 'energy'],
        previous_state_root='experiments/runs/fm_timegan_corrected_observer_energy_v2', native_role='energy'),
    'timegan_original45': dict(pid=13636, create_time=1791874713.935619,
        entry='scripts.flow_matching.run_timegan_corrected_observer_dispatch_v4',
        configuration='configs/runtime/fm_timegan_corrected_observer_dispatch.v4.json',
        arguments=['--config', 'configs/runtime/fm_timegan_corrected_observer_dispatch.v4.json', '--role', 'original'],
        previous_state_root='experiments/runs/fm_timegan_corrected_observer_original_v4', native_role='original'),
    'gtgan_stock4': dict(pid=38352, create_time=1791751485.0781329,
        entry='scripts.flow_matching.run_gtgan_stock_observation_dispatch_v2',
        configuration='configs/runtime/fm_gtgan_stock_observation_dispatch.v2.json',
        arguments=['--runtime-config', 'configs/runtime/fm_gtgan_stock_observation_dispatch.v2.json', '--controller'],
        previous_state_root='experiments/runs/fm_gtgan_stock_observation_dispatch_v2', native_role=None),
}


def original(role):
    module = importlib.import_module(role['entry'])
    config = read(ROOT / role['configuration'])
    queue = module.verify(config) if role['native_role'] is None else module.verify_runtime(config)
    return module, config, queue


def registration():
    roles = {}
    pins = {MODULE.replace('.', '/')+'.py', TEST,
            'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
            'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
            'scripts/flow_matching/cached_resource_snapshot_v1.py'}
    for name, spec in ROLES.items():
        role = deepcopy(spec)
        module, config, queue = original(role)
        base = ROOT / role['previous_state_root']
        proof = read(base / 'handoff.json')
        identity = read(base / 'handoff_identity.json')
        assert proof['waiting_controller_only_stopped'] and not proof['original_native_child_stopped']
        assert identity['no_native_child_present'] and identity['only_controller_will_be_stopped']
        assert identity['previous_owner'] == proof['previous_owner']
        route = config['previous_owner_route'] if role['native_role'] is None else config['roles'][role['native_role']]
        assert route['previous_owner'] == proof['previous_owner']
        role['previous_owner'] = dict(pid=role['pid'], create_time=role['create_time'],
            command=[PYTHON, '-X', 'utf8', '-u', '-m', role['entry'], *role['arguments']])
        role['previous_state_path'] = role['previous_state_root']+'/status.json'
        role['state_root'] = BASE+'/'+name
        role['original_queue'] = config['original_queue']
        role['required_original_slots'] = len(queue['jobs'])
        role['settings'] = config['settings'] if role['native_role'] is None else route['settings']
        role['resource_lock'] = config['resource_lock']
        role['cpu_threads'] = config['cpu_threads']
        if role['native_role'] is None:
            assert len(queue['jobs']) == 4
            role['required_periodic_generations'] = 29
            role['required_trained_evaluators'] = config['required_trained_evaluator_receipts']
            role['required_native_network_checkpoints'] = config['required_native_network_checkpoints']
            role['pending_output_directory'] = None
        else:
            assert len(queue['jobs']) == 45
            from scripts.flow_matching.timegan_native_common_v1 import full_budget
            role['full_native_budget'] = full_budget(read(ROOT / queue['model_config']))
            if name == 'timegan_energy':
                role['pending_output_directory'] = route['output_directory']
            else:
                from scripts.flow_matching.run_timegan_corrected_observer_dispatch_v4 import execution_jobs
                jobs = execution_jobs(config, queue['jobs'])
                role['pending_output_directory'] = next(j['output_directory'] for j in jobs
                    if j['id'] == config['zero_checkpoint_retry']['canonical_job']['id'])
                role['stronger_admission_commit_bytes'] = config['retry_admission_commit_headroom_bytes']
        role['prior_metadata'] = [dict(path=(base/n).relative_to(ROOT).as_posix(), sha256=sha(base/n))
                                  for n in ['handoff.json', 'handoff_identity.json']]
        roles[name] = role
        pins.update([role['entry'].replace('.', '/')+'.py', role['configuration'], role['original_queue']])
        pins.update(p['path'] for p in role['prior_metadata'])
    return dict(roles=roles, state_root=BASE, original_scope=common.SCOPE,
        source_receipts=[dict(path=p, sha256=sha(ROOT/p)) for p in sorted(pins)],
        additional_training_seed_slots=0, all_original_experiments_complete=False,
        boundary='Full original45 TimeGAN jobs and existing energy recovery, GTGAN4 full commands/29 generations/'
                 '580 trained evaluators/116 native checkpoints retained. Only completed metadata handoff callback '
                 'reused after fresh PID/time/command/descendant/lock/output verification. Exact original native '
                 'commands, datasets, checkpoints, random states, all optimizer/evaluator budgets and gates preserved. '
                 'Cached equivalent Windows resource telemetry only; no benchmark-completion claim.')


def verify(config):
    if config != registration():
        raise ValueError('Full original GAN scope, immutable proofs and resource policies must remain exact')


def idle_allowed(role, state, expected, descendants):
    module, config, queue = original(role)
    if role['native_role'] is None:
        return (state.get('pid') == expected['pid'] and state.get('create_time') == expected['create_time']
                and module.idle_parent_phase(state, descendants, module.all_attempts_unstarted(queue)))
    return (common.idle_allowed(state, expected, descendants)
            and not (ROOT / role['pending_output_directory']).exists())


def already_handed_off(role, base):
    """Check old metadata bytes and the new real handoff before reusing its callback."""
    for pin in role['prior_metadata']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Preserved previous metadata proof changed')
    proof = read(base / 'handoff.json')
    if (proof['previous_owner'] != role['previous_owner']
            or proof['original_native_child_stopped'] or not proof['waiting_controller_only_stopped']):
        raise ValueError('New exact idle-owner handoff is required')
    import psutil
    if psutil.pid_exists(role['previous_owner']['pid']):
        raise ValueError('Previous idle owner still exists; never skip its handoff')


def continued_main(role, base):
    module, config, queue = original(role)

    def metadata_only_callback(*args):
        if role['native_role'] is None:
            if len(args) != 4 or args[0] != config or args[1] != queue or args[2] != ROOT/role['previous_state_root']:
                raise ValueError('Only the existing exact GTGAN metadata handoff can be reused')
        else:
            if (len(args) != 5 or args[0] != role['native_role'] or args[1] != config['roles'][role['native_role']]
                    or args[2] != ROOT/role['previous_state_root'] or args[4] != config['poll_seconds']):
                raise ValueError('Only the existing exact TimeGAN metadata handoff can be reused')
        already_handed_off(role, base)

    if role['entry'].endswith('_v4'):
        namespace = module.runtime_namespace()
        namespace['safe_handoff'] = metadata_only_callback
        return namespace['main']
    name = 'handoff' if role['native_role'] is None else 'safe_handoff'
    return FunctionType(module.main.__code__, dict(module.__dict__, **{name: metadata_only_callback}),
                        argdefs=module.main.__defaults__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        common.exclusive(ROOT/CONFIG, registration())
        print(json.dumps(dict(registered=True, roles=list(ROLES))))
        return
    if args.role is None:
        parser.error('--role is required')
    config = read(ROOT/CONFIG)
    verify(config)
    role = config['roles'][args.role]
    common.install_cached_loader(lambda snapshot: None)
    base = ROOT/role['state_root']
    continuation = continued_main(role, base)
    if args.probe_only:
        import psutil
        print(json.dumps(dict(role=args.role, full_original_slots=role['required_original_slots'],
            original_arguments=role['arguments'], private_bytes=psutil.Process().memory_info().private,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            tensorflow_imported='tensorflow' in sys.modules, numerical_work_executed=False)))
        return
    base.mkdir(parents=True, exist_ok=True)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    owner = resource_api()['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'successor_entry.json').exists():
        if owner is not None:
            owner.close()
        raise RuntimeError('Existing continuation is preserved')
    try:
        namespace = dict(common.__dict__, idle_allowed=lambda state, expected, descendants:
                         idle_allowed(role, state, expected, descendants))
        FunctionType(common.handoff.__code__, namespace, argdefs=common.handoff.__defaults__)(role, base)
        verify(config)
        common.exclusive(base/'successor_entry.json', dict(entry=role['entry'], arguments=role['arguments'],
                         original_numerical_worker_unchanged=True, additional_training_seed_slots=0))
        previous = sys.argv
        try:
            sys.argv = [role['entry'], *role['arguments']]
            continuation()
        finally:
            sys.argv = previous
    finally:
        owner.close()


if __name__ == '__main__':
    main()
