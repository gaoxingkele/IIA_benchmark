"""Unchanged full native replay for the saved15000 Sine continuation route.

All45 canonical routes remain registered; this supplementary controller handles
one new pending result without stopping or repeating the existing full45 queue.
TimeGAN: Yoon, Jarrett, van der Schaar, NeurIPS2019, jsyoon0823/TimeGAN.
"""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType

from scripts.flow_matching import run_timegan_full45_completion_replay_v5 as original
from scripts.flow_matching import timegan_saved_joint_continuation_v1 as continuation

BASE = original.BASE
ROOT, read, write, sha = BASE.ROOT, BASE.read, BASE.write, BASE.sha
PREVIOUS = 'configs/runtime/fm_timegan_full45_completion_replay.v6.json'
RECOVERY = 'configs/reproducibility/fm_timegan_sine_saved_joint15000_continuation.2026-10-14_v1.json'
CANONICAL = 'timegan_native__gru__sine__seed1103'


def expected_routes(previous, recovery):
    routes = copy.deepcopy(previous['routes'])
    route = next(r for r in routes if r['canonical_job']['id'] == CANONICAL)
    if route['canonical_job'] != recovery['canonical_job']:
        raise ValueError('Same original Sine training seed required')
    route.update(kind='saved_joint_continuation', recovery_configuration=RECOVERY,
        execution_job=dict(route['canonical_job'], output_directory=recovery['output_directory']),
        result_path=(Path(recovery['output_directory']) / 'result.json').as_posix(),
        native_completion_receipts=[dict(path=recovery['state_root'] + '/receipt.json',
            successful_field='full_native_recovery_result_verified')])
    return routes


def check_routes(queue, config):
    from scripts.flow_matching.run_timegan_full45_completion_replay_v2 import check_routes as inherited
    normalized = copy.deepcopy(config)
    for route in normalized['routes']:
        if route['kind'] == 'saved_joint_continuation':
            recovery = read(ROOT / route['recovery_configuration'])
            continuation.registration(recovery)
            route['kind'] = 'stage_resume'
    inherited(queue, normalized)
    return config['routes']


def verify(config):
    previous, recovery = read(ROOT / PREVIOUS), read(ROOT / RECOVERY)
    if (config['previous_full45_configuration'] != PREVIOUS
            or config['selected_canonical_ids'] != [CANONICAL]
            or config['routes'] != expected_routes(previous, recovery)):
        raise ValueError('All45 routes retained; only the exact pending Sine route may change')
    for field in ('original_queue', 'settings', 'resource_lock', 'python', 'cpu_threads',
            'required_checkpoints_each', 'required_evaluators_each', 'equality_gate',
            'protocol_iterations_each', 'required_full_generated_windows',
            'required_full_predictive_test_windows', 'required_full_discriminative_test_windows', 'original_scope'):
        if config[field] != previous[field]:
            raise ValueError('Original full replay scope/budget/guard changed: ' + field)
    # The original full verifier checks every pin, author source, protocol AST,
    # full45 identity, complete dimensions, optimizer budget and aggregate count.
    namespace = dict(BASE.verify.__globals__, check_routes=check_routes)
    function = FunctionType(BASE.verify.__code__, namespace)
    return function(config)


def completed_native_result(route, queue):
    if route['kind'] == 'saved_joint_continuation':
        config = read(ROOT / route['recovery_configuration'])
        metadata = read(ROOT / route['result_path'])
        return continuation.verify_result(config, metadata['saved_joint_native_proof_sha256'])
    return original.completed_native_result(route, queue)


def native_namespace():
    # Rebind the exact original numerical worker and helper code, without AST
    # changes to its23 restores,20 evaluators,70000 protocol loops or gates.
    namespace = dict(BASE.__dict__)
    for name, value in BASE.__dict__.items():
        if isinstance(value, FunctionType) and value.__module__ == BASE.__name__:
            replacement = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            replacement.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = replacement
    namespace.update(verify=verify, check_routes=check_routes, completed_native_result=completed_native_result)
    return namespace


def collect_full45_evidence(config):
    """Combine real guarded certificates, without rerunning completed models."""
    previous = read(ROOT / PREVIOUS)
    verified = {}
    for route in config['routes']:
        identity = route['canonical_job']['id']
        selected = identity == CANONICAL
        base = ROOT / (config['state_root'] if selected else previous['state_root'])
        receipt_path = base / ('receipt.json' if selected else 'receipts/' + identity + '.json')
        binding_path = base / ('binding.json' if selected else 'bindings/' + identity + '.json')
        report_path = ROOT / (config['output_root'] if selected else previous['output_root']) / identity / 'complete_full_native_replay.json'
        if not all(p.exists() for p in [receipt_path, binding_path, report_path]):
            continue
        receipt = read(receipt_path)
        if (receipt.get('exit_code') != 0 or receipt.get('resource_guard_aborted') is not False
                or receipt.get('complete_full_native_replay_passed') is not True):
            continue
        binding = read(binding_path)
        source_config = config if selected else previous
        if (binding['route'] != route
                or binding['runtime_configuration_sha256'] != sha(ROOT / source_config['runtime_configuration'])
                or binding['result_sha256'] != sha(ROOT / route['result_path'])
                or binding['native_receipt_sha256'] != sha(ROOT / binding['native_completion_receipt']['path'])):
            raise ValueError('Real full native certificate binding changed: ' + identity)
        binding['binding_sha256'] = sha(binding_path)
        report = read(report_path)
        BASE.check_report(route, binding, report, receipt)
        verified[identity] = dict(canonical_job=route['canonical_job'], execution_job=route['execution_job'],
            report_path=str(report_path), report_sha256=sha(report_path),
            binding_path=str(binding_path), binding_sha256=sha(binding_path),
            guarded_replay_receipt_path=str(receipt_path), guarded_replay_receipt_sha256=sha(receipt_path),
            complete_checkpoint_restores=23, complete_trained_evaluator_replays=20,
            full_source_protocol_iterations=70000, additional_training_seed_slots=0)
    return dict(required_original_models=45, verified_original_ids=sorted(verified),
        pending_original_models=45-len(verified), verified_native_evidence=verified,
        all45_TimeGAN_native_replays_complete=len(verified)==45,
        all_original_experiments_complete=False, author_equivalence_certified=False, strict_TAB_result=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--binding')
    args = parser.parse_args(); config = read(ROOT / args.config)
    if config['runtime_configuration'] != args.config:
        raise ValueError('Frozen exact replay configuration required')
    queue, routes = verify(config)
    route = next(r for r in routes if r['canonical_job']['id'] == CANONICAL)
    if args.probe_only:
        print(dict(original_jobs=len(routes), selected_pending_routes=1,
            required_checkpoints_each=23, required_evaluators_each=20,
            source_protocol_iterations_each=70000, equality_gate='bitwise',
            tensorflow_imported='tensorflow' in sys.modules, numpy_imported='numpy' in sys.modules,
            actual_native_replay_executed=False, all_original_experiments_complete=False))
        return
    if args.worker:
        if not args.binding:
            raise ValueError('Actual guarded full native completion binding required')
        native_namespace()['worker'](config, route, args.binding); return
    import psutil
    from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('This supplementary Sine full replay already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        configuration_sha256=sha(ROOT / args.config), required_original_models=45,
        selected_pending_canonical_id=CANONICAL, original_full45_controller_unchanged=True,
        all_original_experiments_complete=False, independent_training_seed_slots_added=0)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)

    def adopt():
        union = collect_full45_evidence(config)
        write(base / 'verified_full45_union.json', union)
        update({k: v for k, v in union.items() if k != 'verified_native_evidence'})

    try:
        while True:
            adopt()
            entry, receipt = BASE.successful_native_receipt(route)
            if entry is not None and (ROOT / route['result_path']).exists():
                break
            if any((ROOT / e['path']).exists() for e in route['native_completion_receipts']):
                update(dict(state='native_continuation_failed_no_replay_or_repeat'))
                return
            update(dict(state='waiting_actual_full50000_joint_and_all20_native_evaluators', shared_lock_owned=False))
            time.sleep(config['poll_seconds'])
        if sha(ROOT / args.config) != state['configuration_sha256']:
            raise ValueError('Live supplementary replay config changed')
        # Parent reads metadata only; every artifact/numerical/continuation gate
        # executes again inside the original guarded native worker.
        result = read(ROOT / route['result_path'])
        BASE.complete_metadata(route, queue, result, receipt, entry['successful_field'])
        binding = dict(route=route, runtime_configuration_sha256=state['configuration_sha256'],
            result_sha256=sha(ROOT / route['result_path']), native_completion_receipt=entry,
            native_receipt_sha256=sha(ROOT / entry['path']))
        binding_path = base / 'binding.json'; BASE.exclusive(binding_path, binding)
        argv = ['saved_joint_full_replay', '--config', args.config, '--worker', '--binding', str(binding_path)]
        bootstrap = ('import sys,runpy;sys.path.insert(0,' + repr(str(ROOT)) + ');sys.argv=' + repr(argv)
            + ";runpy.run_module('scripts.flow_matching.timegan_saved_joint_full_replay_v1',run_name='__main__')")
        command = [queue['python'], '-X', 'utf8', '-u', '-c', bootstrap]
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            verify(config)
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(state='running_complete_saved_joint_native_result_replay', active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        report_path = ROOT / config['output_root'] / CANONICAL / 'complete_full_native_replay.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and report_path.exists()
        final_receipt = dict(exit_code=code, complete_full_native_replay_passed=passed, **usage)
        if passed:
            binding['binding_sha256'] = sha(binding_path)
            BASE.check_report(route, binding, read(report_path), final_receipt)
        write(base / 'receipt.json', final_receipt)
        update(dict(active_pid=None, state='selected_complete_full_native_replay_verified' if passed else 'failed_replay_preserved_no_automatic_repeat'))
        if passed:
            while True:
                adopt()
                if state['all45_TimeGAN_native_replays_complete']:
                    update(dict(state='all45_complete_native_model_and_evaluator_replays_verified')); break
                update(dict(state='waiting_remaining_original_full45_guarded_native_certificates'))
                time.sleep(config['poll_seconds'])
    finally:
        owner.close()


if __name__ == '__main__':
    main()
