"""Observe one full native generation comparison without changing its gate.

TimeGAN: Yoon et al., NeurIPS 2019. This is a forensic observation, not a
benchmark result or a replacement for all45/all20 complete native replays.
Python 3.7 compatible; no numerical framework is imported by the parent.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType

from scripts.flow_matching import run_timegan_full45_completion_replay_v5 as replay

BASE = replay.BASE
DESCRIPTION = 'complete trained normalized generation'


class ObservationComplete(Exception):
    pass


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify(config):
    for pin in config['pins']:
        if BASE.sha(BASE.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen diagnostic evidence changed: ' + pin['path'])
    original = BASE.read(BASE.ROOT / config['full_replay_configuration'])
    queue, routes = replay.verify(original)
    route = next(r for r in routes if r['canonical_job']['id'] == config['canonical_job_id'])
    binding = BASE.read(BASE.ROOT / config['failed_binding'])
    if (binding['route'] != route or binding['runtime_configuration_sha256'] != BASE.sha(
            BASE.ROOT / config['full_replay_configuration'])
            or config['settings'] != original['settings']
            or config['resource_lock'] != original['resource_lock']
            or config['python'] != original['python']
            or config['cpu_threads'] != original['cpu_threads']
            or config['original_scope'] != original['original_scope']
            or config['maximum_generation_observations'] != 1):
        raise ValueError('Exact failed binding and all original guards required')
    if ('ValueError: Full finite bitwise equality required: ' + DESCRIPTION) not in (
            BASE.ROOT / config['failed_stderr']).read_text(encoding='utf-8'):
        raise ValueError('Only the preserved actual generation mismatch may be diagnosed')
    predecessor = BASE.read(BASE.ROOT / config['successful_predecessor_receipt'])
    proof = BASE.read(BASE.ROOT / config['successful_predecessor_report'])
    if (predecessor['full_saved_predecessor_native_replay_passed'] is not True
            or predecessor['report_sha256'] != BASE.sha(BASE.ROOT / config['successful_predecessor_report'])
            or proof['canonical_job']['id'] != config['canonical_job_id']
            or proof['complete_normalized_and_renormalized_generation_bitwise_verified'] is not True):
        raise ValueError('Actual earlier successful native generation evidence required')
    return original, route


def describe_arrays(actual, expected, np):
    shape_equal = actual.shape == expected.shape
    dtype_equal = actual.dtype == expected.dtype
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    row = dict(actual_shape=list(actual.shape), expected_shape=list(expected.shape),
        actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
        all_finite=finite, same_shape=shape_equal, same_dtype=dtype_equal,
        array_equal=bool(shape_equal and dtype_equal and np.array_equal(actual, expected)))
    if shape_equal and finite:
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        row.update(total_coordinates=int(actual.size), unequal_coordinates=int(np.count_nonzero(actual != expected)),
            maximum_absolute_difference=float(difference.max()), mean_absolute_difference=float(difference.mean()),
            first_unequal_indices=np.argwhere(actual != expected)[:10].tolist())
        if dtype_equal and actual.dtype == np.dtype('float32') and (actual >= 0).all() and (expected >= 0).all():
            # TimeGAN's normalized sigmoid outputs are nonnegative float32.
            ulp = np.abs(actual.view(np.uint32).astype(np.int64) - expected.view(np.uint32).astype(np.int64))
            row['maximum_float32_ULP_distance'] = int(ulp.max())
    return row


def observed_worker(original_worker, capture):
    namespace = dict(original_worker.__globals__)
    strict = namespace['bitwise_array']

    def observe(actual, expected, np, description):
        if description != DESCRIPTION:
            return strict(actual, expected, np, description)
        evidence = describe_arrays(actual, expected, np)
        try:
            strict(actual, expected, np, description)
        except ValueError as error:
            evidence.update(original_bitwise_gate_passed=False, original_gate_error=str(error))
        else:
            evidence['original_bitwise_gate_passed'] = True
        capture(actual, expected, np, evidence)
        # Always stop after the single full comparison, including when equal.
        # No full replay completion file or new numerical training is produced.
        raise ObservationComplete()

    namespace['bitwise_array'] = observe
    function = FunctionType(original_worker.__code__, namespace, original_worker.__name__,
        original_worker.__defaults__, original_worker.__closure__)
    function.__kwdefaults__ = original_worker.__kwdefaults__
    return function


def worker(config):
    original, route = verify(config)
    destination = BASE.ROOT / config['output_directory']
    destination.mkdir(parents=True, exist_ok=False)
    isolated = copy.deepcopy(original)
    isolated['output_root'] = config['isolated_native_output_root']
    if {k for k in original if original[k] != isolated[k]} != {'output_root'}:
        raise ValueError('Only the isolated diagnostic filesystem destination may change')
    function = replay.runtime_namespace()['worker']

    def capture(actual, expected, np, evidence):
        arrays = destination / 'full_actual_and_expected_normalized_generation.npz'
        with arrays.open('xb') as stream:
            np.savez_compressed(stream, actual=actual, expected=expected)
        exclusive_json(destination / 'generation_difference.json', dict(
            captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job=route['canonical_job'],
            full_original_data_shape=route['full_original_data_shape'], comparison=evidence,
            full_arrays_sha256=BASE.sha(arrays), original_scope=config['original_scope'],
            single_full_generation_observation=True, original_worker_code_unchanged=True,
            original_guard_settings=config['settings'], numerical_optimizer_updates_executed=0,
            independent_full_checkpoint_replay_completed=False, benchmark_performance=False,
            all_original_experiments_complete=False, diagnostic_only=True))

    try:
        observed_worker(function, capture)(isolated, route, config['failed_binding'])
    except ObservationComplete:
        return
    raise RuntimeError('Expected one full generation comparison was not observed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = BASE.read(BASE.ROOT / args.config)
    verify(config)
    if config['runtime_configuration'] != args.config:
        raise ValueError('Exact immutable diagnostic configuration required')
    if args.probe_only:
        print(json.dumps(dict(registration_verified=True, original_jobs=45,
            numerical_training_executed=False, diagnostic_executed=False)))
        return
    if args.worker:
        worker(config)
        return
    import psutil
    from scripts.flow_matching.run_timegan_checkpointed_stage_recovery_v4 import cached_resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = cached_resource_api()
    state_root = BASE.ROOT / config['state_root']
    state_root.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](state_root / 'controller.lock')
    if owner is None or (state_root / 'receipt.json').exists():
        raise RuntimeError('Existing diagnostic owner or terminal attempt must be preserved')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None,
        state='starting', configuration_sha256=BASE.sha(BASE.ROOT / args.config), diagnostic_only=True)

    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        BASE.write(state_root / 'status.json', state)

    try:
        with api['resource_slot'](BASE.ROOT / config['resource_lock'], config['settings'], update):
            argv = ['native_generation_diagnostic', '--config', args.config, '--worker']
            bootstrap = ('import sys,runpy;sys.path.insert(0,' + repr(str(BASE.ROOT))
                + ');sys.argv=' + repr(argv) + ";runpy.run_module('scripts.flow_matching.diagnose_timegan_generation_bitwise_v1',run_name='__main__')")
            command = [config['python'], '-X', 'utf8', '-u', '-c', bootstrap]
            with (state_root / 'worker.stdout.log').open('xb') as out, (state_root / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=BASE.ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                update(dict(active_pid=child.pid, active_create_time=psutil.Process(child.pid).create_time(), actual_command=command))
                code, usage = capped_wait(child, config, api, update)
        proof = BASE.ROOT / config['output_directory'] / 'generation_difference.json'
        captured = code == 0 and not usage['resource_guard_aborted'] and proof.exists()
        BASE.write(state_root / 'receipt.json', dict(exit_code=code, diagnostic_captured=captured,
            independent_full_checkpoint_replay_completed=False, **usage))
        update(dict(state='diagnostic_captured' if captured else 'failed_diagnostic_preserved', active_pid=None))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
