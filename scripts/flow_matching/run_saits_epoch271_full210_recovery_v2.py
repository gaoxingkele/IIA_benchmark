"""Continue the exact saved BRITS epoch and every original210 imputation job.

The original native runner restores model/Adam/controller/Python/NumPy/Torch/
CUDA RNG state at its saved epoch boundary. No source training code changes.
Admission is strengthened after the actual resource abort; all other guards,
validation stopping, model/data configs and native commands remain unchanged.
SAITS/BRITS comparison: https://arxiv.org/abs/2202.08516
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path
import sys

from scripts.flow_matching import run_saits_guarded_original_queue_v1 as original

ROOT = original.ROOT
PREVIOUS = 'configs/runtime/fm_saits_guarded_original_resume.v1.json'
SELECTED = 'saits_paper_brits_electricity_author_missing0.2_seed27'


def verify(runtime):
    previous = original.read(ROOT / PREVIOUS)
    changed = {k for k in set(runtime) | set(previous) if runtime.get(k) != previous.get(k)}
    allowed = {'state_root', 'source_receipts', 'previous_runtime_configuration',
               'preserved_incident', 'saved_epoch_proof', 'saved_checkpoint_archive',
               'selected_recovery_admission_settings', 'dispatch_policy'}
    if not changed <= allowed or runtime['previous_runtime_configuration'] != PREVIOUS:
        raise ValueError('Only distinct host state, saved proof and stronger admission may change')
    if runtime['dispatch_policy'] != 'stable_ascending_original_per_job_admission_all210':
        raise ValueError('All210 independent native seeded processes retain their own admission')
    expected = dict(previous['settings'], minimum_commit_headroom_bytes=40 * 2**30)
    if (runtime['settings'] != previous['settings'] or runtime['selected_recovery_admission_settings'] != expected
            or runtime['state_root'] == previous['state_root']):
        raise ValueError('Original guards unchanged; only selected saved-epoch job gains stronger40GiB admission')
    pins = {p['path']:p['sha256'] for p in runtime['source_receipts']}
    if len(pins) != len(runtime['source_receipts']):
        raise ValueError('No duplicate source pins')
    if any(pins.get(p['path']) != p['sha256'] for p in previous['source_receipts']):
        raise ValueError('Preserve every original numerical source pin')
    queue = original.verify_binding(runtime)
    if len(queue['jobs']) != 210:
        raise ValueError('All210 original full model/data/seed/variant jobs remain required')
    proof = original.read(ROOT / runtime['saved_epoch_proof'])
    if (proof['canonical_id'] != SELECTED or proof['full_saved_epoch_boundary'] != 271
            or proof['saved_optimizer_updates'] != 1897
            or proof['full_native_model_tensors_restored_bitwise'] != 32
            or proof['full_native_optimizer_parameter_slots_restored_bitwise'] != 28
            or proof['every_optimizer_step_equals1897'] is not True
            or proof['Python_NumPy_Torch_CPU_RNG_restored_exactly'] is not True
            or proof['saved_CUDA_RNG_states'] != 1 or proof['numeric_optimizer_updates_executed'] != 0
            or proof['full_training_or_test_complete'] is not False
            or proof['original_max_epochs'] != 10000 or proof['original_early_stop_patience'] != 30):
        raise ValueError('Actual full saved native model/Adam/RNG proof required; no complete-result credit')
    archive = Path(runtime['saved_checkpoint_archive'])
    if (original.digest(archive / 'resume.pt') != proof['checkpoint_sha256']
            or original.digest(archive / 'best.pt') != proof['best_checkpoint_sha256']):
        raise ValueError('Every original interrupted checkpoint byte must stay preserved')
    incident = original.read(ROOT / runtime['preserved_incident'])
    if not incident['resource_guard_aborted'] or incident['original_native_exit_code'] != 1:
        raise ValueError('Actual preserved original resource failure required')
    return queue, proof, incident


def predecessor_absent(incident):
    import psutil
    for old in (incident['previous_owner'], incident['previous_native_child']):
        if old['pid'] is None:
            # The original terminal state clears active_pid but retains the
            # exact command/create_time. Inspect real handles, never infer from
            # that cleared field alone that an old numerical child is gone.
            for candidate in psutil.process_iter(['pid', 'create_time', 'cmdline']):
                if (candidate.info.get('create_time') == old['create_time']
                        or candidate.info.get('cmdline') == old['command']):
                    raise RuntimeError('Previous exact imputation native child still exists')
            continue
        try:
            process = psutil.Process(old['pid'])
            same_time = abs(process.create_time() - old['create_time']) < .001
            same_command = process.cmdline() == old['command']
        except psutil.NoSuchProcess:
            continue
        if same_time or same_command:
            raise RuntimeError('Previous exact imputation controller or native child still exists')


def saved_epoch_still_current(proof):
    if (original.digest(proof['checkpoint_path']) != proof['checkpoint_sha256']
            or original.digest(proof['best_checkpoint_path']) != proof['best_checkpoint_sha256']):
        raise ValueError('Selected original native saved state changed before continuation')


def job_admission(runtime, identity):
    return runtime['selected_recovery_admission_settings'] if identity == SELECTED else runtime['settings']


def scheduler_jobs(queue, runtime):
    if len(queue['jobs']) != 210 or len({j['config'] for j in queue['jobs']}) != 210:
        raise ValueError('Every original210 unique native seed/config slot required')
    return sorted(queue['jobs'], key=lambda job: job_admission(runtime,
        original.read(ROOT / job['config'])['id'])['minimum_commit_headroom_bytes'])


def controller_ast():
    tree = ast.parse(inspect.getsource(original.main)); matches = 0; loops = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and ast.dump(node.func) == ast.dump(ast.parse("api['resource_slot']", mode='eval').body)):
            if ast.dump(node.args[1]) != ast.dump(ast.parse("runtime['settings']", mode='eval').body):
                raise ValueError('Original native admission settings argument required')
            node.args[1] = ast.parse('_job_admission(runtime, identity)', mode='eval').body
            matches += 1
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'job'
                and ast.dump(node.iter) == ast.dump(ast.parse("queue['jobs']", mode='eval').body)):
            node.iter = ast.parse('_scheduler_jobs(queue, runtime)', mode='eval').body
            loops += 1
    if matches != 1 or loops != 1:
        raise ValueError('Only full210 independent job order and selected admission binding may change')
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    runtime = original.read(ROOT / args.runtime_config)
    queue, proof, incident = verify(runtime)
    if args.probe_only:
        print(dict(full_original_jobs=len(queue['jobs']), saved_epochs=271, saved_optimizer_updates=1897,
            original_max_epochs=10000, original_early_stop_patience=30, native_training_source_unchanged=True,
            full_training_or_test_complete=False, torch_imported='torch' in sys.modules,
            numpy_imported='numpy' in sys.modules))
        return
    predecessor_absent(incident)
    saved_epoch_still_current(proof)
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda snapshot: None)
    # All original dispatcher statements and native numerical commands remain;
    # only independent job order and selected host admission bindings differ.
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--runtime-config', args.runtime_config]
        namespace = dict(original.__dict__, _job_admission=job_admission, _scheduler_jobs=scheduler_jobs)
        exec(compile(controller_ast(), __file__, 'exec'), namespace)
        namespace['main']()
    finally:
        sys.argv = argv


if __name__ == '__main__':
    main()
