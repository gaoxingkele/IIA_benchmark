"""Register the exact existing retry's native identity fix without stopping work."""
from datetime import datetime, timezone
import json
import psutil

from scripts.flow_matching.run_rcgan_seed1103_registered_retry_dispatch_v2 import ROOT, read, sha, verify, actual_native_children
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json


def main():
    path = 'configs/reproducibility/fm_rcgan_seed1103_preserved_full262_retry.2026-10-12_v1.json'
    native = read(ROOT/path)
    process = psutil.Process(38832)
    command = process.cmdline()
    if (process.create_time() != 1791740868.1987271 or path not in command
            or 'scripts.flow_matching.run_preserved_native_gan_retry_v1' not in command
            or actual_native_children(process)):
        raise ValueError('Only the actual waiting lightweight seed1103 retry owner may be registered for handoff')
    state = read(ROOT/native['state_root']/'status.json')
    if (state['pid'] != process.pid or state['create_time'] != process.create_time()
            or state['active_pid'] is not None or state['state'] != 'waiting_for_memory'
            or (ROOT/native['recovery_job']['output_directory']).exists()):
        raise ValueError('No child or actual retry output may have started')
    config = dict(native_configuration=path, native_configuration_sha256=sha(ROOT/path),
        state_root='experiments/runs/fm_rcgan_seed1103_registered_retry_dispatch_v2',
        settings=native['settings'], resource_lock=native['resource_lock'], original_scope=native['original_scope'],
        expected_previous_waiting_owner=dict(pid=process.pid, create_time=process.create_time(), command=command))
    sources = {path,
        'scripts/flow_matching/rcgan_seed1103_registered_retry_runtime_v2.py',
        'scripts/flow_matching/run_rcgan_seed1103_registered_retry_dispatch_v2.py',
        'scripts/flow_matching/register_rcgan_seed1103_registered_retry_dispatch_v2.py',
        'tests/test_fm_rcgan_seed1103_registered_retry_dispatch_v2.py'}
    sources.update(pin['path'] for pin in native['source_receipts'])
    config['source_receipts'] = [dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)]
    verify(config)
    config_path = 'configs/runtime/fm_rcgan_seed1103_registered_retry_dispatch.v2.json'
    exclusive_json(ROOT/config_path, config)
    evidence = ROOT/'projects/flow_matching_research/results/2026-10-12/rcgan_seed1103_registered_retry_dispatch_v2'
    exclusive_json(evidence/'registration.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=config_path, configuration_sha256=sha(ROOT/config_path), original_scope=native['original_scope'],
        previous_actual_waiting_owner=config['expected_previous_waiting_owner'],
        exact_existing_retry_id=native['recovery_job']['id'], full_required_original_jobs=5,
        original_epochs_per_job=500, verified_prior_epochs=238, remaining_epochs=262,
        original_checkpoints_and_RNG_resume_proof_preserved=True,
        original_native_train_evaluator_and_source_loop_unchanged=True,
        original40GiB_admission_and_all_emergency_guards_unchanged=True,
        full_training_child_started=False, additional_training_seed_slots=0,
        all_original_experiments_complete=False))
    print(json.dumps(dict(configuration=config_path, exact_registered_retry_id=native['recovery_job']['id'],
        original_jobs=5, full_remaining_epochs=262, old_owner_stopped=False)))


if __name__=='__main__':main()
