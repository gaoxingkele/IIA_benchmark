"""Check actual native state certificates without initializing numerical backends."""
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
RC=ROOT/'projects/flow_matching_research/results/2026-10-13/rcgan_seed1105_full338_recovery_v1/checkpoint_state/checkpoint_resume_state_audit.json'
CHAIN=ROOT/'projects/flow_matching_research/results/2026-10-13/rcgan_seed1105_full338_recovery_v1/random_chain/random_order_chain_audit.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):d.update(chunk)
    return d.hexdigest()


def test_actual_full338_model_state_gate_and_its_guard_succeeded():
    proof=read(RC);config=read(ROOT/proof['configuration'])
    receipt=read(ROOT/config['state_root']/'receipt.json')
    assert proof['passed'] and proof['configuration_sha256']==sha(ROOT/proof['configuration'])
    assert receipt['exit_code']==0 and receipt['checkpoint_resume_state_passed'] and not receipt['resource_guard_aborted']
    assert proof['actual_full_GAN_parameters']==96929 and proof['native_checkpoint_variable_count']==22
    # Legacy key labels remain frozen for the live continuation binding.
    # Actual record counts, not those labels, prove the338/331 extent below.
    assert proof['all238_full_model_and_optimizer_checkpoints_restored_bitwise']
    assert proof['all231_persisted_native_trace_losses_reproduced_bitwise_float32']
    assert not proof['full500_epoch_result_verified'] and not proof['all_original_experiments_complete']


def test_all338_states_and_actual331_trace_rows_preserve_seven_missing_times():
    proof=read(RC);records=proof['epochs']
    assert [r['epoch'] for r in records]==list(range(338))
    assert all(r['full_model_and_optimizer_restore_bitwise_verified'] for r in records)
    assert [r['epoch'] for r in records if r['original_persisted_trace_loss_bitwise_verified']]==list(range(331))
    assert [r['epoch'] for r in records if r['missing_original_elapsed_time_preserved']]==list(range(331,338))
    assert all(r['original_elapsed_time'] is None for r in records[331:])
    directory=Path(proof['job']['output_directory'])/'experiments/parameters'
    available={r['epoch'] for r in records if (directory/(proof['job']['id']+'_'+str(r['epoch'])+'.npy')).is_file()}
    # Author dumps are periodic/best checkpoints; full native state files
    # above still cover every one of the338 epochs without exception.
    assert 0 in available
    assert {r['epoch'] for r in records if r['original_parameter_dump_bitwise_verified']}==available
    assert all(math.isfinite(r[k]) for r in records for k in ['D_loss','G_loss'])
    assert all(r['loss_provenance']=='native_checkpoint_full_last_batch_recalculation' for r in records)


def test_every_state_matches_verified_rng_chain_full_samples_and_next_step_without_training_credit():
    proof=read(RC);chain=read(CHAIN)
    assert proof['random_chain_proof_sha256']==sha(CHAIN)
    names={'checkpoint','model.data-00000-of-00001','model.index','model.meta','numpy_random_state.npz','python_random_state.json'}
    for state,rng in zip(proof['epochs'],chain['verified_epoch_records']):
        files={Path(p['path']).name:p for p in state['checkpoint_artifacts']}
        assert set(files)==names and all(len(p['sha256'])==64 for p in files.values())
        assert files['numpy_random_state.npz']['sha256']==rng['numpy_state_sha256']
        assert files['python_random_state.json']['sha256']==rng['python_state_sha256']
    assert proof['all4984_latest_generator_samples_replayed_bitwise'] and proof['latest_full_generator_forward_calls']==178
    assert proof['original_next_D_G_update_repeated_bitwise_in_all_model_and_optimizer_variables']
    assert proof['audit_only_optimizer_transition_updates']==dict(D=2,G=2)
    assert proof['audit_transition_weights_not_saved_for_training']
    assert not proof['benchmark_performance'] and proof['additional_training_seed_slots']==0
    assert proof['original_scope']==dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)


@pytest.mark.parametrize('epochs,config_path',[
    (500,'configs/runtime/fm_full500_image_checkpoint_CPU_audit.v2.json'),
    (813,'configs/runtime/fm_full813_image_checkpoint_CPU_audit.v1.json')])
def test_actual_full_image_CPU_state_and_guard_cover_every_epoch_without_GPU_certificate(epochs,config_path):
    config=read(ROOT/config_path)
    report_path=ROOT/config['output_root']/('full%d_native_model_optimizer_CPU_rng_restore.json'%epochs)
    proof=read(report_path);guard=read(ROOT/config['state_root']/'receipt.json')
    assert proof['passed'] and guard['passed'] and guard['exit_code']==0 and not guard['resource_guard_aborted']
    assert guard['report_sha256']==sha(report_path) and proof['configuration_sha256']==sha(ROOT/config_path)
    assert proof['checkpoint_receipt']==config['checkpoint_receipt']
    assert proof['full_native_parameters']==45289539 and proof['completed_original_epochs']==epochs
    assert proof['complete_epoch_receipts_verified']==epochs
    images=proof['canonical_job']['expected_training_images_per_epoch']
    assert proof['verified_original_optimizer_updates']==epochs*((images+31)//32)
    assert proof['required_original_epochs']==2000 and proof['remaining_original_epochs']==2000-epochs
    assert proof['all_model_and_StableAdamW_variables_bitwise_restored'] and proof['all_model_and_optimizer_tensors_finite']
    assert proof['Python_NumPy_native_CPU_and_order_RNG_states_bitwise_restored']
    assert proof['saved_original_CUDA_generator_states_retained']==1 and not proof['CUDA_initialized']
    assert not proof['CUDA_RNG_restoration_verified'] and proof['full_native_GPU_restore_and_training_continuation_still_required']
    assert proof['new_optimizer_updates_executed']==0 and proof['new_benchmark_results']==0
    assert proof['original_full25_jobs_retained'] and not proof['all_original_experiments_complete']
