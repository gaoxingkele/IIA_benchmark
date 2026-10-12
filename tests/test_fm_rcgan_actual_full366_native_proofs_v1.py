import math

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha


CHAIN = 'configs/reproducibility/fm_rcgan_seed1104_full366_random_order_chain.2026-10-12_v3.json'
STATE = 'configs/reproducibility/fm_rcgan_seed1104_full366_actual358_trace_resume_state.2026-10-12_v4.json'


def test_actual_full366_native_chain_covers_every_original_rng_draw_and_sample():
    config = read(ROOT / CHAIN)
    proof = read(ROOT / config['output_directory'] / 'random_order_chain_audit.json')
    terminal = read(ROOT / config['state_root'] / 'receipt.json')
    assert proof['passed'] and terminal['random_order_chain_passed']
    assert terminal['exit_code'] == 0 and not terminal['resource_guard_aborted']
    assert proof['original_completed_epochs'] == 366 and proof['required_original_epochs'] == 500
    assert proof['verified_original_numpy_version'] == '1.18.5'
    assert proof['protocol_draw_counts'] == dict(D=469944, G=469944, loss=366)
    assert [e['epoch'] for e in proof['verified_epoch_records']] == list(range(366))
    assert all(e['full4984_noise_and_real_pixels_bitwise_verified']
               and e['post_shuffle_numpy_and_python_states_bitwise_verified'] for e in proof['verified_epoch_records'])
    assert sha(proof['derived_order_artifact']['path']) == proof['derived_order_artifact']['sha256']
    assert proof['tensorflow_model_updates_executed_by_this_audit'] == 0
    assert not proof['full500_epoch_result_verified'] and not proof['benchmark_performance']


def test_actual_every_full_checkpoint_and_persisted_loss_passed_native_bitwise_gate():
    config = read(ROOT / STATE)
    proof = read(ROOT / config['output_directory'] / 'checkpoint_resume_state_audit.json')
    terminal = read(ROOT / config['state_root'] / 'receipt.json')
    assert proof['passed'] and terminal['checkpoint_resume_state_passed']
    assert terminal['exit_code'] == 0 and not terminal['resource_guard_aborted']
    assert proof['actual_full_GAN_parameters'] == 96929
    assert proof['all366_full_model_and_optimizer_checkpoints_restored_bitwise']
    assert proof['all358_persisted_native_trace_losses_reproduced_bitwise_float32']
    assert [e['epoch'] for e in proof['epochs']] == list(range(366))
    assert all(e['full_model_and_optimizer_restore_bitwise_verified'] for e in proof['epochs'])
    assert all(math.isfinite(e['D_loss']) and math.isfinite(e['G_loss']) for e in proof['epochs'])
    assert sum(e['original_persisted_trace_loss_bitwise_verified'] for e in proof['epochs']) == 358
    assert [e['epoch'] for e in proof['epochs'] if e['original_elapsed_time'] is None] == [81,82,83,361,362,363,364,365]
    assert proof['eight_missing_original_elapsed_times_preserved']


def test_actual_full_generator_and_next_update_replay_is_not_added_training():
    config = read(ROOT / STATE)
    proof = read(ROOT / config['output_directory'] / 'checkpoint_resume_state_audit.json')
    assert proof['all4984_latest_generator_samples_replayed_bitwise']
    assert proof['latest_full_generator_forward_calls'] == 178
    assert proof['original_next_D_G_update_repeated_bitwise_in_all_model_and_optimizer_variables']
    assert proof['audit_only_optimizer_transition_updates'] == dict(D=2, G=2)
    assert proof['audit_transition_weights_not_saved_for_training']
    assert proof['additional_training_seed_slots'] == 0
    assert not proof['full500_epoch_result_verified'] and not proof['benchmark_performance']
    assert not proof['author_equivalence_certified'] and not proof['all_original_experiments_complete']
