"""Read-only provenance for seed1104's 84 + 282 completed native epochs.

The derived evidence view uses same-volume hard links, preserves both failed
attempts, and keeps absent trace rows absent. It is never a training result.
"""
from pathlib import Path

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha


TRACE_EPOCHS = tuple(range(81)) + tuple(range(84, 361))
MISSING_TRACE_EPOCHS = (81, 82, 83, 361, 362, 363, 364, 365)


def verify_pins(items):
    for item in items:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Segmented native evidence changed: ' + item['path'])


def verify_manifest(manifest):
    if (manifest['completed_original_epochs'] != 366
            or manifest['remaining_original_epochs'] != 134
            or manifest['required_original_epochs'] != 500
            or manifest['persisted_trace_epochs'] != list(TRACE_EPOCHS)
            or manifest['missing_trace_epochs'] != list(MISSING_TRACE_EPOCHS)
            or manifest['benchmark_performance']
            or manifest['additional_training_seed_slots'] != 0):
        raise ValueError('Retain the entire original budget and all eight trace gaps')
    verify_pins(manifest['source_receipts'])
    verify_pins(manifest['derived_receipts'])
    verify_pins(read(ROOT / manifest['prior_chain_configuration'])['source_receipts'])
    verify_pins(read(ROOT / manifest['prior_state_configuration'])['source_receipts'])
    progress = read(ROOT / manifest['interrupted_progress'])
    receipt = read(ROOT / manifest['failed_terminal_receipt'])
    if (receipt['exit_code'] == 0 or receipt['full_original500epoch_resume_verified']
            or not receipt['resource_guard_aborted']
            or progress['full_epochs_completed'] != 366
            or progress['required_epochs'] != 500 or progress['full_training_complete']
            or progress['job'] != manifest['interrupted_job']
            or any(progress['optimizer_updates'][k] != 366 * 1284 for k in ('D', 'G'))):
        raise ValueError('Require the authoritative incomplete366-of500 native receipt')
    evaluations = read(ROOT / manifest['completed_evaluators'])
    epochs = read(ROOT / manifest['completed_epochs'])
    if (len(evaluations) != 366 or len(epochs) != 366
            or [e['epoch'] for e in evaluations] != list(range(366))
            or [e['epoch'] for e in epochs] != list(range(366))
            or sum(e['sigma_optimizer_updates'] for e in evaluations) != progress['optimizer_updates']['sigma']):
        raise ValueError('Every cumulative full epoch/evaluator counter must match')
    for evaluation in evaluations:
        if (evaluation['complete_generated_points'] != 4984
                or evaluation['full_fit_points'] != 996 or evaluation['full_test_points'] != 3988
                or not 1 <= evaluation['sigma_optimizer_updates'] <= 2000
                or (evaluation['sigma_optimizer_updates'] < 2000 and evaluation['stopping_delta'] > .001)):
            raise ValueError('The complete source evaluator and stopping rule are required')
    prior = read(ROOT / manifest['prior_checkpoint_proof'])
    restoration = read(ROOT / manifest['first_resume_restoration'])
    if (not prior['passed'] or len(prior['epochs']) != 84
            or not prior['all84_full_model_and_optimizer_checkpoints_restored_bitwise']
            or not prior['all81_persisted_native_trace_losses_reproduced_bitwise_float32']
            or not restoration['every_model_optimizer_variable_bitwise_restored']
            or not restoration['full36000_training_image_order_and_both_random_states_restored']
            or restoration['completed_original_epochs'] != 84
            or restoration['next_original_epoch'] != 84
            or restoration['remaining_full_epochs'] != 416
            or restoration['proof_binding']['checkpoint_state_proof_sha256'] != sha(ROOT / manifest['prior_checkpoint_proof'])
            or evaluations[:84] != read(ROOT / manifest['first_completed_evaluators'])):
        raise ValueError('The actual successful84-epoch restore must bind the resumed segment')
    return evaluations


def verify_registration(config):
    verify_pins(config['source_receipts'])
    manifest = read(ROOT / config['segmented_evidence_manifest'])
    evaluations = verify_manifest(manifest)
    queue = read(ROOT / config['queue'])
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration as verify_queue
    verify_queue(queue)
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    if (job != manifest['canonical_job'] or config['completed_original_epochs'] != 366
            or config['settings'] != read(ROOT / manifest['prior_chain_configuration'])['settings']
            or config['resource_lock'] != read(ROOT / manifest['prior_chain_configuration'])['resource_lock']
            or config['original_scope'] != manifest['original_scope']):
        raise ValueError('Keep the original five-seed queue, read-only guards and full research scope')
    return queue, job, ROOT / manifest['evidence_view'], evaluations
