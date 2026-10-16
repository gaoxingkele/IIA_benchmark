"""Cold, strict source-result provenance before any native model is replayed."""
import math
import re
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching import run_otcfm_singlecell_native_v4 as native


def validate_record(queue, job, result, guard):
    if (result.get('status') != 'completed' or result.get('job') != job
            or result.get('job_sha256') != native.identity(job)
            or not guard.get('passed') or guard.get('exit_code') != 0 or guard.get('resource_guard_aborted')):
        raise ValueError('Actual complete original job and successful native process guard required')
    recipe = job['scientific_protocol']
    if (result['maximum_epochs'] != recipe['max_epochs'] or result['batch_size'] != recipe['batch_size']
            or result['parameter_count'] != 9093 or result['full_timepoint_cell_counts'] != job['data']['days']
            or not result['original_loader_whiten_true']):
        raise ValueError('Complete native data, architecture and original scientific protocol required')
    epochs = result['epoch_history']
    if (len(epochs) != result['actual_completed_epochs'] or not epochs
            or [r['epoch'] for r in epochs] != list(range(1, len(epochs) + 1))):
        raise ValueError('Every full training epoch must be recorded exactly once')
    batches = min(native.contract.fraction_lengths(n, [.8, .1, .1])[0] // recipe['batch_size']
        for n in job['data']['days'].values())
    for index, row in enumerate(epochs):
        expected = row['epoch'] * batches
        if row['optimizer_updates'] == expected: continue
        partial_paper_cap = (index == len(epochs) - 1 and recipe['maximum_training_wall_seconds'] is not None
            and result['termination_reason'] == 'native_paper_wall_time_cap'
            and result['training_seconds'] >= recipe['maximum_training_wall_seconds']
            and (row['epoch'] - 1) * batches < row['optimizer_updates'] < expected)
        if not partial_paper_cap:
            raise ValueError('Every complete min-size-loader epoch/update required; only actual registered paper wall cap may end its last epoch partway')
    # Lightning1.8 test(ckpt_path=best) restores the selected checkpoint's
    # global_step. The original result records that post-test counter, whereas
    # the immutable epoch_history records all actual training updates.
    match = re.fullmatch(r'epoch_(\d+)\.ckpt', Path(result['best_checkpoint']).name)
    finite = [row for row in result['validation_history'] if row['finite']]
    if not match or not finite: raise ValueError('Actual validation-selected checkpoint and complete validation history required')
    selected_epoch = int(match.group(1)) + 1
    selected = next((row for row in finite if row['epoch'] == selected_epoch), None)
    if (selected is None or selected['validation_loss'] != min(row['validation_loss'] for row in finite)
            or selected_epoch > len(epochs)
            or result['optimizer_updates'] != epochs[selected_epoch - 1]['optimizer_updates']):
        raise ValueError('Post-test step counter must equal actual validation-selected checkpoint updates; total training updates stay separate')
    reason = native.contract.terminal_reason(recipe, len(epochs), result['early_stopping_epoch'],
        result['training_seconds'], False, True)
    if reason != result['termination_reason'] or result['early_stopping_patience_checks'] != recipe['early_stopping_patience_validation_checks']:
        raise ValueError('Unchanged full native stopping rule required')
    metrics = result['test_metrics']
    if (not metrics or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values())
            or metrics[queue['primary_metric']] != result['primary_heldout_W1']):
        raise ValueError('Every complete native test metric and heldout1-Wasserstein required')
    if result['historical_paper_equivalence_certified'] or result['strict_TAB_result']:
        raise ValueError('Available single-cell inputs cannot certify historical identity or strictTAB performance')


def inspect(queue, job):
    base = Path(job['artifact_directory']); path = base / 'result.json'
    guard_path = ROOT / queue['state_root'] / 'attempts' / base.name / 'guard_receipt.json'
    result, guard = read(path), read(guard_path)
    validate_record(queue, job, result, guard)
    if guard['result_sha256'] != sha(path): raise ValueError('Original result changed after terminal guard')
    if result['queue_sha256'] != sha(ROOT / native.QUEUE): raise ValueError('Exact complete120 source queue required')
    if result['environment_lock_sha256'] != sha(ROOT / native.ENVIRONMENT): raise ValueError('Exact native author package lock required')
    if sha(ROOT / job['data']['path']) != job['data']['sha256']: raise ValueError('Complete original input changed')
    if not result['artifacts']: raise ValueError('Actual checkpoint, optimizer and native test artifacts required')
    for item in result['artifacts']:
        p = Path(item['path'])
        if not p.resolve().is_relative_to(base.resolve()) or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('Every immutable original trained artifact must match: ' + str(p))
    if sha(result['best_checkpoint']) != result['best_checkpoint_sha256']:
        raise ValueError('Actual original validation-selected checkpoint changed')
    return result, dict(result_path=str(path), result_sha256=sha(path), guard_path=str(guard_path),
        guard_sha256=sha(guard_path), full_source_artifacts_checked=len(result['artifacts']),
        best_checkpoint_sha256=result['best_checkpoint_sha256'],
        actual_total_training_optimizer_updates=result['epoch_history'][-1]['optimizer_updates'],
        original_posttest_selected_checkpoint_updates=result['optimizer_updates'],
        original_source_result_kept_unchanged=True)
