"""Admit eight previously unhandled resource aborts without altering science.

All285 original canonical slots, datasets, seeds, model configs, optimizer
updates and solver tolerances remain frozen. Only eight resource-failed slots
receive fresh outputs and stronger40GiB admission. Numerical failures are not
selected for automatic repetition. CFM-TS: OpenReview Hqn4Aj7xrQ.
"""
import argparse
from pathlib import Path
import sys

from scripts.flow_matching import run_cfm_ts_low_memory_queue_v2 as original
from scripts.flow_matching import run_cfm_ts_stricter_admission_retry_v1 as dispatcher

ROOT = original.ROOT


def verify(runtime):
    for pin in runtime['source_receipts']:
        if original.sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen uncovered recovery source changed: ' + pin['path'])
    canonical = original.read(ROOT / runtime['controllers']['original']['queue'])
    selected, queue = original.binding(runtime, 'recovery')
    if len(queue['jobs']) != 285 or runtime['retry_admission_commit_headroom_bytes'] != 40*2**30:
        raise ValueError('Every285 canonical slot and stronger40GiB admission required')
    values = dict(verify=original.verify)
    from scripts.flow_matching.run_light_controller import load_functions
    load_functions(ROOT, 'scripts/flow_matching/register_cfm_ts_exact_recovery_v1.py', ['validate_recovery'], values)
    values['validate_recovery'](canonical, queue)
    evidence = original.read(ROOT / runtime['uncovered_resource_abort_audit'])
    identities = [row['id'] for row in evidence if not row['registered_selected_recovery']]
    chosen = [j for j in queue['jobs'] if j['id'] in selected['selected_job_ids']]
    if len(chosen) != 8 or {j['recovery_original_job_id'] for j in chosen} != set(identities):
        raise ValueError('Exactly all eight audited uncovered resource-failed slots required')
    byid = {j['id']:j for j in canonical['jobs']}
    for job in chosen:
        old = byid[job['recovery_original_job_id']]
        receiptpath = ROOT / canonical['state_root'] / 'receipts' / (old['id'] + '.json')
        receipt = original.read(receiptpath)
        if receipt['complete'] or not receipt['resource_observations']['resource_guard_aborted']:
            raise ValueError('Only actual resource aborts, never numerical errors, may be retried')
        artifact = Path(old['artifact_directory'])
        if artifact.exists() and any(p.suffix in ('.pt', '.pth', '.ckpt') for p in artifact.rglob('*') if p.is_file()):
            raise ValueError('Saved model/optimizer state requires its own continuation audit')
        if (ROOT / old['output_directory'] / 'result.json').exists():
            raise ValueError('Do not repeat completed original science')
    return chosen, queue


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); runtime = original.read(ROOT / args.runtime_config)
    chosen, queue = verify(runtime)
    if args.probe_only:
        print(dict(full_canonical_jobs=len(queue['jobs']), selected_uncovered_resource_aborts=len(chosen),
            selected_epochs=[j['epochs'] for j in chosen], selected_full_optimizer_updates=[j['expected_optimizer_updates'] for j in chosen],
            original_algorithm_worker_and_running_guards_unchanged=True,
            saved_partial_epoch_credit=0, numerical_failure_selected=False,
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            actual_training_executed=False))
        return
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda snapshot: None)
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--runtime-config', args.runtime_config, '--controller', 'recovery']
        dispatcher.bind()['main']()
    finally:
        sys.argv = argv


if __name__ == '__main__':
    main()
