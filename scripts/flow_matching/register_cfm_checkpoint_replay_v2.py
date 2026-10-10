"""Select the original frozen CFM interpreter, preserving the failed audit attempt."""
import json

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha


def corrected_runtime(previous, original, root=ROOT):
    return dict(previous, python=str((root/original['python']).resolve()),
        state_root='experiments/runs/fm_cfm_checkpoint_replay_v2',
        output_directory='projects/flow_matching_research/results/2026-10-11/cfm_checkpoint_replay_v1/full_checkpoint_replay_v2',
        original_failed_configuration='configs/reproducibility/fm_cfm_checkpoint_replay.2026-10-11_v1.json',
        correction='Use the immutable original scientific queue interpreter, not the registration process system interpreter. All83 jobs, full test points, saved checkpoints, solver, numerical environment lock and comparison tolerance unchanged.')


def main():
    previous_path = 'configs/reproducibility/fm_cfm_checkpoint_replay.2026-10-11_v1.json'
    previous = read(ROOT/previous_path)
    failed = read(ROOT/previous['state_root']/'receipt.json')
    if failed['exit_code'] != 1 or failed['full_replay_passed']:
        raise ValueError('Preserved environment-only failure required')
    original = read(ROOT/previous['original_canonical_queue'])
    config = corrected_runtime(previous, original)
    sources = {r['path'] for r in previous['source_receipts']}
    sources.update([previous_path, 'scripts/flow_matching/register_cfm_checkpoint_replay_v2.py',
        'tests/test_fm_cfm_checkpoint_environment_v2.py'])
    config['source_receipts'] = [dict(path=p, sha256=sha(ROOT/p)) for p in sorted(sources)]
    destination = ROOT/'configs/reproducibility/fm_cfm_checkpoint_replay.2026-10-11_v2.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(full_checkpoint_replays=config['full_job_count'], python=config['python'],
        source_bindings=len(sources), budgets_and_tolerance_unchanged=True)))


if __name__ == '__main__':
    main()
