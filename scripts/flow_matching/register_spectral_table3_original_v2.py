"""Register a new pre-training source-scope correction, retaining all original slots."""
import json
from pathlib import Path

from scripts.flow_matching.register_spectral_table3_original_v1 import write_new
from scripts.flow_matching.run_spectral_table3_original_v2 import ROOT, read, sha, optimizer_assignments


def main():
    parent_path = 'configs/experiments/fm_spectral_table3_original_queue.v1.json'
    parent = read(ROOT / parent_path)
    retirement = ROOT / parent['state_root'] / 'controller_retirement_receipt.json'
    if not retirement.is_file() or read(retirement)['model_children']:
        raise ValueError('Preserved v1 pre-training retirement required')
    queue = dict(parent)
    queue.update(queue_path='configs/experiments/fm_spectral_table3_original_queue.v2.json',
        state_root='experiments/runs/fm_spectral_table3_original_v2',
        cpu_data_audit='experiments/runs/fm_spectral_table3_original_v2/preflight/cpu_data_v1/proof.json',
        original_queue=parent_path, original_queue_sha256=sha(ROOT / parent_path),
        attempt_scope='Same four released generator seed42 slots. No additional seed, budget, metric selection or original-source changes. Correct optimizer AST scope before any model launched.')
    sources = {r['path'] for r in parent['source_receipts']}
    sources.update([parent_path,
        'scripts/flow_matching/run_spectral_table3_original_v2.py',
        'scripts/flow_matching/run_spectral_table3_queue_v2.py',
        'scripts/flow_matching/register_spectral_table3_original_v2.py',
        'tests/test_fm_spectral_table3_original_v2.py'])
    jobs = []
    for original in parent['jobs']:
        previous = read(ROOT / original['model_config'])
        identity = original['id'] + '__source_scope_fix_v2'
        config = dict(previous, id=identity, original_canonical_id=original['id'],
            implementation='scripts.flow_matching.run_spectral_table3_original_v2:run_job')
        assignments = optimizer_assignments(ROOT / config['training_entry'])
        config['original_optimizer_assignments'] = [n.targets[0].id for n in assignments]
        config['preflight_scope_fix'] = 'Compile the unmodified optimizer assignments from the original logger with-block; full training/evaluation source and budget unchanged.'
        path = 'configs/models/fm_' + identity + '.json'
        write_new(ROOT / path, config)
        sources.add(path)
        artifact_root = Path(original['artifact_directory']).parent.parent / 'spectral_table3_original_v2'
        jobs.append(dict(original, id=identity, original_canonical_id=original['id'],
            model_config=path, model_config_sha256=sha(ROOT / path),
            artifact_directory=str(artifact_root / identity),
            preflight_artifact_directory=str(artifact_root / ('preflight_' + identity)),
            output_directory=queue['state_root'] + '/jobs/' + identity))
    queue['jobs'] = jobs
    queue['source_receipts'] = [dict(path=p, sha256=sha(ROOT / p)) for p in sorted(sources)]
    write_new(ROOT / queue['queue_path'], queue)
    print(json.dumps(dict(full_jobs=len(jobs), original_seed_slots=len({j['original_canonical_id'] for j in jobs}),
        source_bindings=len(sources), full_epochs_per_job=1000, queue=queue['queue_path'])))


if __name__ == '__main__':
    main()
