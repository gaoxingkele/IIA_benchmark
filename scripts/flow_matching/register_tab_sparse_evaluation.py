"""Freeze full-scope, bit-exact outer-partition range-evaluation successors."""
import json
from pathlib import Path

from scripts.flow_matching.prepare_tsad_execution import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
PARENTS = ['fm_tab_range_execution.v1.json', 'fm_reflow_tab_range.v1.json',
           'fm_industrial_tab_range.v1.json', 'fm_moment_pretrained_tab_range.v1.json']


def main():
    records = []
    proof = ROOT / 'docs/reports/fm_tab_sparse_affiliation_differential_2026-10-09.json'
    assert all(r['all_six_reference_outputs_bit_exact'] for r in json.loads(proof.read_text(encoding='utf-8'))['records'])
    sources = [ROOT / 'scripts/flow_matching' / name for name in [
        'tab_affiliation_sparse.py', 'tab_sparse_execution.py', 'evaluate_tab_ranges_sparse_job.py',
        'run_tab_ranges_sparse_queue.py', 'register_tab_sparse_evaluation.py', 'heavy_scheduler_v2.py', 'heavy_resources.py']]
    sources.append(proof)
    for name in PARENTS:
        parent_path = ROOT / 'configs/experiments' / name
        config = json.loads(parent_path.read_text(encoding='utf-8'))
        for source in config['source_receipts']:
            assert sha(ROOT / source['path']) == source['sha256'], source['path']
        config.update(schema_version=2, parent_config=parent_path.relative_to(ROOT).as_posix(), parent_config_sha256=sha(parent_path),
                      output_root=config['output_root'].replace('_v1', '_sparse_v2'), worker_script='run_tab_ranges_sparse_queue.py',
                      runtime_resources={'minimum_available_memory_bytes': 8*1024**3,
                                         'minimum_commit_headroom_bytes': 12*1024**3,
                                         'emergency_commit_headroom_bytes': 8*1024**3},
                      engine='pinned_integrals_with_sparse_outer_partition')
        config['source_receipts'] += [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sources + [parent_path]]
        config['boundary'] += (' Only outer zero-contribution None intersections omitted; original integral functions, inner recall partition, '
                               'every entity/timepoint, all 250 VUS thresholds and all integer buffers unchanged. '
                               'Completed original evaluations rebound with all numeric values and source checksums preserved. '
                               '12 GiB gate applies to isolated metric processes, not a change to any original model resource gate/budget.')
        target = parent_path.with_name(name.replace('.v1.json', '.sparse.v2.json'))
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != config:
            raise ValueError('Sparse successor config already frozen differently')
        write_json(target, config)
        count = sum(len(json.loads((ROOT / p).read_text(encoding='utf-8'))['jobs']) for p in config['queue_paths'])
        records.append({'parent_config': config['parent_config'], 'config': target.relative_to(ROOT).as_posix(),
                        'config_sha256': sha(target), 'registered_attempts': count, 'output_root': config['output_root']})
    write_json(ROOT / 'docs/reports/fm_tab_sparse_registration_2026-10-09.json',
               {'configs': records, 'all_original_queue_scopes_preserved': True, 'metric_formula_or_resolution_changed': False,
                'existing_models_scores_thresholds_and_completed_evaluations_preserved': True, 'all_experiments_complete': False})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
