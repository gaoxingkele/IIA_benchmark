"""Recompute MOMENT metrics and verify release-weight/native-data receipts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.run_tsad_experiment import evaluate_saved, verify_job
from scripts.flow_matching.run_tsad_queue import complete_result


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(snapshot_path):
    import torch
    queue_path = ROOT / 'configs/experiments/fm_moment_pretrained_queue.v1.json'
    queue = read(queue_path)
    registration = read(ROOT / 'docs/reports/fm_moment_pretrained_registration_2026-10-09.json')
    preflight = read(ROOT / 'docs/reports/fm_moment_pretrained_preflight_2026-10-09.json')
    assert sha(queue_path) == registration['queue_sha256'] == preflight['queue_sha256']
    frozen = {r['path']: r['sha256'] for j in queue['jobs'] for r in j['frozen_files']}
    verify_job(ROOT, {'frozen_files': [{'path': p, 'sha256': v} for p, v in frozen.items()]})
    acquisition = read(ROOT / 'configs/acquisition/fm_project_checkpoints_and_retries.v1.json')
    releases = [r for r in acquisition['sources'] if r['id'].startswith('moment_') and r['format'] == 'safetensors']
    provenance = []
    for release in releases:
        actual = sha(ROOT / release['path'])
        assert actual == release['checksum']['value']
        provenance.append({'id': release['id'], 'source_url': release['url'],
                           'revision': release['upstream_revision'], 'sha256': actual,
                           'size_bytes': (ROOT / release['path']).stat().st_size,
                           'boundary': release['version_boundary']})
    snapshot = read(snapshot_path)
    captured = {r['id']: r for r in snapshot['completed_new_runs']}
    records = []
    for job in queue['jobs']:
        if job['id'] not in captured:
            continue
        assert complete_result(ROOT, job)
        output = ROOT / job['output_directory']
        result = read(output / 'result.json')
        assert sha(output / 'result.json') == captured[job['id']]['result_sha256']
        config = read(ROOT / job['model_config'])
        receipt = torch.load(output / 'model.pt', map_location='cpu', weights_only=False)
        assert receipt['configuration'] == config and receipt['overrides'] == job['parameter_overrides']
        assert receipt['checkpoint_reference'] == config['parameters']['checkpoint_directory']
        assert receipt['checkpoint_load_audit'] == result['checkpoint_load_audit']
        loaded = result['checkpoint_load_audit']
        assert loaded['checkpoint_sha256'] == config['parameters']['checkpoint_sha256']
        assert loaded['config_sha256'] == config['parameters']['config_sha256']
        assert loaded['all_keys_strictly_loaded'] and not loaded['random_reinitialized_head']
        assert loaded['fine_tuning_epochs'] == 0 and loaded['fit_points_used_for_parameter_estimation'] == 0
        assert result['training_seconds'] == 0 and result['training_losses'] == []
        manifest = read(ROOT / job['dataset_manifest_path'])
        dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
        assert sha(ROOT / dataset['input_npz']) == dataset['input_sha256']
        with np.load(ROOT / dataset['input_npz'], allow_pickle=False) as source:
            arrays = {k: source[k].copy() for k in source.files}
        with np.load(output / 'scores.npz', allow_pickle=False) as source:
            scores = {k: source[k].copy() for k in source.files}
        if job['input_kind'] == 'industrial':
            roles = [(role, group) for role, groups in [('train', dataset['training_groups']),
                      ('validation', dataset['validation_groups']), ('test', dataset['entities'])] for group in groups]
        else:
            roles = [(role, group) for group in dataset['entities'] for role in ['train', 'validation', 'test']]
        training, validation, entities, points = [], [], {}, {}
        for role, group in roles:
            prefix = group['array_prefix']
            source_key = prefix + ('_test' if role == 'test' else '_values') if job['input_kind'] == 'industrial' else prefix + '_' + role
            values = scores[prefix + '_' + role]
            assert values.shape == (len(arrays[source_key]),) and np.isfinite(values).all()
            for suffix in ['_labels', '_timestamps', '_source_rows', '_fault_labels']:
                if prefix + suffix in arrays:
                    assert np.array_equal(scores[prefix + suffix], arrays[prefix + suffix])
            points[role] = points.get(role, 0) + len(values)
            if role == 'train':
                training.append(values)
            elif role == 'validation':
                validation.append(values)
            else:
                entities[group['entity_id']] = (scores[prefix + '_labels'], values)
        evaluation = dict(queue['evaluation'], bootstrap_block_points=max(queue['evaluation']['bootstrap_block_points'], job['window']))
        independent = evaluate_saved(entities, np.concatenate(validation), np.concatenate(training),
                                     dataset['input_sha256'], result['checkpoint_sha256'], evaluation, job['seed'])
        assert independent == result['metrics']
        records.append({'id': job['id'], 'dataset': job['dataset'], 'model_config': job['model_config'],
                        'result_sha256': sha(output / 'result.json'), 'native_points': points,
                        'native_labels_rows_and_coverage_verified': True, 'metrics_independently_recomputed': True,
                        'release_checkpoint_reference_verified': True, 'zero_shot_epochs': 0,
                        'primary_metrics': independent['strict']['1']['micro']})
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_snapshot': str(snapshot_path.relative_to(ROOT)),
              'source_snapshot_sha256': sha(snapshot_path), 'queue_sha256': sha(queue_path), 'registered_jobs': len(queue['jobs']),
              'frozen_files_verified': len(frozen), 'official_release_checkpoints': provenance,
              'completed_records_in_snapshot': records, 'complete_records_verified': len(records),
              'paper_equivalence': False, 'repeat_boundary': registration['repeat_boundary'], 'boundary': queue['boundary']}
    target = ROOT / 'projects/flow_matching_research/results/2026-10-09/moment_pretrained'
    write_json(target / 'execution_audit.json', report)
    (target / 'registration.json').write_bytes((ROOT / 'docs/reports/fm_moment_pretrained_registration_2026-10-09.json').read_bytes())
    (target / 'preflight.json').write_bytes((ROOT / 'docs/reports/fm_moment_pretrained_preflight_2026-10-09.json').read_bytes())
    print(json.dumps({'registered_jobs': len(queue['jobs']), 'verified_snapshot_results': len(records), 'frozen_files_verified': len(frozen)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='projects/flow_matching_research/results/2026-10-09/complete_metrics_0355/tsad_source_snapshot.json')
    audit(ROOT / parser.parse_args().snapshot)
