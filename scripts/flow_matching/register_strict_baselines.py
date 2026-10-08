"""Freeze existing baseline transcriptions on the FM full-data strict track."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def write_immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Frozen baseline registration differs: ' + str(path))
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_strict_baseline_execution.v1.json')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / settings['shared_input_manifest']).read_text(encoding='utf-8'))
    base_queue = json.loads((ROOT / settings['evaluation_reference_queue']).read_text(encoding='utf-8'))
    evaluation = {**base_queue['evaluation'], 'cpu_threads': settings['cpu_threads']}
    common = [args.config.relative_to(ROOT).as_posix(), settings['shared_input_manifest'],
              'scripts/flow_matching/run_tsad_experiment.py', 'scripts/flow_matching/prepare_tsad_execution.py',
              'scripts/flow_matching/model_registry.py', 'src/iia_benchmark/models/mtsad_detectors.py',
              'src/iia_benchmark/models/fm_legacy_window_adapter.py',
              'src/iia_benchmark/evaluation/mtsad_strict.py', 'src/iia_benchmark/evaluation/mtsad_metrics.py']
    hashes = [{'path': path, 'sha256': sha(ROOT / path)} for path in common]
    jobs, models = [], []
    for dataset_id in settings['datasets']:
        dataset = next(d for d in manifest['datasets'] if d['id'] == dataset_id)
        for method in settings['methods']:
            original_path = settings['source_config_directory'] + '/mtsad_' + method + '.json'
            original = json.loads((ROOT / original_path).read_text(encoding='utf-8'))
            params = {**original['parameters']}
            params.update({key: value for key, value in original.get('dataset_overrides', {}).get(dataset_id, {}).items()
                           if key in params})
            epochs = params.pop('epochs')
            params.pop('seed', None)
            params.pop('device', None)
            window = params['window']
            if any(min(e['fit_interval'][1], e['validation_interval'][1] - e['validation_interval'][0], e['test_points']) < window for e in dataset['entities']):
                raise ValueError('Baseline window would omit an entity: ' + method + ' ' + dataset_id)
            variants = [('registered', params)]
            if method == 'usad' and settings['include_usad_signed_loss_control']:
                variants.append(('signed_paper_loss', {**params, 'adversarial': True}))
            for variant, parameters in variants:
                identifier = f'fm_strict_{method}__{dataset_id}__{variant}'
                model_path = settings['model_output_directory'] + '/' + identifier + '.json'
                model = {'schema_version': 1, 'id': identifier, 'task': 'time_series_anomaly_detection',
                         'entrypoint': 'iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter',
                         'parameters': {'method': method, 'parameters': parameters, 'epochs': epochs, 'seed': settings['seeds'][0], 'device': 'cpu'},
                         'citation': 'https://doi.org/' + original['citation_doi'],
                         'data_citations': ['https://github.com/decisionintelligence/TAB'],
                         'source_config': original_path, 'source_config_sha256': sha(ROOT / original_path),
                         'original_reproduction_status': original['reproduction_status'],
                         'reproduction_status': 'existing_local_transcription_strict_track_not_author_pipeline_equivalence',
                         'known_deviations': {k: v for k, v in original.items() if k.endswith('_note') or k in ('source', 'protocol_note')},
                         'variant': variant, 'boundary': settings['boundary']}
                write_immutable(ROOT / model_path, model)
                models.append(model_path)
                frozen = hashes + [{'path': original_path, 'sha256': sha(ROOT / original_path)},
                                    {'path': model_path, 'sha256': sha(ROOT / model_path)},
                                    {'path': dataset['input_npz'], 'sha256': dataset['input_sha256']}]
                for seed in settings['seeds']:
                    job_id = identifier + '__seed' + str(seed)
                    jobs.append({'id': job_id, 'dataset': dataset_id, 'seed': seed, 'window': window,
                                 'model_config': model_path, 'dataset_manifest_path': settings['shared_input_manifest'],
                                 'parameter_overrides': {'seed': seed, 'device': 'cpu'},
                                 'output_directory': settings['output_root'] + '/jobs/' + job_id, 'frozen_files': frozen})
    queue = {'schema_version': 1, 'lane': 'cpu_baselines', 'jobs': jobs, 'evaluation': evaluation,
             'python': sys.executable, 'state_root': settings['output_root'], 'gpu_prerequisite_queues': [],
             'boundary': settings['boundary']}
    write_immutable(ROOT / settings['queue_path'], queue)
    write_json(ROOT / 'docs/reports/fm_strict_baseline_registration_2026-10-09.json',
               {'model_configs': len(models), 'jobs': len(jobs), 'queue': settings['queue_path'],
                'queue_sha256': sha(ROOT / settings['queue_path']), 'input_manifest_sha256': sha(ROOT / settings['shared_input_manifest']),
                'full_budgets': True, 'source_config': args.config.relative_to(ROOT).as_posix(), 'deferred': settings['deferred'],
                'boundary': 'Registration and three adapter invariants; successful full results still required.'})
    print(json.dumps({'registered_models': len(models), 'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
