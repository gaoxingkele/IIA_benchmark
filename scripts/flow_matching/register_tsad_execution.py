"""Register every currently callable full-window TSAD method and ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_tsad_execution import prepare, sha, write_json


def register(root, settings_path):
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    manifest = prepare(root, settings)
    manifest_path = root / settings['prepared_root'] / 'manifest.json'
    common = [settings_path, manifest_path, root / 'scripts/flow_matching/run_tsad_experiment.py',
              root / 'scripts/flow_matching/prepare_tsad_execution.py',
              root / 'src/iia_benchmark/evaluation/mtsad_strict.py',
              root / 'src/iia_benchmark/evaluation/mtsad_metrics.py']
    common_hashes = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in common]
    queues = {'cpu': [], 'gpu': []}
    for dataset in manifest['datasets']:
        for model_path in settings['model_configs']:
            path = root / model_path
            config = json.loads(path.read_text(encoding='utf-8'))
            symbol = config['entrypoint'].split(':')[-1]
            if symbol not in settings['allowed_fit_score_classes']:
                raise ValueError('Unreviewed task adapter: ' + model_path)
            module = config['entrypoint'].split(':')[0]
            source = root / 'src' / Path(*module.split('.')).with_suffix('.py')
            lane = 'cpu' if symbol in settings['cpu_classes'] else 'gpu'
            for seed in settings['seeds']:
                identifier = f"{config['id']}__{dataset['id']}__seed{seed}"
                frozen = common_hashes + [{'path': model_path, 'sha256': sha(path)},
                                          {'path': source.relative_to(root).as_posix(), 'sha256': sha(source)},
                                          {'path': dataset['input_npz'], 'sha256': dataset['input_sha256']}]
                queues[lane].append({'id': identifier, 'dataset': dataset['id'], 'seed': seed,
                                     'model_config': model_path, 'window': config['parameters'].get('window', settings['default_window']),
                                     'dataset_manifest_path': manifest_path.relative_to(root).as_posix(),
                                     'parameter_overrides': {'seed': seed, 'device': 'cpu' if lane == 'cpu' else 'cuda:0'},
                                     'output_directory': settings['output_root'] + '/jobs/' + identifier,
                                     'frozen_files': frozen})
    records = {}
    for lane, jobs in queues.items():
        target = root / settings['queue_paths'][lane]
        queue = {'schema_version': 1, 'lane': lane, 'jobs': jobs, 'evaluation': settings['evaluation'],
                 'python': sys.executable, 'state_root': settings['output_root'],
                 'gpu_prerequisite_queues': settings['gpu_prerequisite_queues'] if lane == 'gpu' else [],
                 'boundary': 'Full local TSAD reconstruction/adaptation track; not author equivalent. No epoch/capacity reduction. Remaining source-only and other-task obligations retained in project scope.'}
        if target.exists():
            previous = json.loads(target.read_text(encoding='utf-8'))
            if previous != queue:
                raise ValueError('Frozen queue differs; register a new version')
        else:
            write_json(target, queue)
        records[lane] = {'path': target.relative_to(root).as_posix(), 'jobs': len(jobs), 'sha256': sha(target)}
    write_json(root / settings['registration_report'], {
        'config': settings_path.relative_to(root).as_posix(), 'config_sha256': sha(settings_path),
        'model_configs': len(settings['model_configs']), 'datasets': len(manifest['datasets']),
        'seeds': settings['seeds'], 'queues': records,
        'dataset_manifest': manifest_path.relative_to(root).as_posix(), 'dataset_manifest_sha256': sha(manifest_path),
        'epochs_preserved': True, 'full_test_coverage_required': True, 'test_tuning_primary': False,
        'boundary': 'Execution registration, not completed experiments. Existing imputation jobs and all 45-paper inventory obligations remain active.'})
    print(json.dumps(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_tsad_execution.v1.json')
    args = parser.parse_args()
    register(ROOT, args.config)


if __name__ == '__main__':
    main()
