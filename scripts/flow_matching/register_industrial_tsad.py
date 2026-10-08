"""Freeze unchanged method budgets on full grouped industrial detection inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.model_registry import load_model_config, verify_config
from scripts.flow_matching.prepare_industrial_tsad import prepare
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def model_roster(root, settings):
    paths = list(load_model_config(root/settings['model_roster_config'])['model_configs'])
    for queue in settings['additional_model_queue_rosters']:
        paths.extend(dict.fromkeys(j['model_config'] for j in load_model_config(root/queue)['jobs']))
    paths.extend(settings['baseline_models'])
    if len(paths) != settings['expected_model_configs'] or len(set(paths)) != len(paths):
        raise ValueError('Complete industrial model roster differs or duplicates')
    return paths


def model_window(configuration, default):
    parameters = configuration['parameters']
    return int(parameters.get('window', parameters.get('parameters', {}).get('window', default)))


def preserve_json(path, value):
    if path.exists():
        if load_model_config(path) != value:
            raise ValueError('Frozen artifact differs; register a new version: '+str(path))
    else:
        write_json(path, value)


def register(root, settings_path):
    settings = load_model_config(settings_path)
    manifest = prepare(root, settings)
    manifest_path = root/settings['prepared_root']/'manifest.json'
    paths = model_roster(root, settings)
    sources = [settings_path, manifest_path, root/settings['preflight_report'], root/settings['model_roster_config'],
               root/settings['evaluation_reference_queue'], *[root/p for p in settings['additional_model_queue_rosters']]]
    code = ['scripts/flow_matching/prepare_industrial_tsad.py', 'scripts/flow_matching/register_industrial_tsad.py',
            'scripts/flow_matching/run_industrial_tsad_experiment.py', 'scripts/flow_matching/run_industrial_tsad_queue.py',
            'scripts/flow_matching/start_industrial_tsad.py', 'scripts/flow_matching/preflight_industrial_tsad.py',
            'scripts/flow_matching/prepare_tsad_execution.py', 'scripts/flow_matching/run_tsad_experiment.py',
            'scripts/flow_matching/run_tsad_queue.py', 'scripts/flow_matching/model_registry.py', 'scripts/flow_matching/phase_gate.py',
            'src/iia_benchmark/data/tep.py', 'src/iia_benchmark/data/skab.py', 'src/iia_benchmark/data/pronto.py', 'src/iia_benchmark/data/schema.py',
            'src/iia_benchmark/evaluation/mtsad_strict.py', 'src/iia_benchmark/evaluation/mtsad_metrics.py',
            'src/iia_benchmark/models/fm_objective_detectors.py', 'src/iia_benchmark/models/fm_objective_stable.py',
            'src/iia_benchmark/models/fm_reflow.py', 'src/iia_benchmark/models/fm_native.py',
            'src/iia_benchmark/models/fm_comparator_reconstructions.py', 'src/iia_benchmark/models/fm_legacy_window_adapter.py',
            'src/iia_benchmark/models/mtsad_detectors.py']
    sources += [root/p for p in code]
    common = [{'path': p.relative_to(root).as_posix(), 'sha256': sha(p)} for p in dict.fromkeys(sources)]
    evaluation = {**load_model_config(root/settings['evaluation_reference_queue'])['evaluation'], **settings['evaluation_overrides']}
    queues = {'cpu': [], 'gpu': []}
    for dataset in manifest['datasets']:
        inputs = [{'path': dataset['input_npz'], 'sha256': dataset['input_sha256']},
                  {'path': dataset['source_config'], 'sha256': sha(root/dataset['source_config'])}]
        inputs += [{'path': s['path'], 'sha256': s['sha256']} for s in dataset['source_files']]
        for model_path in paths:
            config = load_model_config(root/model_path)
            verify_config(root/model_path, root)
            symbol = config['entrypoint'].split(':')[-1]
            if symbol not in settings['allowed_fit_score_classes']:
                raise ValueError('Unreviewed industrial adapter: '+symbol)
            lane = 'cpu' if symbol in settings['cpu_classes'] else 'gpu'
            window = model_window(config, settings['default_window'])
            if min(g['points'] for g in dataset['training_groups']+dataset['validation_groups']+dataset['entities']) < window:
                raise ValueError('A complete native group is shorter than this method window')
            for seed in settings['seeds']:
                identifier = f"{config['id']}__industrial_v1__{dataset['id']}__seed{seed}"
                queues[lane].append({'id': identifier, 'dataset': dataset['id'], 'seed': seed, 'model_config': model_path,
                                     'window': window, 'dataset_manifest_path': manifest_path.relative_to(root).as_posix(),
                                     'parameter_overrides': {'seed': seed, 'device': 'cpu' if lane == 'cpu' else 'cuda:0'},
                                     'output_directory': settings['output_root']+'/jobs/'+identifier,
                                     'frozen_files': common+inputs+[{'path': model_path, 'sha256': sha(root/model_path)}],
                                     'protocol': 'grouped_industrial_full_timestamp_transfer', 'training_group_pooling': 'each_unique_group_once'})
    records = {}
    for lane, jobs in queues.items():
        target = root/settings['queue_paths'][lane]
        queue = {'schema_version': 1, 'lane': lane, 'jobs': jobs, 'evaluation': evaluation, 'python': sys.executable,
                 'state_root': settings['output_root'], 'experiment_runner': 'scripts.flow_matching.run_industrial_tsad_experiment',
                 'gpu_prerequisite_queues': settings['gpu_prerequisite_queues'] if lane == 'gpu' else [],
                 'gpu_prior_tsad_queues': settings['gpu_prior_tsad_queues'] if lane == 'gpu' else [], 'boundary': settings['boundary']}
        preserve_json(target, queue)
        records[lane] = {'path': target.relative_to(root).as_posix(), 'sha256': sha(target), 'jobs': len(jobs)}
    ranges = load_model_config(root/settings['range_template_config'])
    ranges.update(id='fm_industrial_tab_range_v1', queue_paths=list(settings['queue_paths'].values()), output_root=settings['range_output_root'])
    preserve_json(root/settings['range_config'], ranges)
    preserve_json(root/settings['registration_report'], {'config': settings_path.relative_to(root).as_posix(), 'config_sha256': sha(settings_path),
        'model_configs': paths, 'model_count': len(paths), 'dataset_split_settings': len(manifest['datasets']), 'seeds': settings['seeds'],
        'queues': records, 'dataset_manifest': manifest_path.relative_to(root).as_posix(), 'dataset_manifest_sha256': sha(manifest_path),
        'range_config': settings['range_config'], 'epochs_and_capacity_preserved': True,
        'primary_test_tuning': False, 'complete_timestamp_coverage_required': True, 'boundary': settings['boundary']})
    print(json.dumps(records), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    args = parser.parse_args()
    register(ROOT, args.config)


if __name__ == '__main__':
    main()
