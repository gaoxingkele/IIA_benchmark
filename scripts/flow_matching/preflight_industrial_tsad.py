"""Audit native data coverage and all industrial adapter/budget contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.model_registry import load_model_config, build_configured_method, verify_config
from scripts.flow_matching.prepare_industrial_tsad import prepare
from scripts.flow_matching.prepare_tsad_execution import sha, write_json
from scripts.flow_matching.register_industrial_tsad import model_roster, model_window
from scripts.flow_matching.run_industrial_tsad_experiment import pooled_training_windows
from scripts.flow_matching.run_tsad_experiment import covering_windows, align_scores


def preflight(root, config_path):
    settings = load_model_config(config_path)
    manifest = prepare(root, settings)
    records, raw = [], {}
    for path in model_roster(root, settings):
        verify_config(root/path, root)
        model = build_configured_method(root/path, {'device': 'cpu', 'seed': settings['seeds'][0]})
        if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'score', None)):
            raise ValueError('Missing industrial fit/score methods')
    for dataset in manifest['datasets']:
        for source in dataset['source_files']:
            if sha(root/source['path']) != source['sha256']:
                raise ValueError('Native industrial source changed')
            raw[source['path']] = source['sha256']
        with np.load(root/dataset['input_npz'], allow_pickle=False) as source:
            data = {k: source[k].copy() for k in source.files}
        for path in model_roster(root, settings):
            configuration = load_model_config(root/path)
            window = model_window(configuration, settings['default_window'])
            fit, budget = pooled_training_windows(data, dataset, window)
            for role, groups in [('train', dataset['training_groups']), ('validation', dataset['validation_groups']), ('test', dataset['entities'])]:
                for group in groups:
                    values = data[group['array_prefix']+('_test' if role == 'test' else '_values')]
                    windows, starts = covering_windows(values, window)
                    coverage = align_scores(np.ones((len(windows), window)), starts, len(values), window)
                    if not np.array_equal(coverage, np.ones(len(values))):
                        raise ValueError('A native timestamp would be omitted or cross a boundary')
            records.append({'dataset': dataset['id'], 'model_config': path, 'window': window, 'epochs': configuration['parameters']['epochs'],
                            'full_fit_windows': len(fit), 'fit_points': sum(b['points'] for b in budget),
                            'test_entities': len(dataset['entities']), 'full_test_points': sum(g['test_points'] for g in dataset['entities'])})
    report = {'status': 'native_data_and_callable_contract_preflight_passed', 'config_sha256': sha(config_path),
              'model_count': settings['expected_model_configs'], 'dataset_split_settings': len(manifest['datasets']),
              'native_sources_verified': len(raw), 'native_source_hashes': raw, 'records': records,
              'declared_epochs_reduced': False, 'raw_overwritten': False,
              'boundary': 'Native full data, grouped splits, callable constructors and coverage verified. No fitted model, synthetic or native benchmark performance is claimed by preflight.'}
    destination = root/settings['preflight_report']
    if destination.exists() and load_model_config(destination) != report:
        raise ValueError('Existing industrial preflight differs; preserve it')
    if not destination.exists():
        write_json(destination, report)
    print(json.dumps({k: report[k] for k in ['status', 'model_count', 'dataset_split_settings', 'native_sources_verified']}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    args = parser.parse_args()
    preflight(ROOT, args.config)


if __name__ == '__main__':
    main()
