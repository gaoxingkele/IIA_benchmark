"""Freeze full, entity-aware TSAD inputs without altering downloaded data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.data.mtsad import load_mtsad_split


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def split_and_normalize(train, test, labels, recipe):
    """Purge the train/validation boundary; fit every repair and scale on fit only."""
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    labels = np.asarray(labels)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError('Feature shapes differ')
    if labels.shape != (len(test),) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Unaligned or nonbinary test labels')
    cut = int(len(train) * recipe['train_fraction'])
    fit_end = cut - recipe['purge_points']
    if min(fit_end, len(train) - cut, len(test)) < recipe['minimum_segment_points']:
        raise ValueError('Entity too short for the frozen window contracts')
    fit, validation = train[:fit_end].copy(), train[cut:].copy()
    arrays = {'train': fit, 'validation': validation, 'test': test.copy()}
    median = np.nanmedian(np.where(np.isfinite(fit), fit, np.nan), axis=0)
    if not np.isfinite(median).all():
        raise ValueError('Training channel has no finite observation')
    repairs = {}
    for name, values in arrays.items():
        bad = ~np.isfinite(values)
        repairs[name] = int(bad.sum())
        values[bad] = np.broadcast_to(median, values.shape)[bad]
    mean, std = arrays['train'].mean(0), arrays['train'].std(0)
    safe = np.where(std > 0, std, 1.)
    arrays = {name: ((values - mean) / safe).astype(np.float32) for name, values in arrays.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError('Nonfinite standardized values')
    audit = {'raw_train_points': len(train), 'fit_interval': [0, fit_end],
             'purged_interval': [fit_end, cut], 'validation_interval': [cut, len(train)],
             'test_points': len(test), 'test_anomaly_points': int(labels.sum()),
             'features': train.shape[1], 'repairs': repairs,
             'median': median.tolist(), 'mean': mean.tolist(), 'std': std.tolist(),
             'constant_channels': np.flatnonzero(std == 0).tolist()}
    return arrays, labels.astype(np.uint8), audit


def load_entities(root, spec):
    if spec['loader'] == 'native_single':
        split = load_mtsad_split(spec['native_dataset'], root=root / spec['root'], config_dir=root / spec['config_dir'])
        # Native PSM reference loader replaces nonfinite values with zero.
        # Preserve that source-loader repair explicitly, before train-only scaling.
        yield spec['id'], split.train, split.test, split.test_labels, list(split.source_paths.values()), {
            'source_loader_repairs': split.repairs, 'feature_names': list(split.feature_names),
            'timestamp_identity': 'source row order; timestamp column excluded by registered native loader'}
        return
    base = root / spec['root']
    trains, tests = sorted(base.glob('*_train.csv')), sorted(base.glob('*_test.csv'))
    train_ids, test_ids = {p.name[:-10] for p in trains}, {p.name[:-9] for p in tests}
    if not tests or test_ids - train_ids or train_ids - test_ids != set(spec.get('known_train_only_entities', [])):
        raise ValueError('Missing entity train/test pair: ' + spec['id'])
    for train_path in trains:
        entity = train_path.name[:-10]
        if entity not in test_ids:
            continue  # Explicitly catalogued train-only asset; no test performance can be computed.
        test_path = base / (entity + '_test.csv')
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        exclude = set(spec['excluded_columns']) | {spec['label_column']}
        features = [c for c in train.columns if c not in exclude]
        if features != [c for c in test.columns if c not in exclude]:
            raise ValueError('Entity feature ordering mismatch: ' + entity)
        if spec['label_column'] not in test:
            raise ValueError('Test labels absent: ' + entity)
        train_labels = train[spec['label_column']] if spec['label_column'] in train else None
        timestamps = {}
        for split_name, frame in [('train', train), ('test', test)]:
            if 'timestamp' not in frame or frame['timestamp'].duplicated().any():
                raise ValueError('Missing/duplicate within-entity timestamp: ' + entity)
            timestamps[split_name] = {'first': str(frame['timestamp'].iloc[0]), 'last': str(frame['timestamp'].iloc[-1]),
                                      'monotonic': bool(frame['timestamp'].is_monotonic_increasing)}
            if not timestamps[split_name]['monotonic']:
                raise ValueError('Nonmonotonic timestamps: ' + entity)
        yield entity, train[features].to_numpy(float), test[features].to_numpy(float), test[spec['label_column']].to_numpy(), [str(train_path), str(test_path)], {
            'feature_names': features, 'timestamps': timestamps,
            'training_anomaly_points': int(train_labels.sum()) if train_labels is not None else None,
            'timestamp_identity': 'split-specific source timestamps and row order; entity boundaries retained'}


def prepare(root, settings):
    output = root / settings['prepared_root']
    manifest_path = output / 'manifest.json'
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['preparation_sha256'] != fingerprint:
            raise ValueError('Existing prepared inputs belong to a different recipe')
        for record in previous['datasets']:
            for source in record['source_files']:
                if sha(root / source['path']) != source['sha256']:
                    raise ValueError('Prepared raw source changed')
            if sha(root / record['input_npz']) != record['input_sha256']:
                raise ValueError('Prepared inputs changed')
        return previous
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in settings['datasets']:
        payload, audits, sources = {}, [], {}
        for index, (entity, train, test, labels, paths, extra) in enumerate(load_entities(root, spec)):
            arrays, labels, audit = split_and_normalize(train, test, labels, settings['recipe'])
            prefix = f'e{index:03d}'
            payload.update({prefix + '_' + name: values for name, values in arrays.items()})
            payload[prefix + '_labels'] = labels
            audits.append({'entity_id': entity, 'array_prefix': prefix, **audit, **extra})
            for path in paths:
                path = Path(path)
                relative = path.relative_to(root).as_posix()
                sources[relative] = {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}
        if len({r['features'] for r in audits}) != 1:
            raise ValueError('Pooled model needs a consistent feature count')
        target = output / (spec['id'] + '.npz')
        if target.exists():
            with np.load(target, allow_pickle=False) as previous:
                if set(previous.files) != set(payload) or any(not np.array_equal(previous[key], value) for key, value in payload.items()):
                    raise FileExistsError('Different partial prepared data preserved: ' + str(target))
        else:
            np.savez_compressed(target, **payload)
        train_only = []
        for entity in spec.get('known_train_only_entities', []):
            path = root / spec['root'] / (entity + '_train.csv')
            relative = path.relative_to(root).as_posix()
            sources[relative] = {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}
            train_only.append({'entity_id': entity, 'path': relative, 'status': 'test_not_provided; excluded from paired evaluation, raw retained'})
        records.append({'id': spec['id'], 'input_npz': target.relative_to(root).as_posix(),
                        'input_sha256': sha(target), 'source_files': list(sources.values()), 'entities': audits,
                        'known_train_only_entities': train_only,
                        'grouped_splits_verified': True, 'grouping': 'entity-isolated windows; chronological train/validation with purge; native held-out test',
                        'generalization': 'within published entities; no unseen-entity claim', 'source_track': spec['source_track']})
        print(json.dumps({'prepared_dataset': spec['id'], 'entities': len(audits), 'test_points': sum(e['test_points'] for e in audits)}), flush=True)
    manifest = {'preparation_sha256': fingerprint, 'recipe': settings['recipe'], 'datasets': records,
                'boundary': 'Full local TSAD benchmark track, not certified original-paper split/feature reproduction. Native PSM zero repairs are disclosed. Native tests untouched; no test-fit preprocessing.'}
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/fm_tsad_execution.v1.json')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    prepare(ROOT, settings)


if __name__ == '__main__':
    main()
