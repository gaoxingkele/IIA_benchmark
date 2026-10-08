"""Prepare complete industrial anomaly-detection runs with independent groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.data.tep import load_tep_ascii
from iia_benchmark.data.skab import load_skab_csv
from iia_benchmark.data.pronto import load_pronto_merged_csv
from scripts.flow_matching.prepare_tsad_execution import sha, write_json


def contiguous_normal_intervals(labels):
    normal = np.asarray(labels) == 'Normal'
    edges = np.diff(np.r_[False, normal, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


def validate_group_roles(roles):
    if set(roles) != {'train', 'validation', 'test'} or any(not groups for groups in roles.values()):
        raise ValueError('All three roles need nonempty groups')
    identities = {}
    for role, groups in roles.items():
        identities[role] = {g['group_id'] for g in groups}
        intervals = {}
        for group in groups:
            start, stop = group['source_interval']
            if start < 0 or stop <= start or stop - start != len(group['values']):
                raise ValueError('Source interval does not cover this group')
            prior = intervals.setdefault(group['source_path'], [])
            if any(start < b and a < stop for a, b in prior):
                raise ValueError('Repeated or overlapping source samples within role')
            prior.append((start, stop))
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if identities[left] & identities[right]:
            raise ValueError('A run/day group is shared across roles')
        if {g['source_path'] for g in roles[left]} & {g['source_path'] for g in roles[right]}:
            raise ValueError('A raw source is shared across roles')


def normalize_groups(roles):
    validate_group_roles(roles)
    if any(g['values'].ndim != 2 for groups in roles.values() for g in groups):
        raise ValueError('Industrial values need two dimensions')
    features = {g['values'].shape[1] for groups in roles.values() for g in groups}
    if len(features) != 1:
        raise ValueError('Inconsistent industrial feature dimensions')
    fit = np.concatenate([g['values'] for g in roles['train']]).astype(float)
    medians = np.nanmedian(np.where(np.isfinite(fit), fit, np.nan), axis=0)
    if not np.isfinite(medians).all():
        raise ValueError('A training channel has no finite samples')
    repaired = {}
    repairs = {}
    for role, groups in roles.items():
        repaired[role] = []
        repairs[role] = []
        for group in groups:
            values = np.asarray(group['values'], dtype=float).copy()
            bad = ~np.isfinite(values)
            values[bad] = np.broadcast_to(medians, values.shape)[bad]
            repairs[role].append(int(bad.sum()))
            repaired[role].append(values)
    fit = np.concatenate(repaired['train'])
    mean, std = fit.mean(0), fit.std(0)
    scale = np.where(std > 0, std, 1.)
    normalized = {role: [((v - mean) / scale).astype(np.float32) for v in values] for role, values in repaired.items()}
    if not all(np.isfinite(v).all() for values in normalized.values() for v in values):
        raise ValueError('Nonfinite standardized industrial values')
    return normalized, {'median': medians.tolist(), 'mean': mean.tolist(), 'std': std.tolist(),
                        'constant_channels': np.flatnonzero(std == 0).tolist(), 'repairs': repairs,
                        'fit_points': len(fit), 'fit_role': 'train_only_unique_source_intervals'}


def group(path, run_id, values, labels, names, timestamps, start=0, stop=None, fault_labels=None):
    stop = len(values) if stop is None else stop
    return {'source_path': path, 'group_id': run_id, 'entity_id': f'{run_id}[{start}:{stop}]',
            'source_interval': [start, stop], 'values': np.asarray(values[start:stop]),
            'labels': np.asarray(labels[start:stop], dtype=np.uint8), 'feature_names': list(names),
            'timestamps': np.asarray(timestamps[start:stop], dtype=float),
            'fault_labels': None if fault_labels is None else np.asarray(fault_labels[start:stop], dtype=str)}


def load_groups(root, spec, config):
    roles = {'train': [], 'validation': [], 'test': []}
    selection = {}
    if spec['loader'] == 'tep_whole_runs':
        paths = sorted(p for p in root.glob(config['test_glob']) if p.stem not in config['exclude_test_stems'])
        if {p.stem for p in paths} != {f'd{i:02d}_te' for i in range(1, 22)}:
            raise ValueError('Classic TEP full fault roster differs')
        for role, paths_role in [('train', [root / config['train_path']]), ('validation', [root / spec['validation_path']]), ('test', paths)]:
            for path in paths_role:
                run = load_tep_ascii(path, fault_start=config['fault_start'] if role == 'test' else None, sample_period=config['sample_period'])
                relative = path.relative_to(root).as_posix()
                roles[role].append(group(relative, relative, run.values, run.abnormal, run.feature_names, run.timestamps))
        selection['label_rule'] = f"Native classic fault test onset at zero-based index {config['fault_start']}; independent normal runs labelled normal"
    elif spec['loader'] == 'skab_whole_experiments':
        paths = sorted({p for pattern in config['test_globs'] for p in root.glob(pattern)})
        validation = root / spec['validation_path']
        if len(paths) != spec['expected_test_entities'] + 1 or validation not in paths:
            raise ValueError('SKAB full experiment roster differs or validation is absent')
        for role, paths_role in [('train', [root / config['train_path']]), ('validation', [validation]), ('test', [p for p in paths if p != validation])]:
            for path in paths_role:
                run = load_skab_csv(path)
                relative = path.relative_to(root).as_posix()
                roles[role].append(group(relative, relative, run.values, run.abnormal, run.feature_names, run.timestamps))
        selection['calibration_population'] = 'Full preselected validation experiment, including native anomalies; labels not used to pick threshold; nominal tail quantile is not normal-only FPR'
    elif spec['loader'] == 'pronto_whole_day':
        paths = {Path(p).stem: p for p in config['paths']}
        for role, key in [('train', 'training_day'), ('validation', 'validation_day'), ('test', 'test_day')]:
            path = paths[spec[key]]
            run = load_pronto_merged_csv(root / path, alarm_column_count=config['alarm_column_count'], label_column=config['label_column'])
            normal_intervals = contiguous_normal_intervals(run.labels)
            intervals = [(0, len(run.labels))] if role == 'test' else normal_intervals
            for start, stop in intervals:
                roles[role].append(group(path, run.run_id, run.process_values, run.labels != 'Normal', run.process_names,
                                         np.arange(len(run.labels)) * config['sample_period_seconds'], start, stop, run.labels))
            selection[role] = {'day': run.run_id, 'raw_points': len(run.labels), 'normal_intervals': normal_intervals,
                               'selected_points': sum(b-a for a, b in intervals), 'unused_points': len(run.labels)-sum(b-a for a, b in intervals)}
        selection['feature_rule'] = 'All 17 identically ordered process channels; first12 native alarm channels excluded; Fault is label only'
        selection['normal_selection'] = 'Uses training/validation day native fault labels for normal-only population; test day labels are evaluation only'
    else:
        raise ValueError('Unreviewed industrial loader')
    if len(roles['test']) != spec['expected_test_entities']:
        raise ValueError('Incomplete held-out entity roster')
    names = {tuple(g['feature_names']) for groups in roles.values() for g in groups}
    if len(names) != 1 or len(next(iter(names))) != config.get('features', 17):
        raise ValueError('Industrial feature names/order/count differ')
    return roles, selection


def prepare(root, settings):
    configurations = {s['source_config']: json.loads((root / s['source_config']).read_text(encoding='utf-8')) for s in settings['datasets']}
    loader_paths = ['scripts/flow_matching/prepare_industrial_tsad.py', 'src/iia_benchmark/data/tep.py', 'src/iia_benchmark/data/skab.py', 'src/iia_benchmark/data/pronto.py', 'src/iia_benchmark/data/schema.py']
    identity = {'settings': settings, 'source_configs': configurations, 'preprocessing_sources': {p: sha(root/p) for p in loader_paths}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = root / settings['prepared_root']
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['preparation_sha256'] != fingerprint:
            raise ValueError('Existing industrial inputs belong to different preparation rules')
        for dataset in previous['datasets']:
            for source in dataset['source_files']:
                if sha(root/source['path']) != source['sha256']:
                    raise ValueError('Prepared industrial raw source changed')
            if sha(root/dataset['input_npz']) != dataset['input_sha256']:
                raise ValueError('Prepared industrial arrays changed')
        return previous
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in settings['datasets']:
        roles, selection = load_groups(root, spec, configurations[spec['source_config']])
        source_paths = sorted({g['source_path'] for groups in roles.values() for g in groups})
        sources = [{'path': p, 'sha256': sha(root/p), 'bytes': (root/p).stat().st_size} for p in source_paths]
        normalized, scaler = normalize_groups(roles)
        payload, metadata = {}, {}
        for role, groups in roles.items():
            metadata[role] = []
            for index, (g, values) in enumerate(zip(groups, normalized[role], strict=True)):
                prefix = ('e' if role == 'test' else 't' if role == 'train' else 'v') + f'{index:03d}'
                payload[prefix + ('_test' if role == 'test' else '_values')] = values
                payload[prefix + '_labels'] = g['labels']
                payload[prefix + '_timestamps'] = g['timestamps']
                payload[prefix + '_source_rows'] = np.arange(*g['source_interval'])
                if g['fault_labels'] is not None:
                    payload[prefix + '_fault_labels'] = g['fault_labels']
                metadata[role].append({key: g[key] for key in ('group_id', 'entity_id', 'source_path', 'source_interval', 'feature_names')})
                metadata[role][-1].update(array_prefix=prefix, points=len(values), features=values.shape[1], anomaly_points=int(g['labels'].sum()))
                if role == 'test':
                    metadata[role][-1]['test_points'] = len(values)
        target = output / (spec['id'] + '.npz')
        if target.exists():
            with np.load(target, allow_pickle=False) as existing:
                if set(existing.files) != set(payload) or any(not np.array_equal(existing[k], v) for k, v in payload.items()):
                    raise FileExistsError('Different partial industrial arrays preserved')
        else:
            np.savez_compressed(target, **payload)
        if any(sha(root/s['path']) != s['sha256'] for s in sources):
            raise ValueError('Raw source changed during preparation')
        records.append({'id': spec['id'], 'input_npz': target.relative_to(root).as_posix(), 'input_sha256': sha(target),
                        'source_config': spec['source_config'], 'source_files': sources, 'training_groups': metadata['train'],
                        'validation_groups': metadata['validation'], 'entities': metadata['test'], 'scaler': scaler,
                        'selection_audit': selection, 'grouped_splits_verified': True,
                        'source_track': spec['source_track'], 'generalization': 'Whole run/day groups disjoint; PRONTO normal selection label-assisted; no unseen-fault-type or author-equivalence claim'})
        print(json.dumps({'prepared_dataset': spec['id'], 'training_points': scaler['fit_points'], 'test_entities': len(metadata['test']), 'test_points': sum(g['points'] for g in metadata['test'])}), flush=True)
    manifest = {'preparation_sha256': fingerprint, 'recipe': settings['recipe'], 'datasets': records,
                'source_configuration_hashes': {p: sha(root/p) for p in configurations}, 'boundary': settings['boundary']}
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/fm_industrial_tsad_execution.v1.json')
    args = parser.parse_args()
    prepare(ROOT, json.loads(args.config.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
