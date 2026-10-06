"""Prepare a registered industrial subset without copying/altering raw HDF5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.flow_matching.prepare_data import normalize, point_mask, digest

ROOT = Path(__file__).resolve().parents[2]


def group_split(groups, seed):
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    train_end, val_end = int(.7 * len(unique)), int(.8 * len(unique))
    return {name: np.flatnonzero(np.isin(groups, ids)) for name, ids in
            [('train', unique[:train_end]), ('val', unique[train_end:val_end]), ('test', unique[val_end:])]}


def missing_mask(observed, ratio, seed, pattern):
    if pattern == 'point':
        return point_mask(observed, ratio, seed)
    rng = np.random.default_rng(seed)
    result = observed.copy()
    for item in result:
        length, features = item.shape
        if pattern == 'block':
            width = max(1, int(round(length * ratio)))
            for feature in range(features):
                start = rng.integers(0, length - width + 1)
                item[start:start + width, feature] = False
        elif pattern == 'channel':
            count = max(1, int(round(features * ratio)))
            item[:, rng.choice(features, count, replace=False)] = False
        else:
            raise ValueError('Unknown missing pattern')
    return result


def prepare(config_path):
    settings = json.loads(Path(config_path).read_text(encoding='utf-8'))
    output = ROOT / settings['output_root']
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'registered_windows.npz'
    cases = []
    if not cache.exists():
        arrays, group_ids = [], []
        labels = None
        for mode in settings['modes']:
            path = ROOT / settings['raw_root'] / f'TEP_Mode{mode}.h5'
            with h5py.File(path, 'r') as source:
                file_labels = [label.decode() for label in source['Processdata_Labels'][:]]
                if labels is not None and labels != file_labels:
                    raise ValueError('Feature schema differs between modes')
                labels = file_labels
                for fault in settings['fault_ids']:
                    root = f"Mode{mode}/{settings['scenario']}/IDV{fault}/Mode{mode}_IDVInfo_{fault}_{settings['severity']}"
                    for run in range(settings['run_ids'][0], settings['run_ids'][1] + 1):
                        key = f'{root}/Run{run}/processdata'
                        if key not in source:
                            cases.append({'path': key, 'file': path.name, 'status': 'missing_or_stopped'})
                            continue
                        data = source[key]
                        if len(data) < settings['window_length']:
                            raise ValueError('Source episode is shorter than registered window')
                        starts = np.unique(np.linspace(0, len(data) - settings['window_length'], settings['windows_per_case'], dtype=int))
                        keep = [k for k in range(data.shape[1]) if k not in settings['exclude_columns']]
                        for start in starts:
                            arrays.append(data[start:start + settings['window_length'], :][:, keep])
                            group_ids.append(run)
                        cases.append({'path': key, 'file': path.name, 'status': 'selected', 'starts': starts.tolist()})
        values, groups = np.array(arrays), np.array(group_ids)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Invalid real industrial data')
        np.savez_compressed(cache, values=values, groups=groups,
                            labels=np.array([label for index, label in enumerate(labels) if index not in settings['exclude_columns']]))
        (output / 'selected_cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    with np.load(cache, allow_pickle=False) as data:
        values, groups = data['values'], data['groups']
    split = group_split(groups, settings['split_seed'])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(groups[split[a]]) & set(groups[split[b]]):
            raise ValueError('Simulation seed leaked across splits')
    observed = np.isfinite(values)
    normalized, mean, std = normalize(values, observed, split['train'])
    files = []
    for seed in settings['mask_seeds']:
        for pattern in settings['missing_patterns']:
            for ratio in settings['missing_ratios']:
                conditioning = missing_mask(observed, ratio, seed, pattern)
                path = output / f'{pattern}_missing{ratio:g}_seed{seed}.npz'
                if not path.exists():
                    np.savez_compressed(path, values=normalized, observed=observed, conditioning=conditioning,
                                        eval_mask=observed & ~conditioning, groups=groups, mean=mean, std=std, **split)
                else:
                    with np.load(path, allow_pickle=False) as prior:
                        if not np.array_equal(prior['values'], normalized) or not np.array_equal(prior['conditioning'], conditioning):
                            raise ValueError('Existing processed transfer data differs; preserved without overwrite')
                files.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                              'pattern': pattern, 'requested_ratio': ratio, 'realized_ratio': float((observed & ~conditioning).sum() / observed.sum()), 'seed': seed})
    report = {'dataset': settings['id'], 'configuration': settings, 'shape': list(normalized.shape),
              'split_sizes': {name: len(ids) for name, ids in split.items()},
              'split_groups': {name: sorted(set(groups[ids].tolist())) for name, ids in split.items()},
              'files': files, 'raw_preserved': True, 'boundary': settings['boundary']}
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'shape': report['shape'], 'files': len(files), 'split_sizes': report['split_sizes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/fm_tep_transfer_v1.json')
    prepare(parser.parse_args().config)
