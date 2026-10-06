"""Prepare configured industrial CSV files; split whole experiments before windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.flow_matching.prepare_data import normalize, digest
from scripts.flow_matching.prepare_transfer import group_split, missing_mask

ROOT = Path(__file__).resolve().parents[2]


def prepare(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    source = json.loads((ROOT / cfg['source_config']).read_text(encoding='utf-8'))
    if 'paths' in source:
        paths = [ROOT / path for path in source['paths']]
    else:
        paths = [ROOT / source['train_path']]
        for pattern in source['test_globs']:
            paths.extend(sorted(ROOT.glob(pattern)))
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError('Registered industrial CSV is missing')
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=True)
    arrays, groups, provenance = [], [], []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path, sep=cfg['separator'])
        missing_columns = set(cfg['features']) - set(frame)
        if missing_columns:
            raise ValueError(f'Unexpected schema in {path}: {missing_columns}')
        values = frame[cfg['features']].to_numpy(dtype=np.float64)
        starts = list(range(0, len(values) - cfg['window_length'] + 1, cfg['window_stride']))
        if not starts:
            raise ValueError('An experiment is too short for registered windows')
        for start in starts:
            arrays.append(values[start:start + cfg['window_length']])
            groups.append(index)
        provenance.append({'path': path.relative_to(ROOT).as_posix(), 'raw_sha256': digest(path),
                           'group_id': index, 'rows': len(values), 'windows': len(starts),
                           'excluded_columns': sorted(set(frame) - set(cfg['features']))})
    values, groups = np.array(arrays), np.array(groups)
    observed = np.isfinite(values)
    if cfg['split_policy'] == 'ordered_whole_files':
        if len(paths) != len(cfg['ordered_split_names']):
            raise ValueError('Ordered day split differs from registered file count')
        split = {name: np.flatnonzero(groups == index) for index, name in enumerate(cfg['ordered_split_names'])}
    else:
        split = group_split(groups, cfg['split_seed'])
    normalized, mean, std = normalize(values, observed, split['train'])
    files = []
    for seed in cfg['mask_seeds']:
        for pattern in cfg['missing_patterns']:
            for ratio in cfg['missing_ratios']:
                conditioning = missing_mask(observed, ratio, seed, pattern)
                path = output / f'{pattern}_missing{ratio:g}_seed{seed}.npz'
                if not path.exists():
                    np.savez_compressed(path, values=normalized, observed=observed, conditioning=conditioning,
                                        eval_mask=observed & ~conditioning, groups=groups, mean=mean, std=std, **split)
                else:
                    with np.load(path, allow_pickle=False) as prior:
                        if not np.array_equal(prior['values'], normalized) or not np.array_equal(prior['conditioning'], conditioning):
                            raise ValueError('Existing processed data differs and was preserved')
                files.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'pattern': pattern,
                              'requested_ratio': ratio, 'realized_ratio': float((observed & ~conditioning).sum() / observed.sum()), 'seed': seed})
    audit = {'dataset': cfg['id'], 'configuration': cfg, 'shape': list(values.shape), 'source_files': provenance,
             'native_missing_fraction': float(1 - observed.mean()),
             'split_sizes': {name: len(ids) for name, ids in split.items()},
             'split_groups': {name: sorted(set(groups[ids].tolist())) for name, ids in split.items()},
             'training_scale_floor_features': [cfg['features'][index] for index in np.flatnonzero(std == 1e-8)],
             'files': files, 'raw_preserved': True}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'dataset': cfg['id'], 'shape': audit['shape'], 'files': len(files), 'split_sizes': audit['split_sizes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    prepare(parser.parse_args().config)
