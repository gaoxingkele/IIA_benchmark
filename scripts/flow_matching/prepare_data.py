"""Non-destructive real-data preparation with frozen groups, masks and audits."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTES = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS', 'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP', 'Creatinine', 'ALP']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def reject_pointer(path):
    with Path(path).open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Git LFS pointer is not data: {path}')


def parse_patient(payload: bytes):
    """Exactly the authors' last measurement per hour/feature, without 48 scans."""
    frame = pd.read_csv(io.BytesIO(payload))
    frame['hour'] = frame['Time'].str.split(':').str[0].astype(int)
    frame = frame[frame['hour'].between(0, 47) & frame['Parameter'].isin(ATTRIBUTES)]
    frame = frame.drop_duplicates(['hour', 'Parameter'], keep='last')
    array = np.full((48, 35), np.nan, dtype=np.float64)
    mapping = {name: index for index, name in enumerate(ATTRIBUTES)}
    array[frame['hour'].to_numpy(), frame['Parameter'].map(mapping).to_numpy()] = frame['Value'].to_numpy()
    mask = np.isfinite(array)
    return np.nan_to_num(array), mask


def physio_split(count, seed=1, fold=0, generator='legacy'):
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    factory = np.random.RandomState if generator == 'legacy' else np.random.default_rng
    indices = np.arange(count)
    factory(seed).shuffle(indices)
    start, end = int(fold * .2 * count), int((fold + 1) * .2 * count)
    test = indices[start:end]
    remaining = np.delete(indices, np.arange(start, end))
    factory(seed).shuffle(remaining)
    train_end = int(count * .7)
    return {'train': remaining[:train_end], 'val': remaining[train_end:], 'test': test}


def normalize(values, observed, fit_indices):
    selected = values[fit_indices].reshape(-1, values.shape[-1])
    mask = observed[fit_indices].reshape(selected.shape)
    means, scales = [], []
    for index in range(values.shape[-1]):
        column = selected[:, index][mask[:, index]]
        if not len(column):
            raise ValueError(f'Feature {index} has no training observations')
        means.append(column.mean())
        scales.append(max(column.std(), 1e-8))
    means, scales = np.array(means), np.array(scales)
    safe_values = np.where(observed, values, 0.)
    normalized = ((safe_values - means) / scales) * observed
    if not np.isfinite(normalized).all():
        raise ValueError('Non-finite normalized data')
    return normalized.astype(np.float32), means, scales


def point_mask(observed, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError('ratio must lie in (0,1)')
    rng = np.random.RandomState(seed)
    conditioning = observed.copy()
    for item in conditioning:
        flat = item.reshape(-1)
        candidates = np.flatnonzero(flat)
        flat[rng.choice(candidates, int(len(candidates) * ratio), replace=False)] = False
    return conditioning


def prepare_physio(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    archive_path = ROOT / config['raw_archive']
    reject_pointer(archive_path)
    output = ROOT / config['output_root']
    output.mkdir(parents=True, exist_ok=True)
    raw_cache = output / 'last_hourly_raw.npz'
    if not raw_cache.exists():
        values, masks, patients = [], [], []
        with tarfile.open(archive_path) as archive:
            members = sorted((m for m in archive.getmembers() if m.isfile() and re.fullmatch(r'set-a/\d{6}\.txt', m.name)), key=lambda m: m.name)
            if len(members) != config['expected_patients']:
                raise ValueError('Unexpected patient count in raw archive')
            for member in members:
                value, mask = parse_patient(archive.extractfile(member).read())
                values.append(value)
                masks.append(mask)
                patients.append(Path(member.name).stem)
        np.savez_compressed(raw_cache, values=np.array(values), observed=np.array(masks), patients=np.array(patients))
    with np.load(raw_cache, allow_pickle=False) as raw:
        values, observed, patients = raw['values'], raw['observed'], raw['patients']
    manifest = {'dataset': config['id'], 'raw_sha256': digest(archive_path), 'raw_path': config['raw_archive'],
                'patient_count': len(patients), 'shape': list(values.shape), 'native_missing_fraction': float(1 - observed.mean()),
                'feature_order': ATTRIBUTES, 'files': [], 'raw_preserved': True}
    for protocol in config['protocols']:
        for fold in config['folds']:
            split = physio_split(len(values), seed=config['split_seed'], fold=fold, generator=protocol['split_generator'])
            fit = np.arange(len(values)) if protocol['normalization'] == 'author_global' else split['train']
            normalized, mean, std = normalize(values, observed, fit)
            for ratio in config['missing_ratios']:
                conditioning = point_mask(observed, ratio, config['mask_seed'])
                path = output / f"{protocol['id']}_fold{fold}_missing{ratio:g}.npz"
                if not path.exists():
                    np.savez_compressed(path, values=normalized, observed=observed, conditioning=conditioning,
                                        eval_mask=observed & ~conditioning, patients=patients,
                                        mean=mean, std=std, **split)
                with np.load(path, allow_pickle=False) as data:
                    assert np.array_equal(data['eval_mask'], data['observed'] & ~data['conditioning'])
                    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
                        if set(data[a]) & set(data[b]):
                            raise ValueError('Patient overlap across splits')
                    if not np.array_equal(data['values'], normalized) or not np.array_equal(data['conditioning'], conditioning):
                        raise ValueError('Existing processed file disagrees with frozen preparation; not overwritten')
                manifest['files'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                    'protocol': protocol, 'fold': fold, 'missing_ratio': ratio,
                    'split_sizes': {name: len(idx) for name, idx in split.items()},
                    'split_hashes': {name: hashlib.sha256(idx.tobytes()).hexdigest() for name, idx in split.items()}})
    (output / 'audit.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/fm_physionet2012.json')
    print(json.dumps(prepare_physio(parser.parse_args().config)))
