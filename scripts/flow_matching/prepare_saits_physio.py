"""Run the pinned SAITS author preprocessor on preserved PhysioNet a/b/c."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import working_copy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_patients(archives, destination, expected):
    """Flatten only regular numeric patient files; never extract arbitrary paths."""
    destination = Path(destination)
    if destination.exists():
        marker = destination.with_suffix('.manifest.json')
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest['archives'] != {str(p.relative_to(ROOT)): sha(p) for p in archives}:
            raise ValueError('Staged source archive identity changed')
        for name, digest in manifest['patients'].items():
            if sha(destination / name) != digest:
                raise ValueError('Staged patient changed')
        return manifest
    destination.mkdir(parents=True)
    patients = {}
    for archive in archives:
        with tarfile.open(archive, 'r:gz') as handle:
            for member in handle:
                if member.isdir():
                    continue
                name = Path(member.name).name
                if not member.isfile() or not name.endswith('.txt') or not name[:-4].isdigit():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                if name in patients:
                    raise ValueError(f'Duplicate patient: {name}')
                payload = handle.extractfile(member).read()
                with (destination / name).open('xb') as output:
                    output.write(payload)
                patients[name] = hashlib.sha256(payload).hexdigest()
    if len(patients) != expected:
        raise ValueError('Raw patient count mismatch')
    # The author requires a directory containing exclusively numeric patient .txt
    # files. Keep the manifest adjacent, rather than in that input directory.
    manifest = {'archives': {str(p.relative_to(ROOT)): sha(p) for p in archives}, 'patients': patients}
    return manifest


def audit_dataset(directory, config, manifest, config_sha):
    import h5py
    import numpy as np
    with np.load(directory / 'audit_arrays.npz') as arrays, h5py.File(directory / 'datasets.h5', 'r') as handle:
        ids = {split: arrays[split + '_ids'].astype(int) for split in ('train', 'val', 'test')}
        sets = {key: set(value.tolist()) for key, value in ids.items()}
        if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('Patient split overlap')
        raw_ids = {int(name[:-4]) for name in manifest['patients']}
        dropped = raw_ids - set.union(*sets.values())
        if dropped != set(config['expected_dropped_ids']):
            raise ValueError(f'Dropped patient IDs mismatch: {sorted(dropped)}')
        if sum(map(len, sets.values())) != config['expected_valid_patients']:
            raise ValueError('Valid patient count mismatch')
        report = {'status': 'prepared_audited_not_score_verified', 'config_sha256': config_sha,
                  'raw_archives': manifest['archives'], 'dropped_ids': sorted(dropped),
                  'feature_names': arrays['feature_names'].tolist(), 'splits': {}}
        for split in ids:
            x = handle[split]['X'][:]
            if x.shape != (len(ids[split]), config['sequence_length'], config['expected_features']):
                raise ValueError('Author output shape mismatch')
            observed = np.isfinite(x)
            if np.isinf(x).any():
                raise ValueError('Infinite normalized data')
            entry = {'shape': list(x.shape), 'patients': len(ids[split]), 'native_missing_fraction': float((~observed).mean())}
            if split != 'train':
                indication = handle[split]['indicating_mask'][:].astype(bool)
                conditioning = handle[split]['missing_mask'][:].astype(bool)
                if np.any(indication & ~observed) or not np.array_equal(conditioning | indication, observed) or np.any(conditioning & indication):
                    raise ValueError('Invalid held-out target mask')
                entry.update(targets=int(indication.sum()), realized_missing_rate=float(indication.sum() / observed.sum()))
            report['splits'][split] = entry
        # StandardScaler train-only centering/scaling independently checked from
        # serialized H5, allowing float32 output rounding and constant features.
        train = handle['train']['X'][:].astype('float64')
        mean, variance = np.nanmean(train, axis=(0, 1)), np.nanvar(train, axis=(0, 1))
        nonconstant = variance > 1e-8
        if np.max(np.abs(mean)) > 1e-5 or np.max(np.abs(variance[nonconstant] - 1)) > 1e-4:
            raise ValueError('Train-only standardized data audit failed')
    report.update(dataset_sha256=sha(directory / 'datasets.h5'), audit_arrays_sha256=sha(directory / 'audit_arrays.npz'),
                  protocol_boundaries=['Historical source directory ordering not published; frozen lexical order used.',
                      'Author masking draws with replacement: realized validation/test rate is below nominal 10%.',
                      '37-feature all-patient hourly-mean data is distinct from CSDI/CFMI 35-feature set-a.'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data/fm_saits_physio_original.v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['seed'] != 26:
        raise ValueError('Pinned author preprocessor uses fixed seed 26')
    staging = ROOT / config['raw_staging']
    marker = staging.with_suffix('.manifest.json')
    archives = [ROOT / name for name in config['source_archives']]
    if staging.exists():
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest['archives'] != {str(p.relative_to(ROOT)): sha(p) for p in archives}:
            raise ValueError('Staged archives changed')
        for name, digest in manifest['patients'].items():
            if sha(staging / name) != digest:
                raise ValueError('Staged patient identity changed')
    else:
        manifest = extract_patients(archives, staging, config['expected_raw_patients'])
        marker.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    output = ROOT / config['output_root']
    config_sha = sha(args.config)
    if output.exists():
        existing = json.loads((output / 'data_audit.json').read_text(encoding='utf-8'))
        if existing['config_sha256'] != config_sha or existing['dataset_sha256'] != sha(output / 'datasets.h5'):
            raise ValueError('Existing processed data differs from its frozen audit; preserve it')
        print(json.dumps(existing))
        return
    working_copy('saits')
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/corrected'
    # Keep imports resolved to this author snapshot without changing global env.
    import os
    environment = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, str(source / 'dataset_generating_scripts/gene_PhysioNet2012_dataset.py'),
               '--raw_data_path', str(staging), '--outcome_files_dir', str(ROOT / config['outcomes_dir']),
               '--saving_path', str(output.parent), '--dataset_name', output.name,
               '--train_frac', str(config['train_frac']), '--val_frac', str(config['val_frac']),
               '--artificial_missing_rate', str(config['artificial_missing_rate'])]
    subprocess.run(command, cwd=source, env=environment, check=True)
    report = audit_dataset(output, config, manifest, config_sha)
    (output / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
