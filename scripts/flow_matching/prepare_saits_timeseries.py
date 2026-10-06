"""Prepare SAITS original time-series datasets using pinned author scripts."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.audit_author_code import working_copy
from scripts.flow_matching.prepare_saits_physio import sha


def stage_dataset(spec, staging_root):
    source = ROOT / spec['source']
    directory = staging_root / spec['id']
    marker = directory.with_suffix('.manifest.json')
    digest = sha(source)
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        if digest != metadata['source_sha256']:
            raise ValueError('Source changed; staged data preserved')
        for name, expected in metadata['files'].items():
            if sha(directory / name) != expected:
                raise ValueError('Staged raw file changed')
        return directory, metadata
    if directory.exists():
        raise ValueError('Unregistered staging directory exists; preserve it')
    directory.mkdir(parents=True)
    names = []
    if 'nested_zip' in spec:
        with zipfile.ZipFile(source) as outer:
            nested_bytes = outer.read(spec['nested_zip'])
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
            for entry in inner.infolist():
                if entry.is_dir():
                    continue
                name = Path(entry.filename).name
                if not name.startswith('PRSA_Data_') or not name.endswith('.csv') or name in names:
                    raise ValueError('Unexpected or duplicate station entry')
                with inner.open(entry) as src, (directory / name).open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                names.append(name)
        if len(names) != spec['stations']:
            raise ValueError('Station count mismatch')
        # Author concatenates by row, so independently verify every station clock.
        import pandas as pd
        import numpy as np
        baseline = None
        for name in sorted(names):
            frame = pd.read_csv(directory / name, usecols=['year', 'month', 'day', 'hour'])
            times = pd.to_datetime(frame).to_numpy()
            if baseline is None:
                baseline = times
            if not np.array_equal(times, baseline) or np.any(np.diff(times) != np.timedelta64(1, 'h')):
                raise ValueError('Station row timestamps do not align hourly')
    elif 'member' in spec:
        name = Path(spec['member']).name
        with zipfile.ZipFile(source) as archive, archive.open(spec['member']) as src, (directory / name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        names.append(name)
    else:
        name = source.name
        with source.open('rb') as src, (directory / name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        names.append(name)
    metadata = {'source': spec['source'], 'source_sha256': digest, 'files': {name: sha(directory / name) for name in names}}
    marker.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return directory, metadata


def audit_output(output, spec, source, ratio, config_sha, corrected=False, exact_masks=False):
    import h5py
    import numpy as np
    report = {'status': 'prepared_audited_not_score_verified', 'dataset': spec['id'], 'nominal_missing_rate': ratio,
              'config_sha256': config_sha, 'raw_source': source, 'data_citation': spec['data_citation'], 'splits': {},
              'window_protocol': 'corrected_complete_windows' if corrected else 'author_legacy_double_stride',
              'mask_protocol': 'exact_without_replacement' if exact_masks else 'author_with_replacement'}
    with h5py.File(output / 'datasets.h5', 'r') as handle, np.load(output / 'audit_arrays.npz') as arrays:
        months = {key: set(arrays[key + '_fit_months'].tolist()) for key in ('train', 'val', 'test')}
        if any(months[a] & months[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('Calendar split overlaps')
        if len(months['test']) != spec['test_months'] or len(months['val']) != spec['val_months']:
            raise ValueError('Calendar split length mismatch')
        if max(months['test']) >= min(months['val']) or max(months['val']) >= min(months['train']):
            raise ValueError('Author calendar split direction mismatch')
        report['feature_names'] = arrays['feature_names'].tolist()
        if len(report['feature_names']) != spec['features']:
            raise ValueError('Feature count mismatch')
        for split in months:
            x = handle[split]['X'][:]
            count = len(arrays[split + '_ids'])
            if x.shape != (count, spec['seq_len'], spec['features']) or np.isinf(x).any():
                raise ValueError('Invalid author data shape or infinite values')
            starts, ends = arrays[split + '_start'], arrays[split + '_end']
            if not np.all(starts <= ends) or (count > 1 and not np.all(starts[1:] > starts[:-1])):
                raise ValueError('Window chronology invalid')
            entry = {'shape': list(x.shape), 'windows': count, 'months': sorted(months[split]),
                     'first_timestamp': str(starts[0]), 'last_timestamp': str(ends[-1]),
                     'missing_fraction_in_serialized_X': float(np.isnan(x).mean())}
            if split != 'train':
                observed = np.isfinite(x)
                target = handle[split]['indicating_mask'][:].astype(bool)
                condition = handle[split]['missing_mask'][:].astype(bool)
                if np.any(target & ~observed) or np.any(target & condition) or not np.array_equal(condition | target, observed):
                    raise ValueError('Evaluation mask audit failed')
                entry.update(targets=int(target.sum()), realized_missing_rate=float(target.sum() / observed.sum()))
                if exact_masks and target.sum() != int(observed.sum() * ratio):
                    raise ValueError('Exact mask does not match registered target count')
            report['splits'][split] = entry
        total = sum(record['windows'] for record in report['splits'].values())
        report.update(total_samples=total, paper_reported_samples=spec['expected_samples'],
                      paper_sample_count_matches=total == spec['expected_samples'])
    mask_boundary = 'Masks sampled without replacement to realize the nominal rate (rounded down by one coordinate).' if exact_masks else 'Author replacement masks retained; realized missing fraction differs from the nominal rate.'
    report.update(dataset_sha256=sha(output / 'datasets.h5'), audit_arrays_sha256=sha(output / 'audit_arrays.npz'),
                  protocol_boundaries=['Preprocessing NumPy seed 26 registered locally; historical author seed unspecified.',
                    mask_boundary,
                    'Author chronological month split retained: earlier test, intermediate validation, later training.',
                    'Month bootstrap is a temporal block sensitivity estimate; temporal independence is not guaranteed.'])
    if spec['id'] == 'air_quality_132':
        report['protocol_boundaries'].append('Station filename order fixed lexically; historical order unspecified. This 132-feature dataset is distinct from Air-36.')
    if spec['id'] == 'ett':
        report['protocol_boundaries'].append('Author stride-12 length-24 windows retained: repeated timestamps count multiple times in original metrics; month blocks have boundary dependence.')
    if corrected:
        report['protocol_boundaries'].append('Corrected full-window enumeration is a separate protocol; author-original data remain preserved.')
    elif not report['paper_sample_count_matches']:
        report['protocol_boundaries'].append(f'Released author window_truncate multiplies the last start by stride twice and drops the final window. Local count {total} differs from paper {spec["expected_samples"]}; protocol alignment remains unverified.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data/fm_saits_timeseries_original.v1.json')
    parser.add_argument('--dataset', choices=['air_quality_132', 'electricity', 'ett'])
    parser.add_argument('--all_missing_rates', action='store_true')
    parser.add_argument('--correct_windowing', action='store_true')
    parser.add_argument('--exact_masks', action='store_true')
    args = parser.parse_args()
    if args.exact_masks and not args.correct_windowing:
        raise ValueError('Exact-mask experiment uses the jointly registered window correction')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['seed'] != 26:
        raise ValueError('Registered source compatibility uses seed 26')
    working_copy('saits')
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/saits/corrected'
    environment = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for spec in config['datasets']:
        if args.dataset and args.dataset != spec['id']:
            continue
        directory, raw = stage_dataset(spec, ROOT / config['staging_root'])
        for ratio in spec['missing_ratios'] if args.all_missing_rates else spec['missing_ratios'][:1]:
            suffix = '_corrected_windows' if args.correct_windowing else ''
            if args.exact_masks:
                suffix += '_exact_masks'
            output = ROOT / config['processed_root'] / f'{spec["id"]}{suffix}_missing{ratio:g}'
            if output.exists():
                audit_path = output / 'data_audit.json'
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text(encoding='utf-8'))
                    if audit['config_sha256'] != sha(args.config) or audit['dataset_sha256'] != sha(output / 'datasets.h5'):
                        raise ValueError('Existing output changed; preserve it')
                else:
                    # A completed author preprocessing run may have stopped at
                    # postprocessing audit. Validate its existing arrays without
                    # rewriting H5 or raw data; reject any incomplete output.
                    if not all((output / name).exists() for name in ('datasets.h5', 'audit_arrays.npz', 'scaler.pkl')):
                        raise ValueError('Incomplete unregistered output exists; preserve it')
                    audit = audit_output(output, spec, raw, ratio, sha(args.config), args.correct_windowing, args.exact_masks)
                    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
                continue
            input_path = directory if spec['id'] == 'air_quality_132' else directory / next(iter(raw['files']))
            command = [sys.executable, str(source / 'dataset_generating_scripts' / spec['script']),
                       '--file_path', str(input_path), '--seq_len', str(spec['seq_len']),
                       '--artificial_missing_rate', str(ratio), '--saving_path', str(output.parent), '--dataset_name', output.name]
            if spec['stride'] != spec['seq_len']:
                command += ['--sliding_len', str(spec['stride'])]
            if args.correct_windowing:
                command = [sys.executable, str(ROOT / 'scripts/flow_matching/run_saits_preprocessor.py'),
                           '--script', spec['script']] + command[2:]
                if args.exact_masks:
                    command.insert(4, '--exact_masks')
            subprocess.run(command, cwd=source, env=environment, check=True)
            audit = audit_output(output, spec, raw, ratio, sha(args.config), args.correct_windowing, args.exact_masks)
            (output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
            print(json.dumps({'dataset': spec['id'], 'ratio': ratio, 'splits': audit['splits']}), flush=True)


if __name__ == '__main__':
    main()
