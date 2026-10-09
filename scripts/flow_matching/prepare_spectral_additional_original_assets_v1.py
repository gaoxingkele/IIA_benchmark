"""Preserve and audit released Table2/3 data; this does not generate scores."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit_asset(path):
    if path.suffix == '.npy':
        with path.open('rb') as stream:
            if stream.read(6) != b'\x93NUMPY':
                raise ValueError('Not an actual NPY array')
            version = tuple(stream.read(2))
            size = struct.unpack('<H' if version == (1, 0) else '<I',
                                 stream.read(2 if version == (1, 0) else 4))[0]
            header = ast.literal_eval(stream.read(size).decode('latin1').strip())
        if 'O' in header['descr']:
            raise ValueError('Object data is not an accepted numeric source')
        return {'format': 'npy', 'shape': list(header['shape']), 'dtype': header['descr'],
                'boundary': 'Header and byte hash audited; finite-value and author-loader checks remain required.'}
    if path.suffix == '.tsf':
        lengths = []; missing = 0; metadata = {}; values = 0
        active = False
        with path.open(encoding='cp1252') as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('@'):
                    if line == '@data':
                        active = True
                    else:
                        key, _, value = line[1:].partition(' ')
                        if key != 'attribute':
                            metadata[key] = value
                elif active:
                    series = line.rsplit(':', 1)[1].split(',')
                    lengths.append(len(series))
                    for value in series:
                        if value == '?':
                            missing += 1
                        elif not math.isfinite(float(value)):
                            raise ValueError('Nonfinite released TSF source')
                    values += len(series)
        if not lengths:
            raise ValueError('Missing actual TSF data')
        return {'format': 'tsf', 'series_count': len(lengths), 'lengths': sorted(set(lengths)),
                'value_count': values, 'missing_values': missing, 'metadata': metadata,
                'boundary': 'Raw full-series availability only. Author per-series normalization and80/20 split must be audited before executing Table3.'}
    return {'format': path.suffix.lstrip('.'),
            'boundary': 'Raw bytes audited; author-loader identity still required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_path = ROOT / config['manifest']
    if manifest_path.exists():
        raise FileExistsError('Preserve immutable asset audit')
    if sha(ROOT / config['paper']['path']) != config['paper']['sha256']:
        raise ValueError('Primary paper changed')
    records = []
    for item in config['assets']:
        source = ROOT / item['source']; target = ROOT / item['destination']
        if sha(source) != item['sha256']:
            raise ValueError('Released original source differs')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != item['sha256']:
                raise ValueError('Never overwrite existing raw data')
        else:
            shutil.copy2(source, target)
        if sha(target) != item['sha256']:
            raise ValueError('Copy integrity failed')
        records.append(dict(item, byte_count=target.stat().st_size, audit=audit_asset(target),
                            resolved_destination=str(target.resolve())))
    manifest = {'captured_utc': datetime.now(timezone.utc).isoformat(),
        'config': args.config, 'config_sha256': sha(config_path), 'paper': config['paper'],
        'assets': records, 'original_raw_files_preserved': True,
        'all_checksums_verified': True, 'formal_training_started': False,
        'benchmark_performance_available': False}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'assets': len(records), 'all_checksums_verified': True,
        'TSF': [dict(id=r['id'], **r['audit']) for r in records if r['audit']['format'] == 'tsf']}))


if __name__ == '__main__':
    main()
