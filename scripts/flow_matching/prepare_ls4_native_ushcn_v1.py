"""Audit the complete author CSV and execute its unchanged five-fold recipe."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_frame(frame):
    import numpy as np
    names = ['ID', 'Time'] + [f'Value_{i}' for i in range(5)] + [f'Mask_{i}' for i in range(5)]
    if list(frame.columns) != names or not len(frame) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Full finite five-variable author CSV required')
    ids = np.sort(frame.ID.unique())
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError('Author split generator assumes contiguous original IDs')
    masks = frame[[f'Mask_{i}' for i in range(5)]].to_numpy()
    values = frame[[f'Value_{i}' for i in range(5)]].to_numpy()
    if not np.isin(masks, [0, 1]).all() or not (masks.sum(axis=1) > 0).all():
        raise ValueError('Every author row must contain binary observation masks')
    if np.any(values[masks == 0] != 0) or frame.duplicated(['ID', 'Time']).any():
        raise ValueError('Original missing-value placeholders or row identity differ')
    if frame.Time.min() < 0 or frame.Time.max() > 200:
        raise ValueError('Original small climate time range differs')
    before = set(frame.loc[frame.Time <= 150, 'ID'])
    after = frame.loc[frame.Time > 150].groupby('ID').size()
    if before != set(ids) or set(after.index) != set(ids) or after.min() < 4:
        raise ValueError('All original validation-eligible climate series required')
    return dict(rows=len(frame), series=len(ids), variables=5,
                time_min=float(frame.Time.min()), time_max=float(frame.Time.max()),
                observed_values=int(masks.sum()),
                observations_per_channel=masks.sum(axis=0).astype(int).tolist(),
                minimum_rows_after_150=int(after.min()),
                missing_fraction=float(1-masks.mean()))


def audit_partition(parts, count):
    import numpy as np
    arrays = [np.asarray(parts[k]) for k in ('train', 'val', 'test')]
    if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
        raise ValueError('Integer original series split IDs required')
    flat = np.concatenate(arrays)
    if not np.array_equal(np.sort(flat), np.arange(count)):
        raise ValueError('Every series must occur exactly once in the grouped split')
    if len(arrays[2]) != int(np.ceil(count*.1)) or len(arrays[1]) != int(np.ceil((count-len(arrays[2]))*.2)):
        raise ValueError('Original 10-percent test and 20-percent remainder validation required')
    return {k: len(parts[k]) for k in ('train', 'val', 'test')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Pinned preparation input changed')
    lock = json.loads((ROOT / config['environment_lock']).read_text(encoding='utf-8'))
    if sys.version != lock['python'] or any(importlib.metadata.version(k) != v for k, v in lock['packages'].items()):
        raise ValueError('Original locked native LS4 environment required')
    import numpy as np
    import pandas as pd
    csv_path = ROOT / config['csv_path']
    folder = ROOT / config['prepared_directory']
    report = ROOT / config['report_directory']
    if folder.exists() or report.exists():
        raise FileExistsError('Preserve raw data and every prior prepared split')
    frame = pd.read_csv(csv_path)
    data_audit = audit_frame(frame)
    del frame
    folder.mkdir(parents=True)
    report.mkdir(parents=True)
    generator = ROOT / config['original_fold_generator']
    # Bind only the original script's machine-specific CSV path; execute all original code.
    original_read = pd.read_csv
    seen = []
    def read_bound(path, *a, **kw):
        if str(path) != '../processed_csv/small_chunked_sporadic.csv':
            raise ValueError('Unexpected upstream generator data access')
        seen.append(str(path))
        return original_read(csv_path, *a, **kw)
    old_cwd = Path.cwd()
    pd.read_csv = read_bound
    try:
        os.chdir(folder)
        exec(compile(generator.read_bytes(), str(generator), 'exec'), {'__name__': '__main__'})
    finally:
        os.chdir(old_cwd)
        pd.read_csv = original_read
    if len(seen) != 1:
        raise ValueError('Exactly one full original CSV read required')
    folds = []
    for i in range(5):
        directory = folder / f'small_chunk_fold_idx_{i}'
        parts = {k: np.load(directory / f'{k}_idx.npy', allow_pickle=False) for k in ('train', 'val', 'test')}
        counts = audit_partition(parts, data_audit['series'])
        folds.append(dict(fold=i, counts=counts, files=[dict(path=str(p), sha256=sha(p)) for p in sorted(directory.glob('*.npy'))]))
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=args.config,
                   configuration_sha256=sha(ROOT / args.config), csv_sha256=sha(csv_path),
                   original_generator_sha256=sha(generator), generator_seed=432,
                   unchanged_upstream_recipe_executed=True, data_audit=data_audit, folds=folds,
                   original_historical_fold_arrays_available=False,
                   historical_author_split_equivalence_certified=False,
                   raw_csv_after_sha256=sha(csv_path), boundary=config['boundary'])
    (report / 'preparation_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(data=data_audit, folds=[f['counts'] for f in folds])), flush=True)


if __name__ == '__main__':
    main()
