"""Prepare exact released Air-36 windows while preserving raw/author files."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import shutil
import sys
import types
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_data import digest


def main():
    cfg = json.loads((ROOT / 'configs/datasets/fm_air36_original_v1.json').read_text(encoding='utf-8'))
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=True)
    staging = ROOT / cfg['cfmi_data_root'] / 'pm25'
    staging.mkdir(parents=True, exist_ok=True)
    raw = ROOT / cfg['raw_zip']
    archive = staging / 'STMVL-Release.zip'
    if not archive.exists():
        shutil.copy2(raw, archive)
    if digest(raw) != digest(archive):
        raise ValueError('Staged air data differs from raw source')
    extracted = staging / 'data'
    if not extracted.exists():
        with zipfile.ZipFile(archive) as payload:
            for member in payload.infolist():
                if not (extracted / member.filename).resolve().is_relative_to(extracted.resolve()):
                    raise ValueError('Unsafe archive member')
            if payload.testzip() is not None:
                raise ValueError('Corrupt ZIP')
            payload.extractall(extracted)
    sys.path.insert(0, str(ROOT / cfg['cfmi_source']))
    from imp_cfm.data.pm25 import load_dataset
    frame, missing_frame, mean, std = load_dataset(str(staging))
    if list(frame.shape) != cfg['expected_shape'] or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError('Unexpected Air-36 release')
    # Data-only relocation adapter: all author windowing/historical mask logic stays intact.
    source = ROOT / cfg['csdi_source'] / 'dataset_pm25.py'
    original = source.read_text(encoding='utf-8')
    adapted = original
    for relative in ['pm25_meanstd.pk', 'Code/STMVL/SampleData/pm25_ground.txt', 'Code/STMVL/SampleData/pm25_missing.txt']:
        old = '"./data/pm25/' + relative + '"'
        if adapted.count(old) != 1:
            raise ValueError('Author data path layout changed')
        adapted = adapted.replace(old, repr(str(extracted / relative)))
    (output / 'data_relocation.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True), fromfile='author/dataset_pm25.py', tofile='configured-data/dataset_pm25.py')), encoding='utf-8')
    module = types.ModuleType('air36_author_data')
    exec(compile(adapted, str(source), 'exec'), module.__dict__)
    files = []
    for fold in cfg['validation_indices']:
        values, observed, conditioning, historical, cuts = [], [], [], [], []
        splits, offset = {}, 0
        for split, mode in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
            dataset = module.PM25_Dataset(mode=mode, validindex=fold, eval_length=cfg['window_length'])
            for item in dataset:
                values.append(item['observed_data'])
                observed.append(item['observed_mask'])
                conditioning.append(item['gt_mask'])
                historical.append(item['hist_mask'])
                cuts.append(item['cut_length'])
            splits[split] = np.arange(offset, offset + len(dataset))
            offset += len(dataset)
        values = np.asarray(values, dtype=np.float32)
        observed, conditioning = np.asarray(observed, dtype=bool), np.asarray(conditioning, dtype=bool)
        if not np.isfinite(values).all() or (conditioning & ~observed).any():
            raise ValueError('Invalid author Air-36 observations or missing masks')
        path = output / f'csdi_fold{fold}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as prior:
                if not np.array_equal(prior['values'], values) or not np.array_equal(prior['conditioning'], conditioning):
                    raise ValueError('Existing prepared data differs; preserved')
        else:
            np.savez_compressed(path, values=values, observed=observed, conditioning=conditioning,
                                eval_mask=observed & ~conditioning, hist_mask=np.asarray(historical, dtype=bool),
                                cut_length=np.asarray(cuts), mean=mean, std=std, **splits)
        files.append({'fold': fold, 'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                      'shape': list(values.shape), 'split_sizes': {key: len(ids) for key, ids in splits.items()}})
    audit = {'configuration': cfg, 'raw_sha256': digest(raw), 'raw_preserved': True,
             'author_dataset_sha256': digest(source), 'files': files, 'shape_raw': list(frame.shape),
             'normalization_includes_validation': True, 'test_overlap_cutoff_preserved': True}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'prepared_folds': len(files), 'first_split': files[0]['split_sizes']}))


if __name__ == '__main__':
    main()
