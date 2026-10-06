"""Stage exact author graph HDF5 data without modifying downloaded archives."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.flow_matching.prepare_saits_physio import sha


def main():
    import h5py
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data/fm_graph_author_air.v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    raw = ROOT / config['raw_zip']
    directory = ROOT / config['output_root']
    marker = directory / 'data_audit.json'
    config_sha, raw_sha = sha(args.config), sha(raw)
    if marker.exists():
        report = json.loads(marker.read_text(encoding='utf-8'))
        if report['config_sha256'] != config_sha or report['raw_sha256'] != raw_sha:
            raise ValueError('Registered graph data identity changed')
        if any(sha(directory / name) != entry['sha256'] for name, entry in report['files'].items()):
            raise ValueError('Staged author graph HDF5 changed')
        print(json.dumps(report))
        return
    if directory.exists():
        raise ValueError('Unregistered graph staging directory; preserve it')
    directory.mkdir(parents=True)
    report = {'status': 'staged_raw_author_graph_data_not_model_preprocessed', 'config_sha256': config_sha,
              'raw_sha256': raw_sha, 'data_citation': config['data_citation'], 'files': {}}
    with zipfile.ZipFile(raw) as archive:
        for name, expected_nodes in config['members'].items():
            path = directory / name
            with archive.open(name) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output)
            with h5py.File(path, 'r') as handle:
                values = handle['pm25/block0_values'][:]
                timestamps = handle['pm25/axis1'][:]
                if values.ndim != 2 or values.shape[1] != expected_nodes or len(timestamps) != len(values):
                    raise ValueError('Released graph dataset dimensions mismatch')
                if np.isinf(values).any() or np.any(np.diff(timestamps) <= 0):
                    raise ValueError('Invalid graph data values or timestamp order')
                item = {'shape': list(values.shape), 'timestamps': len(timestamps),
                        'first_timestamp': str(np.datetime64(int(timestamps[0]), 'ns')),
                        'last_timestamp': str(np.datetime64(int(timestamps[-1]), 'ns')),
                        'native_missing_fraction': float(np.isnan(values).mean()), 'hdf5_groups': list(handle.keys())}
                if 'eval_mask' in handle:
                    mask = handle['eval_mask/block0_values'][:].astype(bool)
                    if mask.shape != values.shape or np.any(mask & np.isnan(values)):
                        raise ValueError('Original graph evaluation mask includes unknown targets')
                    item.update(released_evaluation_targets=int(mask.sum()))
            item['sha256'] = sha(path)
            report['files'][name] = item
    report['boundary'] = config['boundary']
    marker.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
