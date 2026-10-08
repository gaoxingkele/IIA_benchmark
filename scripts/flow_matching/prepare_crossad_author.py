"""Parse every full official dataset losslessly; preserve original raw files."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from iia_benchmark.models.fm_crossad_author_runtime import prepare_runtime, import_author, digest
from scripts.flow_matching.prepare_tsad_execution import write_json


def main():
    settings = json.loads((ROOT / 'configs/experiments/fm_crossad_author_execution.v1.json').read_text())
    prepare_runtime(ROOT / settings['source'], ROOT / settings['runtime_root'])
    import_author(ROOT / settings['runtime_root'])
    provider = importlib.import_module('data_provider.data_provider')
    cache = ROOT / settings['cache_root']
    cache.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name in settings['datasets']:
        raw = ROOT / settings['raw_root'] / 'data' / (name + '.csv')
        target, receipt_path = cache / (name + '.pkl'), cache / (name + '.json')
        raw_sha = digest(raw)
        if target.exists():
            receipt = json.loads(receipt_path.read_text())
            assert raw_sha == receipt['raw_sha256'] and digest(target) == receipt['cache_sha256']
        else:
            frame = provider.read_data(str(raw))
            features = frame.drop(columns=['label']).to_numpy()
            labels = frame['label'].to_numpy()
            assert np.isfinite(features).all() and np.isin(labels, [0, 1]).all()
            frame.to_pickle(target)
            receipt = {'dataset': name, 'raw_path': raw.relative_to(ROOT).as_posix(),
                       'raw_sha256': raw_sha, 'cache_sha256': digest(target), 'shape': list(frame.shape),
                       'columns': list(frame.columns), 'train_length': int(provider.read_meta(str(ROOT / settings['raw_root']), name)[1]),
                       'finite_features': True, 'binary_labels': True,
                       'parser': 'Unmodified official read_data, complete source CSV, no raw overwrite'}
            write_json(receipt_path, receipt)
        receipts.append(receipt)
        print(json.dumps({'prepared': name, 'shape': receipt['shape']}), flush=True)
    write_json(ROOT / 'docs/reports/fm_crossad_author_data_2026-10-09.json', {'datasets': receipts,
               'raw_preserved': True, 'grouping_boundary': 'Official flat CSV order; entity provenance is not reconstructed or claimed'})


if __name__ == '__main__':
    main()
