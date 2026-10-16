"""Losslessly package complete released EB/CITE PC arrays for the native loader."""
import argparse
from pathlib import Path
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG = 'configs/data/fm_otcfm_released_EB_CITE_lossless_native_inputs.v1.json'


def labels(h5, label_key):
    import h5py
    import numpy as np
    field = h5['obs'][label_key]
    if isinstance(field, h5py.Group):
        categories = field['categories'].asstr()[:]
        codes = field['codes'][:]
        if (codes < 0).any() or (codes >= len(categories)).any():
            raise ValueError('Every original cell requires its actual ordered timepoint')
        return codes.astype('int64'), categories.tolist()
    values = field[:]
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Original numeric day values must remain exact integers')
    return values, [str(x) for x in np.unique(values)]


def package(config):
    import h5py
    import numpy as np
    base = Path(config['output_directory'])
    base.mkdir(parents=True, exist_ok=False)
    records = []
    for item in config['inputs']:
        source = ROOT / item['source']; checksum = sha(source)
        if checksum != item['source_sha256']: raise ValueError('Complete original author data changed')
        with h5py.File(source, 'r') as h5:
            pcs = h5['obsm/X_pca'][:]
            y, time_names = labels(h5, item['label_key'])
            cell_ids = np.asarray(h5['obs'][h5['obs'].attrs['_index']].asstr()[:], dtype='U')
        counts = {str(k): int((y == k).sum()) for k in np.unique(y)}
        if (pcs.shape != (item['cells'], item['released_PC_dimensions']) or len(cell_ids) != item['cells']
                or len(np.unique(cell_ids)) != item['cells'] or counts != item['model_timepoint_counts']
                or not np.isfinite(pcs).all()):
            raise ValueError('All source cells, original PC columns, label counts and finite coordinates required')
        output = base / item['filename']
        with output.open('xb') as stream:
            np.savez_compressed(stream, pcs=pcs, sample_labels=y, cell_ids=cell_ids,
                timepoint_names=np.asarray(time_names, dtype='U'))
        with np.load(output, allow_pickle=False) as packed:
            if (not np.array_equal(packed['pcs'], pcs) or packed['pcs'].dtype != pcs.dtype
                    or not np.array_equal(packed['sample_labels'], y)
                    or not np.array_equal(packed['cell_ids'], cell_ids)
                    or not np.array_equal(packed['timepoint_names'], np.asarray(time_names, dtype='U'))):
                raise ValueError('Every packaged original coordinate/label/traceID must remain exactly equal')
        if sha(source) != checksum: raise ValueError('Raw input must remain immutable during packaging')
        records.append(dict(id=item['id'], path=str(output), sha256=sha(output), bytes=output.stat().st_size,
            cells=item['cells'], PC_dimensions=item['released_PC_dimensions'], native_max_dim=5,
            native_whiten=True, PCs_already_whitened=False, model_timepoint_counts=counts,
            timepoint_names=time_names, exact_all_PC_values_dtype_labels_and_IDs_preserved=True,
            all_arrays_allow_pickle_false=True, source=str(source), source_sha256=checksum,
            required_intermediate_leaveout_indices=list(range(1, len(counts)-1)),
            historical_paper_identity_certified=False, boundary=item['boundary']))
    report = dict(status='completed', records=records, model_training_updates=0,
        full_raw_data_preserved=True, new_benchmark_performance_claim=False,
        original_scope=config['original_scope'], all_original_experiments_complete=False)
    exclusive(base / 'input_packaging_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default=CONFIG)
    args = parser.parse_args()
    import json
    print(json.dumps(package(read(ROOT / args.config))))


if __name__ == '__main__': main()
