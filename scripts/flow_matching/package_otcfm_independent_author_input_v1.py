"""Package complete existing PCs for the original author loader, without pickle.

Numerical coordinates/labels are unchanged. Cell/gene identifiers are encoded as
Unicode arrays in a new directory; original preprocessing artifacts are retained.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching import prepare_otcfm_multiome13176_public_independent_v1 as preparation

ROOT, read, sha, exclusive = preparation.ROOT, preparation.read, preparation.sha, preparation.exclusive
MODULE = 'scripts.flow_matching.package_otcfm_independent_author_input_v1'
CONFIG = 'configs/data/fm_otcfm_multiome13176_public_independent_author_input.v1.json'
TEST = 'tests/test_fm_otcfm_independent_author_input_package_v1.py'


def registration():
    parent = read(ROOT / preparation.CONFIG)
    report_path = parent['output_directory'] + '/public_paper_cohort_preprocessing_report.json'
    guard_path = parent['state_root'] + '/guard_receipt.json'
    report, guard = read(ROOT / report_path), read(ROOT / guard_path)
    if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or guard['report_sha256'] != sha(ROOT / report_path) or not report['passed']
            or report['cells'] != 35384 or report['features'] != 23418 or report['PCA_dimensions'] != 5
            or report['strict_full_reference_identity_gate_passed']
            or report['historical_paper_artifact_identity_certified']):
        raise ValueError('Complete guarded independent preprocessing and explicit historical limitations required')
    paths = [MODULE.replace('.', '/') + '.py', TEST, preparation.CONFIG, report_path, guard_path,
        preparation.original.AUTHOR_ROOT + '/runner/src/datamodules/components/time_dataset.py']
    output = 'data/public_datasets/flow_matching/otcfm_multiome13176_public_independent_author_input_v1'
    return dict(data_track=preparation.TRACK, original_preprocessing=preparation.CONFIG,
        source_report=report_path, source_guard=guard_path,
        source_input=report['target_input_for_author_loader'], raw_public=parent['input'],
        output_directory=output, input_npz=output + '/author_loader_input.npz',
        cells=35384, original_gene_features=23418, PC_dimensions=5, donor='13176',
        day_counts={'2': 9546, '3': 8965, '4': 9407, '7': 7466}, days=[2, 3, 4, 7],
        author_whiten=True, input_PCs_already_whitened=False, all_timepoints_in_preprocessing_fit=True,
        historical_paper_artifact_identity_certified=False, strict_full_reference_identity_gate_passed=False,
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        citation=parent['citation'], data_citation=parent['data_citation'], original_scope=parent['original_scope'],
        boundary='Numerically unchanged complete independently reconstructed donor13176 input. '
            'All arrays load with allow_pickle=False; trace IDs use Unicode and original PC/label/source-row '
            'arrays match the completed guarded preprocessing. Original author NPZ loader is executed exactly. '
            'Whiten=True applies once in the original data module; no new PCA, whitening or model fitting. '
            'Historical target roster/PCA identity remains unknown;4 unmatched reference cells remain unresolved.',
        all_original_experiments_complete=False)


def verify(config):
    if config != registration():
        raise ValueError('Complete independent loader input and frozen historical disclosure required')


def validate_payload(config, payload):
    import numpy as np
    pcs, labels, rows, ids, genes = (payload[k] for k in ['pcs', 'sample_labels', 'source_rows', 'cell_ids', 'feature_ids'])
    if (pcs.dtype != np.float32 or pcs.shape != (config['cells'], config['PC_dimensions'])
            or not np.isfinite(pcs).all() or labels.shape != rows.shape or labels.shape != ids.shape
            or labels.shape != (config['cells'],) or rows.dtype.kind not in 'iu'
            or not (np.diff(rows) > 0).all() or len(np.unique(ids)) != config['cells']
            or ids.dtype.kind != 'U' or genes.dtype.kind != 'U'
            or len(genes) != config['original_gene_features'] or len(np.unique(genes)) != len(genes)
            or dict(zip(*np.unique(labels.astype(str), return_counts=True))) != config['day_counts']
            or any(np.asarray(value).dtype.hasobject for value in payload.values())):
        raise ValueError('Every complete finite original numeric coordinate and pickle-free lineage array required')


def package(config):
    import h5py
    import numpy as np
    verify(config)
    report = read(ROOT / config['source_report'])
    for artifact in report['artifacts']:
        if sha(ROOT / artifact['path']) != artifact['sha256']:
            raise ValueError('Original completed preprocessing artifact changed')
    with np.load(ROOT / config['source_input'], allow_pickle=False) as source:
        payload = dict(pcs=source['pcs'].copy(), sample_labels=source['sample_labels'].copy(), source_rows=source['source_rows'].copy())
    with h5py.File(ROOT / config['raw_public']['path'], 'r') as f:
        donors = preparation.original.column(f['obs'], 'donor').astype(str)
        days = preparation.original.column(f['obs'], 'day').astype(int)
        rows = np.flatnonzero((donors == config['donor']) & np.isin(days, config['days']))
        if not np.array_equal(rows, payload['source_rows']) or not np.array_equal(days[rows], payload['sample_labels']):
            raise ValueError('Complete actual public cell roster must match original prepared input')
        payload['cell_ids'] = preparation.original.column(f['obs'], f['obs'].attrs['_index'])[rows].astype(str)
        payload['feature_ids'] = preparation.original.column(f['var'], f['var'].attrs['_index']).astype(str)
    validate_payload(config, payload)
    out = ROOT / config['output_directory']; out.mkdir(parents=True, exist_ok=False)
    path = ROOT / config['input_npz']; np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(payload) or any(not np.array_equal(stored[key], value) for key, value in payload.items()):
            raise ValueError('Every numeric and lineage array must serialize without pickle or changes')
    tree = ast.parse((ROOT / preparation.original.AUTHOR_ROOT / 'runner/src/datamodules/components/time_dataset.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tnet_dataset']
    if len(functions) != 1: raise ValueError('Exact original author loader required')
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<unchanged-author-loader>', 'exec'), namespace)
    pcs, labels, levels = namespace['tnet_dataset'](str(path))
    if not np.array_equal(pcs, payload['pcs']) or not np.array_equal(labels, payload['sample_labels']) or levels.tolist() != config['days']:
        raise ValueError('Original author loader must return identical complete numerical input')
    exclusive(out / 'input_packaging_report.json', dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG), data_track=preparation.TRACK,
        input_npz=str(path), input_sha256=sha(path), cells=config['cells'], PC_dimensions=config['PC_dimensions'],
        days=config['day_counts'], all_numeric_arrays_unchanged=True, all_arrays_pickle_free=True,
        exact_original_author_loader_verified=True, author_whiten_must_remain_true=True,
        new_PCA_whitening_or_training_executed=False, source_input_preserved=True,
        strict_full_reference_identity_gate_passed=False, historical_paper_artifact_identity_certified=False,
        boundary=config['boundary'], all_original_experiments_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register: exclusive(ROOT / CONFIG, registration())
    else: package(read(ROOT / CONFIG))
