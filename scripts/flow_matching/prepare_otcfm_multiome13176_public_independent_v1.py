"""Independent public reconstruction of the paper-described donor13176 cohort.

The strict whole-reference identity gate remains failed. This separately named
track preserves that result, all target cells/genes and the demonstrated encoding.
It does not claim a historical paper artifact or replace missing author data.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v3 as original
from scripts.flow_matching import audit_otcfm_multiome_full_reference_overlap_v1 as overlap

ROOT, read, sha, write, exclusive = original.ROOT, original.read, original.sha, original.write, original.exclusive
MODULE = 'scripts.flow_matching.prepare_otcfm_multiome13176_public_independent_v1'
CONFIG = 'configs/data/fm_otcfm_multiome13176_public_independent.v1.json'
TEST = 'tests/test_fm_otcfm_multiome13176_public_independent_v1.py'
TRACK = 'independent_public_paper_described_donor13176'


def validate_alignment_evidence(report, guard):
    if (not report['passed'] or not report['diagnostic_only'] or not guard['passed']
            or guard['exit_code'] != 0 or guard['resource_guard_aborted']
            or report['author_reference_cells'] != 33701 or report['full_gene_features'] != 23418
            or not report['all_author_rows_examined'] or not report['all_gene_columns_examined']
            or not report['implicit_zero_coordinates_included']
            or report['mapped_rows_full_coordinate_checks'] != 33697
            or report['fully_matching_reference_donor_rows'] != 33697
            or report['fully_matching_global_same_day_rows'] != 0
            or report['full_coordinate_comparisons'] != 33697 * 23418
            or report['unresolved_anchor_rows'] != [1755, 3095, 3424, 4330]
            or report['duplicate_mapping_author_rows'] or report['mapped_but_coordinate_mismatching_rows']
            or report['original_full_reference_identity_qualified']):
        raise ValueError('Complete frozen encoding evidence and explicit unresolved historical reference are required')


def registration():
    parent = read(ROOT / original.CONFIG)
    audit = read(ROOT / overlap.CONFIG)
    report_path = audit['output_directory'] + '/full_reference_overlap_report.json'
    guard_path = audit['state_root'] + '/guard_receipt.json'
    report, guard = read(ROOT / report_path), read(ROOT / guard_path)
    validate_alignment_evidence(report, guard)
    if guard['report_sha256'] != sha(ROOT / report_path):
        raise ValueError('Successful actual full-reference audit must bind its report')
    paths = [MODULE.replace('.', '/') + '.py', TEST, original.CONFIG, original.MODULE.replace('.', '/') + '.py',
        overlap.CONFIG, report_path, guard_path,
        original.AUTHOR_ROOT + '/runner/src/datamodules/components/time_dataset.py',
        original.AUTHOR_ROOT + '/runner/src/datamodules/distribution_datamodule.py']
    return dict(data_track=TRACK, input=parent['inputs']['public'], paper_donor='13176', days=[2, 3, 4, 7],
        expected_cells=35384, expected_features=23418, public_layer=parent['public_layer'],
        encoding_scale=parent['encoding_scale'], encoding_transform=parent['encoding_transform'],
        pca_dimensions=5, pca_initial_vector_seed=0, pca_arpack_tolerance=0.,
        original_author_whiten_flag=True, all_timepoints_in_preprocessing_fit=True,
        historical_paper_artifact_identity_certified=False, strict_full_reference_identity_gate_passed=False,
        alignment_report=report_path, alignment_guard=guard_path,
        settings=parent['settings'], resource_lock=parent['resource_lock'],
        stable_samples=parent['stable_samples'], poll_seconds=parent['poll_seconds'], numerical_threads=2,
        state_root='experiments/runs/fm_otcfm_multiome13176_public_independent_v1',
        output_directory='data/public_datasets/flow_matching/processed_otcfm_multiome13176_public_independent_pca5_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        original_scope=parent['original_scope'], citation=parent['citation'], paper_section='E.5',
        data_citation=parent['public_data'],
        boundary='Independent reconstruction from the complete official public donor13176 days2,3,4,7 cohort, '
            'all35384 cells and23418 genes. Fixed encoding is supported by every coordinate of33697 shared '
            'reference cells;4 unresolved author reference rows remain explicit. No target cells are removed. '
            'First5 fresh centered PCs and exact released-author ddof0 whitening block; author-loader input '
            'contains unwhitened PCs so author whiten=True applies once. Historical PCA solver/seed/artifact '
            'identity and original target cell roster remain unknown. This new track does not satisfy or overwrite '
            'the failed strict whole-reference reconstruction. No training/performance or strict-inductive TSAD claim.',
        all_original_experiments_complete=False)


def verify(config):
    if config != registration():
        raise ValueError('Independent track, full cohort, alignment disclosure and original resource contract required')


def validate_target(config, rows, days, features):
    import numpy as np
    if (config['data_track'] != TRACK or len(rows) != config['expected_cells']
            or len(days) != len(rows) or len(np.unique(rows)) != len(rows)
            or set(days.tolist()) != set(config['days']) or len(features) != config['expected_features']
            or len(np.unique(features)) != len(features)):
        raise ValueError('Every public paper-described target cell/day/gene must remain')


def worker(config):
    import h5py
    import numpy as np
    import scipy
    from scipy import sparse
    import sklearn
    verify(config)
    item = config['input']; path = ROOT / item['path']
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError('All original public input bytes must remain unchanged')
    out = ROOT / config['output_directory']; out.mkdir(parents=True, exist_ok=False)
    def progress(**values):
        write(out / 'progress.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), **values))
    with h5py.File(path, 'r') as f:
        donors = original.column(f['obs'], 'donor').astype(str)
        days = original.column(f['obs'], 'day').astype(int)
        target = np.flatnonzero((donors == config['paper_donor']) & np.isin(days, config['days']))
        target_days = days[target].copy()
        features = original.column(f['var'], f['var'].attrs['_index'])
        target_ids = original.column(f['obs'], f['obs'].attrs['_index'])[target]
        validate_target(config, target, target_days, features)
        layer = f['layers'][config['public_layer']]
        if layer.attrs['encoding-type'] != 'csc_matrix':
            raise ValueError('Complete actual organizer CSC expression layer required')
        reverse = np.full(len(donors), -1, dtype=np.int32); reverse[target] = np.arange(len(target), dtype=np.int32)
        pointer = layer['indptr'][:]; target_pointer = [0]
        data_path, index_path = out / 'public13176_expression_data.f32', out / 'public13176_expression_indices.i32'
        observed = np.zeros(len(target), dtype=np.int64)
        with data_path.open('xb') as data_file, index_path.open('xb') as index_file:
            for feature in range(len(features)):
                lo, hi = int(pointer[feature]), int(pointer[feature + 1])
                rows, values = layer['indices'][lo:hi], layer['data'][lo:hi]
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Every original public expression value must be finite and nonnegative')
                selected = reverse[rows]; keep = (selected >= 0) & (values > 0)
                indices = selected[keep]; order = np.argsort(indices); indices = indices[order]
                if len(indices) > 1 and (np.diff(indices) <= 0).any():
                    raise ValueError('Unique complete target cell/gene coordinates required')
                data = original.transformed(values[keep][order], config['encoding_scale']).astype('<f4')
                data_file.write(data.tobytes()); index_file.write(indices.astype('<i4').tobytes())
                target_pointer.append(target_pointer[-1] + len(indices)); observed[indices] += 1
                if feature % 500 == 0:
                    progress(phase='exporting_every_public_target_cell_and_gene', completed_features=feature,
                        total_features=len(features), target_cells=len(target))
        if (observed == 0).any():
            raise ValueError('Zero-expression target cells must be reported, never silently removed')
    np.save(out / 'public13176_expression_indptr.npy', np.asarray(target_pointer, dtype=np.int64), allow_pickle=False)
    np.savez_compressed(out / 'public13176_cell_identity.npz', source_rows=target, cell_ids=target_ids,
        day=target_days, feature_ids=features, nonzeros_per_target_cell=observed)
    nnz = target_pointer[-1]
    raw_data = np.memmap(data_path, dtype='<f4', mode='r', shape=(nnz,))
    cache = out / 'centered_PCA_operator_data.f64'
    with cache.open('xb') as stream:
        for start in range(0, nnz, 1024 * 1024):
            stream.write(raw_data[start:start + 1024 * 1024].astype('<f8').tobytes())
    raw_data._mmap.close(); del raw_data
    data = np.memmap(cache, dtype='<f8', mode='r', shape=(nnz,))
    indices = np.memmap(index_path, dtype='<i4', mode='r', shape=(nnz,))
    matrix = sparse.csc_matrix((data, indices, np.asarray(target_pointer, dtype=np.int64)),
        shape=(len(target), len(features)), copy=False)
    progress(phase='computing_full_public_target_centered_PCA5', cells=len(target), features=len(features), nonzeros=nnz)
    pcs, pca = original.centered_pca(matrix, config['pca_dimensions'], config['pca_initial_vector_seed'], config['pca_arpack_tolerance'])
    whitened, scaler = original.author_whitening(pcs)
    independent = (pcs.astype(np.float64) - pcs.mean(axis=0, dtype=np.float64)) / pcs.std(axis=0, dtype=np.float64)
    if not np.allclose(whitened, independent, rtol=2e-6, atol=2e-6):
        raise ValueError('Independent complete-cohort ddof0 whitening check failed')
    np.savez_compressed(out / 'public13176_author_loader_input.npz', pcs=pcs, sample_labels=target_days, cell_ids=target_ids, source_rows=target)
    np.savez_compressed(out / 'public13176_whitened_diagnostic.npz', pcs=whitened, sample_labels=target_days)
    np.savez_compressed(out / 'public13176_PCA_and_whitening_parameters.npz',
        **{k: v for k, v in pca.items() if hasattr(v, 'shape')}, whitening_mean=scaler.mean_, whitening_scale=scaler.scale_)
    tree = ast.parse((ROOT / original.AUTHOR_ROOT / 'runner/src/datamodules/components/time_dataset.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tnet_dataset']
    if len(functions) != 1:
        raise ValueError('Exact original NPZ author loader required')
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<unchanged-author-loader>', 'exec'), namespace)
    actual, labels, levels = namespace['tnet_dataset'](str(out / 'public13176_author_loader_input.npz'))
    if not np.array_equal(actual, pcs) or not np.array_equal(labels, target_days) or levels.tolist() != config['days']:
        raise ValueError('Original full author NPZ loader must retain every cell, PC and timepoint')
    artifacts = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())
                 if p.is_file() and p.name != 'progress.json']
    exclusive(out / 'public_paper_cohort_preprocessing_report.json', dict(passed=True, data_track=TRACK,
        captured_utc=datetime.now(timezone.utc).isoformat(), configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
        cells=len(target), features=len(features), PCA_dimensions=5, days=dict(Counter(str(x) for x in target_days)),
        target_input_for_author_loader=str(out / 'public13176_author_loader_input.npz'), author_whiten_flag_must_remain_true=True,
        strict_full_reference_identity_gate_passed=False, historical_paper_artifact_identity_certified=False,
        original_paper_target_cell_roster_known=False, all_selected_target_cells_and_genes_retained=True,
        all_timepoints_in_preprocessing_fit=True, exact_author_loader_and_whitening_block_verified=True,
        pca_projection_relative_residual=pca['relative_projection_residual'], pca_orthogonality_error=pca['orthogonality_error'],
        original_inputs_overwritten=False, training_or_model_inference_executed=False,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__, sklearn=sklearn.__version__),
        artifacts=artifacts, boundary=config['boundary'], all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true'); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.register:
        exclusive(ROOT / CONFIG, registration()); return
    config = read(ROOT / CONFIG); verify(config)
    if args.worker:
        worker(config); return
    import psutil
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value: None)
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('Independent complete-cohort preprocessing already owned')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None, data_track=TRACK)
    def update(values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    try:
        if (ROOT / config['output_directory']).exists():
            raise FileExistsError('Preserve complete or partial independent cohort outputs')
        stable = 0
        while stable < config['stable_samples']:
            snapshot = api['resource_snapshot']()
            stable = stable + 1 if api['has_headroom'](snapshot, config['settings']) else 0
            update(dict(state='waiting_original_full_memory_admission', stable_samples=stable, **snapshot))
            time.sleep(config['poll_seconds'])
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [sys.executable, '-B', '-X', 'utf8', '-u', '-m', MODULE, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time(), command=command)
                update(dict(state='running_independent_public_paper_cohort_preprocessing', active_pid=child.pid, exact_child_handle=handle))
                code, usage = capped_wait(child, config, api, update)
        report = ROOT / config['output_directory'] / 'public_paper_cohort_preprocessing_report.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
        exclusive(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, exact_child_handle=handle,
            **usage, report_sha256=sha(report) if report.exists() else None))
        update(dict(state='completed' if passed else 'failed_attempt_preserved', active_pid=None,
            independent_public_track_preprocessing_complete=passed, strict_reference_reconstruction_complete=False))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
