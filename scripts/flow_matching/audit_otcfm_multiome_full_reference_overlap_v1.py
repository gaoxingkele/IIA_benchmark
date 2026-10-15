"""Audit every author reference cell and gene after the preserved identity failure.

Negative correspondences are diagnostic outcomes, never permission to preprocess
or train. A global same-day search diagnoses provenance differences separately
from the registered donor32606 comparison. No RNA coordinate is fitted or changed.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching import reconstruct_otcfm_multiome13176_public_v3 as source

ROOT, read, sha, write, exclusive = source.ROOT, source.read, source.sha, source.write, source.exclusive
MODULE = 'scripts.flow_matching.audit_otcfm_multiome_full_reference_overlap_v1'
CONFIG = 'configs/reproducibility/fm_otcfm_multiome_full_reference_overlap.v1.json'
TEST = 'tests/test_fm_otcfm_multiome_full_reference_overlap_v1.py'


def registration():
    parent = read(ROOT / source.CONFIG)
    paths = [MODULE.replace('.', '/') + '.py', TEST, source.CONFIG,
        source.MODULE.replace('.', '/') + '.py',
        parent['state_root'] + '/guard_receipt.json', parent['state_root'] + '/worker.stderr.log']
    return dict(inputs=parent['inputs'], reference_donor=parent['reference_donor'],
        days=parent['days'], expected_reference_cells=33701, expected_public_reference_candidates=33704,
        expected_public_cells=161868, expected_features=23418, public_layer=parent['public_layer'],
        encoding_scale=parent['encoding_scale'], encoding_tolerance=parent['encoding_tolerance'],
        settings=parent['settings'], resource_lock=parent['resource_lock'],
        stable_samples=parent['stable_samples'], poll_seconds=parent['poll_seconds'], numerical_threads=2,
        original_scope=parent['original_scope'],
        state_root='experiments/runs/fm_otcfm_multiome_full_reference_overlap_v1',
        output_directory='data/public_datasets/flow_matching/otcfm_multiome_full_reference_overlap_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in paths],
        citation=parent['citation'], author_data=parent['author_data'], public_data=parent['public_data'],
        boundary='Complete33701-cell reference, all23418 genes and original expression tolerance. '
            'Every failed anchor is retained; optional global same-day candidates are provenance diagnostics, '
            'not substitutes for the registered donor. Every resolved row is then checked across every gene, '
            'including implicit zeros. No PCA, whitening, training input or model is produced. '
            'Numerical expression agreement does not prove biological identity or which metadata source is correct.',
        all_original_experiments_complete=False)


def verify(config):
    if config != registration():
        raise ValueError('Complete reference/gene scope, fixed encoding and original resource policy required')


def expression_candidates(columns, values, frequencies, selected, public_column, tolerance):
    import numpy as np
    if len(columns) != len(values) or not len(columns):
        raise ValueError('Every positive author reference coordinate required')
    candidates = set(selected)
    for i in np.argsort(frequencies[columns], kind='stable'):
        rows, actual = public_column(int(columns[i]))
        candidates.intersection_update(rows[np.isclose(actual, values[i], **tolerance)].tolist())
        if len(candidates) <= 1:
            break
    return sorted(candidates)


def mapped_public_values(public_rows, public_values, mapping, public_cells, scale):
    """Gather complete sparse columns, including zeros and duplicate mappings."""
    import numpy as np
    if (len(public_rows) != len(public_values) or len(np.unique(public_rows)) != len(public_rows)
            or (public_rows < 0).any() or (public_rows >= public_cells).any()
            or (mapping < -1).any() or (mapping >= public_cells).any()):
        raise ValueError('Canonical full sparse public coordinates required')
    active = np.flatnonzero(mapping >= 0)
    unique, inverse = np.unique(mapping[active], return_inverse=True)
    reverse = np.full(public_cells, -1, dtype=np.int32)
    reverse[unique] = np.arange(len(unique), dtype=np.int32)
    destinations = reverse[public_rows]
    keep = destinations >= 0
    by_public_cell = np.zeros(len(unique), dtype=np.float64)
    by_public_cell[destinations[keep]] = source.transformed(public_values[keep], scale)
    full = np.zeros(len(mapping), dtype=np.float64)
    full[active] = by_public_cell[inverse]
    return full, active


def compare_reference_column(author_rows, author_values, actual, active, tolerance):
    import numpy as np
    expected = np.zeros(len(actual), dtype=np.float64)
    if len(np.unique(author_rows)) != len(author_rows):
        raise ValueError('Canonical author sparse coordinates required')
    expected[author_rows] = author_values
    difference = np.abs(expected - actual)
    mismatch = ~np.isclose(expected, actual, **tolerance)
    mask = np.zeros(len(actual), dtype=bool)
    mask[active] = True
    mismatch &= mask
    difference[~mask] = 0
    return mismatch, difference


def worker(config):
    import gc
    import h5py
    import numpy as np
    from scipy import sparse
    verify(config)
    for item in config['inputs'].values():
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Every original author/public byte must match')
    out = ROOT / config['output_directory']
    out.mkdir(parents=True, exist_ok=False)
    def progress(**values):
        write(out / 'progress.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(), **values))
    with h5py.File(ROOT / config['inputs']['author']['path'], 'r') as af, \
            h5py.File(ROOT / config['inputs']['public']['path'], 'r') as pf:
        features = source.column(af['var'], af['var'].attrs['_index'])
        public_features = source.column(pf['var'], pf['var'].attrs['_index'])
        ad = source.column(af['obs'], 'donor').astype(str)
        at = source.column(af['obs'], 'day').astype(int)
        pd = source.column(pf['obs'], 'donor').astype(str)
        pt = source.column(pf['obs'], 'day').astype(int)
        selected = np.flatnonzero((pd == config['reference_donor']) & np.isin(pt, config['days']))
        if (len(ad) != config['expected_reference_cells'] or len(pd) != config['expected_public_cells']
                or len(selected) != config['expected_public_reference_candidates']
                or set(ad) != {config['reference_donor']} or set(at) != set(config['days'])
                or len(features) != config['expected_features'] or not np.array_equal(features, public_features)):
            raise ValueError('Complete actual reference/public roster and gene order required')
        raw = af['X']
        if raw.attrs['encoding-type'] != 'csr_matrix':
            raise ValueError('Exact full author CSR required')
        matrix = sparse.csr_matrix((raw['data'][:], raw['indices'][:], raw['indptr'][:]), shape=tuple(raw.attrs['shape']))
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
        if matrix.shape != (33701, 23418) or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError('All finite nonnegative actual author reference coordinates required')
        layer = pf['layers'][config['public_layer']]
        if layer.attrs['encoding-type'] != 'csc_matrix':
            raise ValueError('Exact full public CSC required')
        pointer = layer['indptr'][:]
        frequencies = np.diff(pointer)
        reference_set = set(selected.tolist())
        day_sets = {int(day): set(np.flatnonzero(pt == day).tolist()) for day in config['days']}
        @lru_cache(maxsize=2048)
        def all_column(feature):
            lo, hi = int(pointer[feature]), int(pointer[feature + 1])
            rows, values = layer['indices'][lo:hi], layer['data'][lo:hi]
            keep = values > 0
            return rows[keep], source.transformed(values[keep], config['encoding_scale'])
        mapping = np.full(len(ad), -1, dtype=np.int64)
        primary = np.full(len(ad), -1, dtype=np.int64)
        unresolved = []
        global_only = []
        with (out / 'all_author_anchor_outcomes.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            for row in range(len(ad)):
                lo, hi = matrix.indptr[row:row + 2]
                args = (matrix.indices[lo:hi], matrix.data[lo:hi], frequencies)
                candidates = expression_candidates(*args, reference_set, all_column, config['encoding_tolerance'])
                outside = []
                if len(candidates) == 1:
                    primary[row] = mapping[row] = candidates[0]
                else:
                    outside = expression_candidates(*args, day_sets[int(at[row])], all_column, config['encoding_tolerance'])
                    if len(outside) == 1:
                        mapping[row] = outside[0]
                        global_only.append(row)
                    else:
                        unresolved.append(row)
                journal.write(json.dumps(dict(author_row=row, author_day=int(at[row]),
                    reference_candidates=candidates, global_same_day_candidates=outside)) + '\n')
                if row % 1000 == 0:
                    journal.flush()
                    progress(phase='mapping_every_author_reference_cell', completed_cells=row,
                        total_cells=len(ad), unresolved_cells=len(unresolved), global_only_candidates=len(global_only))
        all_column.cache_clear()
        np.savez_compressed(out / 'all_author_reference_mappings.npz', primary=primary, diagnostic_mapping=mapping,
            author_days=at, public_days=pt, public_donors=pd, public_cell_ids=source.column(pf['obs'], pf['obs'].attrs['_index']))
        columns = matrix.tocsc(); del matrix; gc.collect()
        mismatch_counts = np.zeros(len(ad), dtype=np.int64)
        max_errors = np.zeros(len(ad), dtype=np.float64)
        compared = int(np.count_nonzero(mapping >= 0))
        for feature in range(config['expected_features']):
            lo, hi = int(pointer[feature]), int(pointer[feature + 1])
            rows, values = layer['indices'][lo:hi], layer['data'][lo:hi]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Every public matrix coordinate must remain finite and nonnegative')
            actual, active = mapped_public_values(rows, values, mapping, len(pd), config['encoding_scale'])
            a, b = columns.indptr[feature:feature + 2]
            bad, difference = compare_reference_column(columns.indices[a:b], columns.data[a:b], actual, active, config['encoding_tolerance'])
            mismatch_counts += bad
            max_errors = np.maximum(max_errors, difference)
            if feature % 500 == 0:
                progress(phase='checking_every_gene_and_implicit_zero', completed_features=feature,
                    total_features=len(features), mapped_cells=compared, cells_with_coordinate_mismatches=int(np.count_nonzero(mismatch_counts)))
        matched = (mapping >= 0) & (mismatch_counts == 0)
        within = matched & (primary >= 0)
        counts = np.bincount(mapping[mapping >= 0], minlength=len(pd))
        duplicate = np.flatnonzero((mapping >= 0) & (counts[np.maximum(mapping, 0)] > 1))
        qualified = bool(within.all() and not len(duplicate) and np.array_equal(at, pt[mapping]))
        np.savez_compressed(out / 'all_reference_coordinate_comparisons.npz', mismatch_counts=mismatch_counts,
            maximum_absolute_errors=max_errors, full_expression_match=matched, full_reference_donor_match=within)
        discrepancies = [dict(author_row=int(i), author_day=int(at[i]), public_row=int(mapping[i]),
            public_donor=str(pd[mapping[i]]) if mapping[i] >= 0 else None,
            public_day=int(pt[mapping[i]]) if mapping[i] >= 0 else None,
            mismatching_gene_coordinates=int(mismatch_counts[i]), maximum_absolute_error=float(max_errors[i]))
            for i in np.flatnonzero(~within)]
        exclusive(out / 'full_reference_overlap_report.json', dict(passed=True, diagnostic_only=True,
            captured_utc=datetime.now(timezone.utc).isoformat(), configuration=CONFIG, configuration_sha256=sha(ROOT / CONFIG),
            author_reference_cells=len(ad), public_reference_candidates=len(selected), full_gene_features=len(features),
            all_author_rows_examined=True, all_gene_columns_examined=True, mapped_rows_full_coordinate_checks=compared,
            full_coordinate_comparisons=compared * len(features), implicit_zero_coordinates_included=True,
            fully_matching_reference_donor_rows=int(np.count_nonzero(within)),
            fully_matching_global_same_day_rows=int(np.count_nonzero(matched & (primary < 0))),
            unresolved_anchor_rows=unresolved, duplicate_mapping_author_rows=duplicate.tolist(),
            mapped_but_coordinate_mismatching_rows=np.flatnonzero((mapping >= 0) & ~matched).tolist(),
            discrepancies=discrepancies, original_full_reference_identity_qualified=qualified,
            paper_PCA_input_produced=False, original_singlecell_training_started=False,
            full_historical_biological_cell_identity_proven=False, boundary=config['boundary'],
            all_original_experiments_complete=False))
    print(json.dumps(dict(full_reference_donor_matches=int(np.count_nonzero(within)),
        required_author_cells=len(ad), full_reference_qualified=qualified)), flush=True)


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
        raise RuntimeError('This full-reference audit already has an owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), active_pid=None)
    def update(value):
        state.update(value, captured_utc=datetime.now(timezone.utc).isoformat()); write(base / 'status.json', state)
    try:
        if (ROOT / config['output_directory']).exists():
            raise FileExistsError('Preserve all prior full or partial diagnostic artifacts')
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
                update(dict(state='running_full_reference_overlap_audit', active_pid=child.pid, exact_child_handle=handle))
                code, usage = capped_wait(child, config, api, update)
        report = ROOT / config['output_directory'] / 'full_reference_overlap_report.json'
        passed = code == 0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
        exclusive(base / 'guard_receipt.json', dict(passed=passed, exit_code=code, exact_child_handle=handle,
            **usage, report_sha256=sha(report) if report.exists() else None))
        update(dict(state='completed' if passed else 'failed_attempt_preserved', active_pid=None,
            full_overlap_audit_complete=passed, original_input_reconstruction_complete=False))
    finally:
        owner.close()


if __name__ == '__main__':
    main()
