"""Recompute full139 strict CrossAD metrics without NumPy/sklearn/model imports.

This extra audit consumes completed full-checkpoint proofs. It neither trains
models nor substitutes for full native inference, bootstrap, VUS or TAB runs.
"""
import argparse
import array
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha, write

MODULE = 'scripts.flow_matching.audit_crossad_full139_independent_metrics_v1'
CONFIG = 'configs/reproducibility/fm_crossad_full139_independent_metrics.v1.json'
QUEUE = 'configs/experiments/fm_crossad_original_CUDA_full139.v2.json'
UPSTREAM = 'configs/reproducibility/fm_crossad_CUDA_full139_independent_results.v1.json'


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def registration():
    q = read(ROOT / QUEUE)
    files = [QUEUE, UPSTREAM, q['delegated_first_configuration'],
             MODULE.replace('.', '/') + '.py',
             'tests/test_fm_crossad_full139_independent_metrics_v1.py',
             'scripts/flow_matching/cached_resource_snapshot_v1.py']
    return dict(queue=QUEUE, upstream=UPSTREAM, original_scope=q['original_scope'],
        required_conditions=139, state_root='experiments/runs/fm_crossad_full139_independent_metrics_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_full139_independent_metrics_v1',
        source_receipts=[dict(path=p, sha256=sha(ROOT / p)) for p in files],
        settings=dict(minimum_commit_headroom_bytes=16 * 2**30,
            emergency_commit_headroom_bytes=8 * 2**30, maximum_rss_bytes=2**30,
            maximum_private_bytes=1536 * 2**20, poll_seconds=10),
        numerical_metric_tolerance=dict(rtol=2e-6, atol=2e-6),
        metrics=['validation_only_linear_threshold', 'confusion_counts', 'precision',
                 'recall', 'f1', 'auroc_tie_aware_pair_ranks', 'AP_tie_group_step_integral',
                 'anomaly_events_and_detection_delay'],
        independently_recomputed_bootstrap=False, independently_recomputed_VUS=False,
        independently_recomputed_affiliation=False, strict_TAB_result=False,
        paper_equivalence=False, new_training_seed_slots=0, original_CPU_completion_credit=0,
        citation='papers/literature/mtsad_pdfs/li2025_crossad.pdf',
        data_citations='Per-condition raw SHA256 and complete timestamp proof in upstream full139 registry',
        all_original_experiments_complete=False)


def verify(c):
    if c != registration():
        raise ValueError('Frozen full139 independent metric audit changed')
    q = read(ROOT / c['queue'])
    if len(q['jobs']) != 139 or len({j['canonical_CPU_condition_reference'] for j in q['jobs']}) != 139:
        raise ValueError('Every original full139 condition remains required')
    return q, read(ROOT / c['upstream'])


def vector(stream, expected, descr):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('NPY required')
    version = stream.read(2)
    if version == b'\x01\x00':
        size = struct.unpack('<H', stream.read(2))[0]
    elif version in (b'\x02\x00', b'\x03\x00'):
        size = struct.unpack('<I', stream.read(4))[0]
    else:
        raise ValueError('Unsupported NPY version')
    if size > 65536:
        raise ValueError('Oversized NPY header')
    header = ast.literal_eval(stream.read(size).decode('utf-8' if version == b'\x03\x00' else 'latin1'))
    if header != dict(descr=descr, fortran_order=False, shape=(expected,)):
        raise ValueError('Complete frozen 1D dtype/shape required')
    code, width = {'<f4': ('f', 4), '|u1': ('B', 1), '<i8': ('q', 8)}[descr]
    payload = stream.read(expected * width + 1)
    if len(payload) != expected * width or stream.read(1):
        raise ValueError('Truncated/trailing vector payload')
    values = array.array(code)
    values.frombytes(payload)
    if sys.byteorder != 'little' and width > 1:
        values.byteswap()
    if descr == '<f4' and not all(map(math.isfinite, values)):
        raise ValueError('Every score must be finite')
    if descr == '|u1' and any(v not in (0, 1) for v in values):
        raise ValueError('Every test label must be binary')
    return values, hashlib.sha256(payload).hexdigest()


def percentile(sorted_values, alarm_percent):
    if not sorted_values or not 0 < alarm_percent < 100:
        raise ValueError('Complete validation scores and fixed alarm budget required')
    index = ((100.0 - alarm_percent) / 100.0) * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = index - low
    # Stable linear interpolation, including exact ties, in binary64.
    return sorted_values[low] + fraction * (sorted_values[high] - sorted_values[low])


def ranking(scores, labels):
    if len(scores) != len(labels) or not scores:
        raise ValueError('Full aligned scores and labels required')
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_above = 0
    negative_above = 0
    concordant_twice = 0
    AP_terms = []
    first = 0
    while first < len(order):
        last = first + 1
        value = scores[order[first]]
        while last < len(order) and scores[order[last]] == value:
            last += 1
        p = sum(labels[order[k]] for k in range(first, last))
        n = last - first - p
        concordant_twice += p * (2 * (negatives - negative_above - n) + n)
        positive_above += p
        negative_above += n
        if p:
            AP_terms.append((p / positives) * (positive_above / last))
        first = last
    return dict(auroc=concordant_twice / (2 * positives * negatives) if positives and negatives else None,
                average_precision=math.fsum(AP_terms) if positives else None)


def at_threshold(scores, labels, threshold):
    tp = fp = fn = events = hits = delay_sum = 0
    in_event = hit = False
    start = 0
    for i, (score, label) in enumerate(zip(scores, labels)):
        pred = score > threshold
        tp += bool(pred and label)
        fp += bool(pred and not label)
        fn += bool(not pred and label)
        if label and not in_event:
            events += 1
            start = i
            hit = False
        if label and pred and not hit:
            hits += 1
            delay_sum += i - start
            hit = True
        in_event = bool(label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    normal = len(labels) - sum(labels)
    return dict(micro=dict(precision=precision, recall=recall,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        true_positives=tp, false_positives=fp, false_negatives=fn),
        event_metrics=dict(event_recall=hits / events if events else None,
            events=events, detected_events=hits,
            mean_detection_delay_points=delay_sum / hits if hits else None,
            false_alarms_per_1000_normal_points=1000 * fp / normal if normal else None))


def equal_fields(actual, expected, tolerance):
    for key, value in actual.items():
        original = expected[key]
        if isinstance(value, dict):
            equal_fields(value, original, tolerance)
        elif value is None:
            if original is not None:
                raise ValueError('Undefined metric mismatch: ' + key)
        elif key in ('true_positives', 'false_positives', 'false_negatives', 'events', 'detected_events'):
            if value != original:
                raise ValueError('Exact full-population count mismatch: ' + key)
        elif not math.isfinite(original) or not math.isclose(value, original, rel_tol=tolerance['rtol'], abs_tol=tolerance['atol']):
            raise ValueError('Independent metric mismatch: ' + key)


def audit(c, q, upstream, job):
    identity = job['canonical_CPU_condition_reference']
    proof_path = Path(upstream['artifact_root']) / 'conditions' / (identity + '.json')
    proof = read(proof_path)
    if proof['canonical_CPU_condition_reference'] != identity or proof['raw_sha256'] != job['raw_sha256']:
        raise ValueError('Original full split/data identity required')
    source_job = read(ROOT / q['delegated_first_configuration'])['job'] if identity == q['delegated_first_original_condition'] else job
    source_path = ROOT / source_job['output_directory'] / 'result.json'
    source = read(source_path)
    if sha(source_path) != proof['source_result_sha256'] or source['status'] != 'completed' or source['id'] != source_job['id']:
        raise ValueError('Completed source result identity changed')
    arrays = ROOT / source_job['output_directory'] / 'native_scores.npz'
    bound = [a for a in source['artifacts'] if a['path'].replace('\\', '/').endswith('/native_scores.npz')]
    if len(bound) != 1 or sha(arrays) != bound[0]['sha256']:
        raise ValueError('Completed full score archive changed')
    counts = proof['native_points']
    with zipfile.ZipFile(arrays) as archive:
        if set(archive.namelist()) != {'train.npy', 'validation.npy', 'test.npy', 'labels.npy', 'source_rows.npy'}:
            raise ValueError('All full native splits and provenance arrays required')
        payload = {}
        hashes = {}
        for name in ('train', 'validation', 'test', 'labels', 'source_rows'):
            count = counts[name] if name in counts else counts['test']
            dtype = '|u1' if name == 'labels' else '<i8' if name == 'source_rows' else '<f4'
            with archive.open(name + '.npy') as stream:
                payload[name], hashes[name] = vector(stream, count, dtype)
    rows = payload.pop('source_rows')
    if any(v != source_job['train_length'] + i for i, v in enumerate(rows)):
        raise ValueError('Full timestamp order or tail coverage changed')
    labels = payload['labels']
    review = proof['label_and_source_row_review']
    if (review['labels_payload_sha256'] != hashes['labels'] or review['source_rows_payload_sha256'] != hashes['source_rows']
            or review['positive_labels'] != sum(labels) or review['full_test_points'] != len(labels)):
        raise ValueError('Independent labels/timestamps disagree with native proof')
    ranks = ranking(payload['test'], labels)
    sorted_validation = sorted(payload['validation'])
    validation_digest = hashlib.sha256(b''.join(struct.pack('<d', v) for v in payload['validation'])).hexdigest()
    checked = {}
    for ratio, original in proof['native_validation_control']['strict'].items():
        calibration = original['calibration']
        if (original['point_adjustment'] or calibration['calibration_split'] != 'validation'
                or calibration['calibration_points'] != counts['validation']
                or calibration['calibration_sha256'] != validation_digest
                or calibration['calibration_dataset_sha256'] != job['raw_sha256']):
            raise ValueError('Strict calibration must use only the full frozen validation split')
        threshold = percentile(sorted_validation, float(ratio))
        computed = at_threshold(payload['test'], labels, threshold)
        computed['micro'].update(ranks)
        computed['macro_f1'] = computed['micro']['f1']  # Original CrossAD uses one concatenated dataset entity.
        equal_fields(dict(threshold=threshold), calibration, c['numerical_metric_tolerance'])
        equal_fields(computed, original, c['numerical_metric_tolerance'])
        if set(original['per_entity']) != {job['dataset']}:
            raise ValueError('Revisit macro averaging if source entity grouping changes')
        checked[ratio] = dict(threshold=threshold, **computed)
    if set(checked) != {'0.1', '1', '5'}:
        raise ValueError('Every frozen strict alarm budget required')
    return dict(passed=True, canonical_CPU_condition_reference=identity,
        actual_execution_id=source['id'], dataset=job['dataset'], seed=job['seed'],
        mode=job['mode'], ablation_row=job.get('ablation_row'),
        source_result_sha256=sha(source_path), full_score_archive_sha256=sha(arrays),
        native_checkpoint_array_proof_sha256=sha(proof_path), complete_points=counts,
        all_score_label_and_timestamp_payload_sha256=hashes, positive_test_labels=sum(labels),
        independent_strict_metrics=checked, metric_tolerance=c['numerical_metric_tolerance'],
        independently_recomputed_bootstrap=False, independently_recomputed_VUS=False,
        independently_recomputed_affiliation=False, author_prefix_metrics_not_recomputed=True,
        new_neural_inference_executed=False, strict_TAB_result=False, paper_equivalence=False,
        original_CPU_completion_credit=0, all_original_experiments_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    if args.register:
        exclusive(ROOT / CONFIG, registration())
        return
    c = read(ROOT / CONFIG)
    q, upstream = verify(c)
    base, out = ROOT / c['state_root'], Path(c['artifact_root'])
    if args.probe_only:
        print(json.dumps(dict(required_conditions=len(q['jobs']), numpy_imported='numpy' in sys.modules,
                              torch_imported='torch' in sys.modules, sklearn_imported='sklearn' in sys.modules)))
        return
    if args.case:
        job = next(j for j in q['jobs'] if j['canonical_CPU_condition_reference'] == args.case)
        result = audit(c, q, upstream, job)
        exclusive(out / 'conditions' / (args.case + '.json'), result)
        return
    import psutil
    from scripts.flow_matching.cached_resource_snapshot_v1 import resource_snapshot
    base.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), required_conditions=139,
                 verified_ids=[], failed_preserved={}, active_pid=None, all_original_experiments_complete=False)
    def update(**values):
        state.update(values, captured_utc=datetime.now(timezone.utc).isoformat())
        write(base / 'status.json', state)
    update(state='waiting_full_native_proofs')
    while len(state['verified_ids']) + len(state['failed_preserved']) < 139:
        for job in q['jobs']:
            identity = job['canonical_CPU_condition_reference']
            if identity in state['verified_ids'] or identity in state['failed_preserved']:
                continue
            if not (Path(upstream['artifact_root']) / 'conditions' / (identity + '.json')).exists():
                continue
            resource = resource_snapshot()
            if resource['commit_available_bytes'] < c['settings']['minimum_commit_headroom_bytes']:
                update(state='waiting_for_memory', commit_available_bytes=resource['commit_available_bytes'])
                break
            command = [sys.executable, '-B', '-X', 'utf8', '-u', '-m', MODULE, '--case', identity]
            logs = base / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / (identity + '.stdout.log')).open('xb') as stdout, (logs / (identity + '.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                handle = psutil.Process(child.pid)
                update(state='independent_full_metric_recalculation', active_pid=child.pid,
                    active_create_time=handle.create_time(), actual_command=command, active_condition=identity)
                aborted = False
                peak_rss = peak_private = 0
                while child.poll() is None:
                    try:
                        memory = handle.memory_info()
                        peak_rss = max(peak_rss, memory.rss)
                        peak_private = max(peak_private, memory.private)
                        resource = resource_snapshot()
                        if (peak_rss > c['settings']['maximum_rss_bytes'] or peak_private > c['settings']['maximum_private_bytes']
                                or resource['commit_available_bytes'] < c['settings']['emergency_commit_headroom_bytes']):
                            child.terminate()
                            aborted = True
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.2)
                code = child.wait()
            report = out / 'conditions' / (identity + '.json')
            passed = code == 0 and not aborted and report.exists() and read(report)['passed']
            exclusive(base / 'guards' / (identity + '.json'), dict(exit_code=code, passed=passed,
                resource_guard_aborted=aborted, peak_rss_bytes=peak_rss, peak_private_bytes=peak_private,
                report_sha256=sha(report) if report.exists() else None))
            if passed:
                state['verified_ids'].append(identity)
            else:
                state['failed_preserved'][identity] = dict(exit_code=code, resource_guard_aborted=aborted)
            update(active_pid=None, state='between_full_metric_audits')
        update(state='waiting_full_native_proofs')
        time.sleep(c['settings']['poll_seconds'])
    update(state='all139_metrics_verified_other_experiments_pending' if not state['failed_preserved'] else 'completed_with_preserved_failures')


if __name__ == '__main__':
    main()
