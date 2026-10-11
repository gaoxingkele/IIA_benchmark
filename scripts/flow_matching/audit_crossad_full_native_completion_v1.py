"""Audit a complete original CrossAD training run and every saved test point."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def training_budget(job, details, trace):
    import numpy as np
    maximum, patience = job['train_parameters']['train_epochs'], job['train_parameters']['patience']
    losses = np.asarray(trace['validation_losses'])
    n = len(losses)
    if (details['declared_train_parameters'] != job['train_parameters']
            or not np.isfinite(losses).all() or not 0 < n <= maximum
            or trace['epochs_finished'] != n or details['epochs_completed'] != n
            or trace['declared_max_epochs'] != maximum):
        raise ValueError('The complete original training configuration and epochs must agree')
    if n < maximum and (n <= patience or np.any(losses[-patience:] < losses[:-patience].min())):
        raise ValueError('An early stop must exhaust the original patience after the best validation loss')
    expected = math.ceil((details['native_points']['train'] - job['model_parameters']['seq_len'] + 1)
        / job['train_parameters']['batch_size']) * n
    if details['training_forward_calls'] != expected or trace['training_forward_calls'] != expected:
        raise ValueError('Every original stride1 training window and batch must be included')
    return dict(original_maximum_epochs=maximum, original_patience=patience,
        actual_completed_epochs=n, full_training_forward_calls=expected,
        best_validation_epoch=int(np.argmin(losses))+1, original_early_stop_consistent=True)


def complete_scores(arrays, labels, start, lengths):
    import numpy as np
    for role in ('train', 'validation', 'test'):
        if arrays[role].shape != (lengths[role],) or not np.isfinite(arrays[role]).all():
            raise ValueError('Every full original score point must be finite and present')
    if (arrays['labels'].shape != (lengths['test'],)
            or not np.array_equal(arrays['labels'], labels[start:])
            or not np.array_equal(arrays['source_rows'], np.arange(start, len(labels)))):
        raise ValueError('Every test label and original raw source row must agree')


def audit(config_path):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
    config = read(config_path)
    for item in config['source_receipts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Preregistered full native audit source changed')
    queue = read(config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    for item in job['frozen_files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('An original scientific source or configuration changed')
    output = ROOT / job['output_directory']
    result = read(output / 'result.json')
    if (result['status'] != 'completed' or result['id'] != job['id']
            or result['mode'] != 'fresh_train' or result['seed'] != job['seed']
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            or result['paper_equivalence'] or result['strict_TAB_result']):
        raise ValueError('A complete fresh original training job with explicit evaluation boundaries required')
    for item in result['artifacts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('A complete original model or native output artifact changed')
    details, trace = result['details'], read(output / 'training_trace.json')
    budget = training_budget(job, details, trace)
    usage = read(output / 'resource_usage.json')
    if usage['resource_guard_aborted'] or details['parameter_count'] != config['full_model_parameters']:
        raise ValueError('The full original architecture and nonaborted resource execution required')
    settings = read(config['author_manifest'])
    raw = ROOT / settings['raw_root'] / 'data' / (job['dataset']+'.csv')
    if digest(raw) != job['raw_sha256']:
        raise ValueError('Original full raw data changed')
    labels = []
    for frame in pd.read_csv(raw, usecols=['data', 'cols'], chunksize=65536):
        labels.extend(frame.loc[frame['cols']=='label', 'data'].tolist())
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('All raw source labels must be binary')
    labels = labels.astype(np.uint8)
    reviewed = {}
    with np.load(output / 'native_scores.npz', allow_pickle=False) as arrays:
        complete_scores(arrays, labels, job['train_length'], details['native_points'])
        y, score = arrays['labels'], arrays['test']
        for ratio, recorded in result['metrics']['native_validation_control']['strict'].items():
            threshold = float(np.percentile(np.asarray(arrays['validation'], dtype=np.float64),
                100-float(ratio), method='linear'))
            prediction = score > threshold
            tp = int(np.sum(prediction & (y==1))); fp = int(np.sum(prediction & (y==0)))
            fn = int(np.sum(~prediction & (y==1)))
            values = dict(precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn), auroc=float(roc_auc_score(y, score)),
                average_precision=float(average_precision_score(y, score)))
            if (recorded['calibration']['threshold'] != threshold or recorded['point_adjustment']
                    or any(not math.isclose(recorded['micro'][k], v, rel_tol=1e-12, abs_tol=1e-12)
                        for k,v in values.items())):
                raise ValueError('Full raw-aligned validation-only point metrics must independently agree')
            reviewed[ratio] = dict(threshold=threshold, true_positives=tp, false_positives=fp,
                false_negatives=fn, metrics=values)
        native_count = details['original_scored_test_points']
        original_energy = np.load(output / 'workspace/test_results' / job['dataset'] / 'results_0/a_test_energy.npy').mean(axis=-1)
        original_labels = np.load(output / 'workspace/test_results' / job['dataset'] / 'results_0/a_gt.npy').reshape(-1)
        if (not np.array_equal(original_labels, y[:native_count])
                or not np.allclose(original_energy, score[:native_count], rtol=1e-6, atol=1e-7)
                or len(y)-native_count != details['original_unscored_tail_points']):
            raise ValueError('Original native inference and complete tail coverage must agree')
        p, r, thresholds = precision_recall_curve(original_labels, original_energy)
        f = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=p+r>0)
        i = int(np.argmax(f))
        original = result['metrics']['original_author']['_best_f1'][0]
        for k, value in [('P_best', p[i]), ('R_best', r[i]), ('F1_best', f[i])]:
            if not math.isclose(original[k], float(value), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Independent full native test-label-oracle F1 differs')
        oracle = dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
            full_native_points=native_count, test_label_selection=True, point_adjustment=False)
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        job_id=job['id'], original_seed=job['seed'], training_budget=budget,
        full_model_parameters=details['parameter_count'], raw_sha256=job['raw_sha256'],
        full_native_points=details['native_points'], original_unscored_tail_points=details['original_unscored_tail_points'],
        full_raw_labels_and_source_rows_checked=True, validation_only_point_metrics=reviewed,
        original_test_label_oracle_points_independently_checked=oracle,
        original_range_and_affiliation_artifacts_hash_checked_only=True,
        model_reinference_performed=False, required_original_main_seeds=6,
        registered_CrossAD_jobs=len(queue['jobs']), original_source_result=result,
        original_scope=config['original_scope'], author_equivalence_certified=False,
        strict_TAB_result=False, all_original_experiments_complete=False)
    target = ROOT / config['output_directory']
    target.mkdir(parents=True, exist_ok=False)
    (target/'independent_full_native_completion.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, job=job['id'], epochs=budget['actual_completed_epochs'],
        full_test_points=details['native_points']['test'], required_training_seeds=6)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    audit(parser.parse_args().config)
