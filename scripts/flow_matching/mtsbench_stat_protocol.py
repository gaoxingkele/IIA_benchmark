"""Frozen, original mTSBench HBOS/COPOD replay, explicitly transductive."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def fingerprint(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def environment_versions():
    # A system-site-packages venv can expose two distributions of the same name.
    # metadata.version follows import path precedence; last-wins enumeration does not.
    return {d.metadata['Name'].lower(): importlib.metadata.version(d.metadata['Name'])
            for d in importlib.metadata.distributions()}


def verify(queue, root=ROOT):
    for receipt in queue['source_receipts']:
        if sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen source changed: ' + receipt['path'])
    if len({j['id'] for j in queue['jobs']}) != len(queue['jobs']):
        raise ValueError('Duplicate job IDs')
    for job in queue['jobs']:
        if job['protocol'] != 'native_transductive_test_feature_fit_oracle_f1':
            raise ValueError('Native replay must disclose transductive fit and oracle F1')


def complete(job, root=ROOT):
    path = root / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'completed' or result.get('job_sha256') != fingerprint(job):
        raise ValueError('Result bound to a different job')
    if result.get('strict_TAB_result') is not False or result.get('diagnostic') is not False:
        raise ValueError('Native result incorrectly promoted or diagnostic')
    for artifact in result['artifacts']:
        if sha(root / artifact['path']) != artifact['sha256']:
            raise ValueError('Result artifact changed: ' + artifact['path'])
    return True


def run(queue, job, root=ROOT):
    verify(queue, root)
    if complete(job, root):
        return
    output = root / job['output_directory']
    if output.exists():
        raise FileExistsError('Partial outputs preserved; register a new attempt')
    actual = environment_versions()
    expected = json.loads((root / queue['environment_lock']).read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('Frozen execution environment changed')
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(queue['settings']['cpu_threads'])
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    torch.manual_seed(job['seed'])
    sys.path.insert(0, str(root / queue['author_source']))
    from Detectors import model_wrapper
    from Detectors.evaluation.basic_metrics import basic_metricor
    from importlib import import_module
    module = import_module('Detectors.models.' + job['algorithm'])
    cls = getattr(module, job['algorithm'])
    original_fit = cls.fit
    captured = []

    def fit_receipt(self, *args, **kwargs):
        value = original_fit(self, *args, **kwargs)
        captured.append(self)
        return value

    output.mkdir(parents=True)
    artifact_root = Path(queue['artifact_root']) / job['id']
    artifact_root.mkdir(parents=True, exist_ok=False)
    artifacts, entities = [], []
    metricor = basic_metricor()
    started = time.perf_counter()
    cls.fit = fit_receipt
    try:
        for entity in job['entities']:
            path = root / entity['test_path']
            if sha(path) != entity['test_sha256']:
                raise ValueError('Raw test file changed')
            frame = pd.read_csv(path)
            features = list(frame.columns[1:-1])
            if features != entity['feature_names'] or frame.columns[-1] != 'is_anomaly':
                raise ValueError('Feature/label boundary changed')
            values = frame.iloc[:, 1:-1].values.astype(float)
            labels = frame['is_anomaly'].astype(int).to_numpy()
            if not np.isin(labels, [0, 1]).all() or set(labels) != {0, 1}:
                raise ValueError('Binary label coverage required for ROC')
            begin = time.perf_counter()
            scores = getattr(model_wrapper, 'run_' + job['algorithm'])(values, **job['parameters'])
            runtime = time.perf_counter() - begin
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != labels.shape or not np.isfinite(scores).all() or len(captured) != 1:
                raise ValueError('Native score coverage/model capture failure')
            model = captured.pop()
            if not np.array_equal(scores, np.asarray(model.decision_scores_).ravel()):
                raise ValueError('Captured native fitted model differs from evaluated scores')
            metrics = {'PRC': float(metricor.metric_PR(labels, scores)),
                       'ROC': float(metricor.metric_ROC(labels, scores)),
                       'Best_F1': float(metricor.metric_PointF1(labels, scores))}
            prediction = artifact_root / (entity['id'] + '_scores.npz')
            checkpoint = artifact_root / (entity['id'] + '_native_model.joblib')
            np.savez_compressed(prediction, scores=scores, labels=labels, timestamps=frame.iloc[:, 0].to_numpy().astype(str))
            joblib.dump(model, checkpoint, compress=3)
            for artifact in (prediction, checkpoint):
                artifacts.append({'path': artifact.as_posix(), 'sha256': sha(artifact)})
            if sha(path) != entity['test_sha256']:
                raise ValueError('Raw data modified by native replay')
            entities.append({'entity': entity['id'], 'features': len(features), 'points': len(scores),
                             'anomaly_points': int(labels.sum()), 'fit_and_score_seconds': runtime,
                             'metrics': metrics, 'native_wrapper_parameters': job['parameters']})
            write_json(output / 'progress.json', {'job': job['id'], 'completed_entities': len(entities),
                                                  'required_entities': len(job['entities']), 'last_entity': entity['id']})
            del values, scores, model, frame
    finally:
        cls.fit = original_fit
    if len(entities) != len(job['entities']):
        raise ValueError('Entity coverage incomplete')
    verify(queue, root)
    result = {'status': 'completed', 'diagnostic': False, 'strict_TAB_result': False,
              'job_sha256': fingerprint(job), 'algorithm': job['algorithm'], 'dataset': job['dataset'],
              'seed': job['seed'], 'protocol': job['protocol'], 'entities': entities,
              'metrics': {k: float(np.mean([e['metrics'][k] for e in entities])) for k in ('PRC', 'ROC', 'Best_F1')},
              'total_seconds': time.perf_counter() - started, 'artifacts': artifacts,
              'boundary': 'Exact current mTSBench native wrapper and fitted models. Test features used to fit; labels only in evaluation; Best-F1 is test-label oracle with original epsilon. Entity macro. No PA. Original GRASP release/version equivalence not certified; feature counts differ from paper prose.'}
    write_json(output / 'result.json', result)
    print(json.dumps(result['metrics']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = json.loads((ROOT / args.queue).read_text(encoding='utf-8'))
    run(queue, next(j for j in queue['jobs'] if j['id'] == args.job_id))


if __name__ == '__main__':
    main()
