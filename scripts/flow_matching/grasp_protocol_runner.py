"""Full per-entity GRASP successor runs, with frozen provenance and no PA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def environment_versions():
    return {'python': sys.version, **{name: importlib.metadata.version(name)
                                     for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'psutil']}}


def verify(queue, root=ROOT):
    for receipt in queue['source_receipts']:
        if sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen GRASP source changed: '+receipt['path'])
    manifest = json.loads((root / queue['manifest']).read_text(encoding='utf-8'))
    registration = json.loads((root / queue['registration_config']).read_text(encoding='utf-8'))
    from scripts.flow_matching.register_grasp_protocol import score_recipes
    declared_profiles = {p['id']:{**registration['base_parameters'],**p['overrides']} for p in registration['profiles']}
    expected = {(profile, data['id'], entity['id'], seed)
                for profile in queue['profiles'] for data in manifest['datasets']
                for entity in data['entities'] for seed in queue['seeds']}
    actual = {(j['profile'], j['dataset'], j['entity'], j['seed']) for j in queue['jobs']}
    if expected != actual or len(actual) != len(queue['jobs']) or len({j['id'] for j in queue['jobs']}) != len(actual):
        raise ValueError('Full profile/dataset/entity/seed Cartesian scope differs')
    models = {}
    for profile, model_path in queue['profiles'].items():
        model = json.loads((root / model_path).read_text(encoding='utf-8'))
        models[profile] = (model,sha(root / model_path))
        p = model['parameters']
        if p != declared_profiles[profile]:
            raise ValueError('Paper main/ablation profile differs')
        for key, value in {'window':50,'hidden':128,'blocks':2,'epochs':1500,'validation_interval':50,
                           'batch_size':256,'learning_rate':.001,'device':'cuda:0'}.items():
            if p[key] != value:
                raise ValueError('Paper full training budget differs: '+profile+' '+key)
    if manifest['train_fraction'] != .8 or manifest['purge_points'] != 0 or len(queue['seeds']) != 10:
        raise ValueError('Original split/seed count differs')
    if queue['seeds'] != registration['seeds'] or set(queue['profiles']) != set(declared_profiles):
        raise ValueError('Frozen training seeds or profiles differ')
    entities = {(d['id'],e['id']):e for d in manifest['datasets'] for e in d['entities']}
    for job in queue['jobs']:
        if job['model_config'] != queue['profiles'][job['profile']]:
            raise ValueError('Profile binding differs')
        model, model_sha = models[job['profile']]
        if job['model_config_sha256'] != model_sha or not model['citation']:
            raise ValueError('Model config or citation binding differs')
        entity = entities[job['dataset'],job['entity']]
        if (job['expected_optimizer_updates'] != 1500*math.ceil(entity['fit_window_count']/256)
                or job['test_points'] != entity['test_points'] or job['validation_points'] != entity['validation_points']
                or job['score_recipes'] != score_recipes(job['profile'],registration)):
            raise ValueError('Entity training budget or inference scope differs')
    return manifest


def complete(job, root=ROOT):
    path = root / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('job_sha256') != fingerprint(job) or result.get('status') != 'completed':
        raise ValueError('Result job binding differs')
    if result.get('diagnostic') is not False or result.get('paper_equivalence_certified') is not False:
        raise ValueError('Diagnostic or unproved paper equivalence promoted')
    if result['epochs_completed'] != 1500 or result['optimizer_updates'] != job['expected_optimizer_updates']:
        raise ValueError('Partial training cannot count as complete')
    if result['score_recipes'] != job['score_recipes']:
        raise ValueError('Inference sensitivities missing')
    if [m['recipe'] for m in result['metrics']] != job['score_recipes']:
        raise ValueError('Actual inference metrics missing')
    if result['scored_points'] != job['test_points'] or result['validation_points'] != job['validation_points']:
        raise ValueError('Incomplete point coverage')
    for artifact in result['artifacts']:
        if sha(root / artifact['path']) != artifact['sha256']:
            raise ValueError('Checkpoint/score artifact differs')
    if len(result['artifacts']) != len(job['score_recipes'])+2:
        raise ValueError('Checkpoint, epoch history or full scores missing')
    history_path = root / job['output_directory'] / 'training_history.jsonl'
    history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
    if (len(history) != 1500 or [r['epoch'] for r in history] != list(range(1,1501))
            or history[-1]['optimizer_updates'] != job['expected_optimizer_updates']
            or [r['epoch'] for r in history if 'validation_loss' in r] != list(range(50,1501,50))):
        raise ValueError('Actual full epoch and validation history differs')
    return True


def metric_records(native, labels, scores, validation):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    labels, scores, validation = np.asarray(labels), np.asarray(scores), np.asarray(validation)
    if scores.shape != labels.shape or not np.isin(labels, [0,1]).all() or not np.isfinite(scores).all() or not np.isfinite(validation).all():
        raise ValueError('Incomplete or nonfinite full score/label coverage')
    threshold = float(np.quantile(validation, .99))
    predicted = scores > threshold
    p, r, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    return {'paper_retrospective': {'AP':float(native.metric_PR(labels, scores)),
                                    'ROC':float(native.metric_ROC(labels, scores)) if len(np.unique(labels)) == 2 else None,
                                    'ROC_undefined_reason':None if len(np.unique(labels)) == 2 else 'one_label_class',
                                    'Best_F1':float(native.metric_PointF1(labels, scores)),
                                    'test_labels_used_for_threshold_optimization':True,'point_adjustment':False},
            'validation_only_control': {'precision':float(p),'recall':float(r),'f1':float(f1),
                                        'threshold':threshold,'quantile':.99,'threshold_comparator':'>',
                                        'test_features_used_for_calibration':False,'test_labels_used_for_calibration':False,
                                        'point_adjustment':False}}


def run(queue, job, root=ROOT):
    import numpy as np
    import torch
    manifest = verify(queue, root)
    if complete(job, root):
        return
    output = root / job['output_directory']
    if output.exists():
        raise FileExistsError('Partial outputs preserved; register recovery rather than overwrite')
    if environment_versions() != json.loads((root / queue['environment_lock']).read_text(encoding='utf-8')):
        raise ValueError('Frozen environment differs')
    entity = next(e for d in manifest['datasets'] if d['id'] == job['dataset'] for e in d['entities'] if e['id'] == job['entity'])
    if sha(root / entity['prepared_path']) != entity['prepared_sha256']:
        raise ValueError('Prepared entity differs')
    for source in entity['raw_sources']:
        if sha(root / source['path']) != source['sha256']:
            raise ValueError('Original raw source differs')
    with np.load(root / entity['prepared_path'], allow_pickle=False) as arrays:
        train, validation, test, labels = (arrays[k] for k in ('train','validation','test','labels'))
    sys.path.insert(0, str(root / 'src'))
    from iia_benchmark.models.grasp_protocol_v2 import GRASPProtocolDetector
    torch.set_num_threads(queue['settings']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    config = json.loads((root / job['model_config']).read_text(encoding='utf-8'))
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA required; no CPU or budget fallback')
    torch.cuda.reset_peak_memory_stats(0)
    output.mkdir(parents=True)
    artifacts = Path(queue['artifact_root']) / job['id']
    if artifacts.exists():
        raise FileExistsError('External partial artifacts preserved')
    artifacts.mkdir(parents=True)
    detector = GRASPProtocolDetector(**config['parameters'], seed=job['seed'])
    started = time.perf_counter()
    history_path = output / 'training_history.jsonl'
    def record_epoch(record):
        with history_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        write_json(output / 'progress.json', {'phase':'training', **record,'job_sha256':fingerprint(job),
                                             'elapsed_seconds':time.perf_counter()-started})
        if record['epoch'] == 1 or record['epoch'] % 50 == 0:
            print(json.dumps({'id':job['id'], **record}), flush=True)
    detector.fit(train, validation, callback=record_epoch)
    training_seconds = time.perf_counter()-started
    if detector.optimizer_updates_ != job['expected_optimizer_updates']:
        raise ValueError('Full training update count differs')
    checkpoint = artifacts / 'checkpoint.pt'
    torch.save({'model':detector.model.state_dict(),'path':detector.path.state_dict(),
                'parameters':config['parameters'],'seed':job['seed'],'scaler':detector.scaler_,
                'adjacency':detector.adjacency_,'graph_audit':detector.graph_audit_,
                'epochs_completed':detector.epochs_completed_,'optimizer_updates':detector.optimizer_updates_,
                'job_sha256':fingerprint(job),'entity_data_audit':entity}, checkpoint)
    spec = importlib.util.spec_from_file_location('grasp_native_metrics', root / queue['native_metrics'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.basic_metricor()
    receipts = [{'path':str(checkpoint),'sha256':sha(checkpoint)},
                {'path':history_path.relative_to(root).as_posix(),'sha256':sha(history_path)}]
    results = []
    for recipe in job['score_recipes']:
        write_json(output / 'progress.json', {'phase':'scoring','recipe':recipe,'job_sha256':fingerprint(job)})
        options = {'source_samples':recipe['source_samples'],'flow_evaluations':recipe['flow_evaluations']}
        start = time.perf_counter()
        validation_scores = detector.score(validation, split_tag=1, **options)
        validation_seconds = time.perf_counter()-start
        attribution = job['profile'] == 'main' and recipe == {'source_samples':5,'flow_evaluations':10}
        start = time.perf_counter()
        scored = detector.score(test, split_tag=2, return_details=attribution, **options)
        test_seconds = time.perf_counter()-start
        test_scores, details = scored if attribution else (scored, None)
        name = f"s{recipe['source_samples']}_t{recipe['flow_evaluations']}"
        score_path = artifacts / (name+'.npz')
        saved = {'test_scores':test_scores,'validation_scores':validation_scores,'labels':labels,
                 'test_row_ids':np.arange(len(test)),'validation_raw_row_ids':np.arange(entity['split_cut'],entity['raw_train_points'])}
        if details:
            for key, value in details.items():
                if isinstance(value, np.ndarray):
                    saved[key] = value
            saved['attribution_metadata_json'] = np.array(json.dumps({k:v for k,v in details.items() if not isinstance(v,np.ndarray)}))
        np.savez_compressed(score_path, **saved)
        receipts.append({'path':str(score_path),'sha256':sha(score_path)})
        results.append({'recipe':recipe,'metrics':metric_records(native, labels, test_scores, validation_scores),
                        'validation_score_seconds':validation_seconds,'test_score_seconds':test_seconds,
                        'window_evaluations':len(test)-detector.window+1,'spectral_decomposition_in_scoring':False,
                        'attribution_saved':attribution,'scores_path':str(score_path)})
    # Recheck all registered source and data files before completion is recorded.
    verify(queue, root)
    if environment_versions() != json.loads((root / queue['environment_lock']).read_text(encoding='utf-8')):
        raise ValueError('Environment changed during full experiment')
    if sha(root / entity['prepared_path']) != entity['prepared_sha256']:
        raise ValueError('Prepared source changed during run')
    for source in entity['raw_sources']:
        if sha(root / source['path']) != source['sha256']:
            raise ValueError('Raw source changed during run')
    result = {'status':'completed','diagnostic':False,'paper_equivalence_certified':False,
              'strict_TAB_result':False,'job':job,'job_sha256':fingerprint(job),
              'epochs_completed':detector.epochs_completed_,'optimizer_updates':detector.optimizer_updates_,
              'scored_points':len(test),'validation_points':len(validation),'score_recipes':job['score_recipes'],
              'parameter_count':sum(p.numel() for p in detector.model.parameters()),'training_seconds':training_seconds,
              'gpu':torch.cuda.get_device_name(0),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0),
              'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(0),'cpu_threads':queue['settings']['cpu_threads'],
              'checkpoint_selection':detector.checkpoint_selection_,'metrics':results,
              'data_audit':entity,'artifacts':receipts,'boundary':queue['boundary']}
    write_json(output / 'result.json', result)
    if not complete(job, root):
        raise ValueError('Completed artifact audit failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    queue = json.loads((ROOT / args.queue).read_text(encoding='utf-8'))
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(queue, job)


if __name__ == '__main__':
    main()
