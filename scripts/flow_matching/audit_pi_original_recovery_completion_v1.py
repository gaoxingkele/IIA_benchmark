"""Independently audit the complete original Pi seed, preserving recovery identity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def same_numbers(actual, expected):
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError('Complete metric keys differ')
        for key in actual:
            same_numbers(actual[key], expected[key])
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Independently computed original Pi metric differs')


def interval_pa(labels, raw):
    """Independent segment calculation retaining the original index-zero bug."""
    import numpy as np
    result = raw.copy()
    edges = np.diff(np.r_[0, labels, 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if raw[start:end].any():
            result[max(start, 1):end] = 1
    return result


def count_metrics(labels, predictions):
    import numpy as np
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    return dict(accuracy=(tp+tn)/len(labels), precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.), dict(tp=tp,fp=fp,fn=fn,tn=tn)


def binding(config):
    for item in config['source_receipts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen Pi recovery audit source changed: '+item['path'])
    recovery = read(config['recovery_queue'])
    case = next(c for c in recovery['jobs'] if c['job']['id'] == config['recovery_job_id'])
    if digest(case['original_queue']) != case['original_queue_sha256']:
        raise ValueError('Original full Pi queue changed')
    original = read(case['original_queue'])
    job = next(j for j in original['jobs'] if j['id'] == case['original_id'])
    if case['original_job'] != job:
        raise ValueError('Canonical original Pi job changed')
    if {k:v for k,v in case['job'].items() if k not in ('id','output_directory')} != {
            k:v for k,v in job.items() if k not in ('id','output_directory')}:
        raise ValueError('Full native Pi recovery must preserve all original settings')
    for item in job['frozen_files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Original Pi paper/source/raw data changed: '+item['path'])
    result = read(case['job']['output_directory']+'/result.json')
    expected = hashlib.sha256(json.dumps(case['job'],sort_keys=True).encode()).hexdigest()
    if result['status'] != 'completed' or result['experiment_sha256'] != expected:
        raise ValueError('Recovery completion fingerprint differs')
    if result['strict_TAB_result'] or not result['test_used_for_early_stopping'] or not result['test_scores_used_for_calibration']:
        raise ValueError('Test-selected historical mirror cannot be certified strict TAB')
    for item in result['artifacts']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Original Pi completed artifact changed')
    return case, job, result


def audit(config, config_path):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scripts.flow_matching.audit_pi_completed_native_v2 import full_budget
    from scripts.flow_matching.run_pi_transformer_job import score_controls
    import yaml
    torch.set_num_threads(config['cpu_threads'])
    case, job, result = binding(config)
    base = ROOT / case['job']['output_directory']
    effective = yaml.safe_load((base/'effective_author_config.yaml').read_text())
    if effective['datasets'][job['dataset']] != dict(job['author_config'],raw_directory=str(ROOT/job['raw_directory'])):
        raise ValueError('Effective full source configuration differs')
    if effective['general']['device'] != 'cpu':
        raise ValueError('Original Pi execution device changed')
    epochs = read(base/'checkpoints/epochs.json')
    budget = full_budget(job,result,epochs)
    source_config = yaml.safe_load((ROOT/job['runtime']/'config.yaml').read_text())
    if source_config['datasets'][job['dataset']] != job['author_config']:
        raise ValueError('Original full mirror hyperparameters differ')
    if result['torch_version'] != torch.__version__:
        raise ValueError('Preserve original numerical framework')
    with zipfile.ZipFile(base/'checkpoints/author_scores.npz') as archive:
        size = sum(a.file_size for a in archive.infolist())
    if size > config['maximum_uncompressed_score_bytes']:
        raise ValueError('Complete original arrays require bounded memory admission')
    with np.load(base/'checkpoints/author_scores.npz',allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    expected_keys = {'train_energy','train_phase','fused_train','test_energy','labels','raw_predictions',
                     'pa_predictions','reconstruction_train','reconstruction_test','mismatch_test',
                     'threshold','med_E','iqr_E','med_P','iqr_P'}
    if set(values) != expected_keys or any(not np.isfinite(x).all() for x in values.values()):
        raise ValueError('Complete finite original Pi streams required')
    window = job['author_config']['data']['win_size']
    test = job['data_audit']['expected_shapes']['test'][0]
    scored = test//window*window
    for key in ('train_energy','train_phase','fused_train','reconstruction_train'):
        if values[key].shape != (budget['training_windows']*window,):
            raise ValueError('All overlapping Pi training scores required')
    for key in ('test_energy','labels','raw_predictions','pa_predictions','reconstruction_test','mismatch_test'):
        if values[key].shape != (scored,):
            raise ValueError('All original nonoverlap Pi test scores required')
    raw = ROOT/job['raw_directory']
    for role in ('train','test'):
        if list(pd.read_csv(raw/(role+'.csv')).iloc[:,1:].shape) != job['data_audit']['expected_shapes'][role]:
            raise ValueError('Native full raw data geometry differs')
    native_labels = pd.read_csv(raw/'test_label.csv').iloc[:,1:].to_numpy().reshape(-1)
    if len(native_labels) != test or not np.isin(native_labels,[0,1]).all() or not np.array_equal(native_labels[:scored],values['labels']):
        raise ValueError('Full source labels and exact original coverage differ')
    if result['native_test_points'] != test or result['scored_test_points'] != scored or result['dropped_test_tail_points'] != test-scored:
        raise ValueError('Original dropped inference tail must be declared exactly')
    medians = {}; iqrs = {}
    for name, key in [('E','train_energy'),('P','train_phase')]:
        medians[name] = np.percentile(values[key],50)
        iqrs[name] = max(np.percentile(values[key],75)-np.percentile(values[key],25),1e-8)
        if medians[name] != values['med_'+name] or iqrs[name] != values['iqr_'+name]:
            raise ValueError('Native train-only median/IQR normalization differs')
    fused = np.maximum(np.maximum(0,(values['train_energy']-medians['E'])/iqrs['E']),
                       np.maximum(0,(values['train_phase']-medians['P'])/iqrs['P']))
    if not np.array_equal(fused,values['fused_train']):
        raise ValueError('Full original normalized train scores differ')
    threshold = np.percentile(np.concatenate([fused,values['test_energy']]),100-job['author_config']['data']['anomaly_ratio'])
    if threshold != values['threshold'] or not np.array_equal(values['raw_predictions'],(values['test_energy']>threshold).astype(int)):
        raise ValueError('Exact source pooled test calibration differs')
    if not np.array_equal(interval_pa(values['labels'],values['raw_predictions']),values['pa_predictions']):
        raise ValueError('Independent original interval PA differs')
    metrics, adjusted_counts = count_metrics(values['labels'],values['pa_predictions'])
    no_pa, raw_counts = count_metrics(values['labels'],values['raw_predictions'])
    no_pa.update(auroc=float(roc_auc_score(values['labels'],values['test_energy'])),
                 average_precision=float(average_precision_score(values['labels'],values['test_energy'])))
    metrics['pointwise_no_PA_same_author_threshold'] = no_pa
    same_numbers(metrics,result['metrics'])
    controls = score_controls(values,job['author_config']['testing']['temperature'],job['author_config']['data']['anomaly_ratio'],
                              window,case['settings']['score_temperature_multipliers'],case['settings']['score_streams'])
    saved_controls = read(base/'posthoc_score_controls.json')['records']
    if controls != saved_controls or len(controls) != 15:
        raise ValueError('All original 15 posthoc controls must reproduce without selecting on test labels')
    usage = read(base/'resource_usage.json')
    if usage.get('resource_guard_aborted'):
        raise ValueError('A resource-aborted run cannot be a full completion')
    proof = dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,config=config_path,config_sha256=digest(config_path),
        original_id=job['id'],recovery_id=case['job']['id'],seed=job['seed'],branch=job['branch'],dataset=job['dataset'],
        recipe=job['recipe'],original_required_training_seeds=case['settings']['seeds'],completed_original_training_seed_slots=1,
        retries_are_not_additional_training_seeds=True,full_native_budget=budget,original_parameters=result['parameter_count'],
        actual_active_layers=result['active_attention_layers'],full_original_raw_labels_verified=True,
        scored_test_points=scored,dropped_test_tail_points=test-scored,source_threshold=float(threshold),
        independently_recomputed_metrics=metrics,confusion_no_PA=raw_counts,confusion_author_PA=adjusted_counts,
        original_full_score_shapes={k:list(v.shape) for k,v in values.items()},full_score_uncompressed_bytes=size,
        all_posthoc_controls_recomputed=15,original_source_and_artifacts_verified=True,
        result_path=(base/'result.json').relative_to(ROOT).as_posix(),result_sha256=digest(base/'result.json'),
        full_native_checkpoint_reinference_passed=False,strict_TAB_result=False,author_equivalence_certified=False,
        all_original_experiments_complete=False,resource_usage=usage,boundary=config['boundary'])
    target = ROOT/config['output_directory']
    target.mkdir(parents=True,exist_ok=False)
    (target/'independent_full_recovery_completion.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,original_id=job['id'],metrics=metrics,full_model_reinference_passed=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    audit(read(args.config),args.config)


if __name__ == '__main__':
    main()
