"""Audit every original released-checkpoint score, label and operating point."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT,read,digest,complete_scores


def release_only(job,result):
    details=result['details']
    if (job['mode']!='release_checkpoint' or job['author_recipe']!='release_checkpoint'
            or result['mode']!='release_checkpoint' or result['status']!='completed'
            or result['id']!=job['id'] or result['seed']!=job['seed']
            or result['experiment_sha256']!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
            or result['paper_equivalence'] or result['strict_TAB_result']
            or details['epochs_completed']!=0 or details['training_forward_calls']!=0
            or details['training_seconds']!=0 or details['declared_train_parameters']!=job['train_parameters']
            or not details['algorithm_weights']['all_algorithm_weights_strictly_loaded']):
        raise ValueError('Complete fixed release evaluation only; never count it as a training seed')


def channel_mean_energy(values,expected):
    import numpy as np
    array=np.asarray(values)
    if array.ndim!=2 or array.shape[0]!=expected or not np.isfinite(array).all():
        raise ValueError('Complete native timestamp/channel energy matrix required')
    return array.mean(axis=-1)


def audit(config_path):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve,roc_auc_score,average_precision_score
    config=read(config_path)
    for item in config['source_receipts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen complete release audit source changed')
    queue=read(config['queue']);job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    for item in job['frozen_files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Original scientific source changed')
    output=ROOT/job['output_directory'];result=read(output/'result.json');release_only(job,result)
    for item in result['artifacts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Original complete release artifact changed')
    if digest(job['release_checkpoint'])!=result['details']['algorithm_weights']['checkpoint_sha256']:
        raise ValueError('Original published checkpoint weights changed')
    details=result['details'];usage=read(output/'resource_usage.json')
    if usage['resource_guard_aborted'] or details['parameter_count']!=config['full_model_parameters']:
        raise ValueError('Complete original architecture and nonaborted release execution required')
    raw=ROOT/queue['settings']['raw_root']/'data'/(job['dataset']+'.csv')
    if digest(raw)!=job['raw_sha256']:raise ValueError('Original complete raw data changed')
    labels=[]
    for frame in pd.read_csv(raw,usecols=['data','cols'],chunksize=65536):
        labels.extend(frame.loc[frame['cols']=='label','data'].tolist())
    labels=np.asarray(labels)
    if not np.isin(labels,[0,1]).all():raise ValueError('Every raw source label must be binary')
    labels=labels.astype(np.uint8);reviewed={}
    with np.load(output/'native_scores.npz',allow_pickle=False) as arrays:
        complete_scores(arrays,labels,job['train_length'],details['native_points'])
        y,score=arrays['labels'],arrays['test']
        strict=result['metrics']['native_validation_control']['strict']
        if set(strict)!={'0.1','1','5'}:raise ValueError('All original validation operating points required')
        for ratio,recorded in strict.items():
            threshold=float(np.percentile(np.asarray(arrays['validation'],dtype=np.float64),100-float(ratio),method='linear'))
            predicted=score>threshold;tp=int(np.sum(predicted&(y==1)));fp=int(np.sum(predicted&(y==0)));fn=int(np.sum(~predicted&(y==1)))
            values=dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,auroc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)))
            if (recorded['calibration']['threshold']!=threshold or recorded['point_adjustment']
                    or recorded['calibration']['calibration_split']!='validation'
                    or any(not math.isclose(recorded['micro'][k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in values.items())):
                raise ValueError('Every independent full-point validation metric must agree')
            reviewed[ratio]=dict(threshold=threshold,true_positives=tp,false_positives=fp,false_negatives=fn,metrics=values)
        count=details['original_scored_test_points'];directory=output/'workspace/test_results'/job['dataset']/'results_0'
        energy=channel_mean_energy(np.load(directory/'a_test_energy.npy',allow_pickle=False),count)
        original_labels=np.load(directory/'a_gt.npy',allow_pickle=False).reshape(-1)
        if (not np.array_equal(original_labels,y[:count]) or len(y)-count!=details['original_unscored_tail_points']
                or not np.allclose(energy,score[:count],rtol=1e-6,atol=1e-7)):
            raise ValueError('Every original nonoverlap test point and complete tail must agree')
        p,r,thresholds=precision_recall_curve(original_labels,energy)
        f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0);i=int(np.argmax(f))
        values=dict(P_best=float(p[i]),R_best=float(r[i]),F1_best=float(f[i]),
            ACC_best=float(np.mean((energy>thresholds[i])==original_labels)))
        for k,v in values.items():
            if not math.isclose(result['metrics']['original_author']['_best_f1'][0][k],v,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Complete author oracle PR curve and strict-threshold accuracy must agree')
        for field,value in [('AUC_ROC',roc_auc_score(original_labels,energy)),('AUC_PR',average_precision_score(original_labels,energy))]:
            if not math.isclose(result['metrics']['original_author']['_auc'][0][field],float(value),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Complete native author AUROC/AP must independently agree')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,job_id=job['id'],original_seed=job['seed'],
        fixed_release_checkpoint_evaluation=True,training_epochs_completed=0,completed_training_seed_slots=0,
        full_model_parameters=details['parameter_count'],full_native_points=details['native_points'],
        full_raw_labels_and_source_rows_checked=True,raw_sha256=job['raw_sha256'],
        original_unscored_tail_points=details['original_unscored_tail_points'],validation_only_point_metrics=reviewed,
        original_test_label_oracle_points_independently_checked=values,
        original_range_and_affiliation_artifacts_hash_checked_only=True,model_reinference_performed=False,
        original_source_result=result,original_scope=config['original_scope'],
        registered_required_release_evaluations=1,author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False)
    target=ROOT/config['output_directory'];target.mkdir(parents=True,exist_ok=False)
    (target/'independent_full_release_completion.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,job=job['id'],full_native_points=details['native_points'],completed_training_seeds=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);audit(parser.parse_args().config)
