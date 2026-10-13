"""Frozen calibration/PA controls on complete independently replayed Pi models.

Use the same original saved scores: pooled train/test or train-only percentile,
each with/without original label-dependent PA. No optimizer, model inference,
checkpoint selection or threshold tuning occurs here. These forensic controls
cannot remove earlier test-selected early stopping or certify strict TAB.
Paper/data citations remain in each unchanged canonical full job config.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile

from scripts.flow_matching import verify_pi_original_full_replay_completion_v1 as review
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write

ROOT = review.ROOT


def protocol_controls(train, test, labels, anomaly_ratio):
    import numpy as np
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
    if (np.asarray(train).ndim != 1 or np.asarray(test).ndim != 1
            or len(labels) != len(test) or not len(train) or not len(test)
            or not np.isfinite(train).all() or not np.isfinite(test).all()
            or not np.isin(labels,[0,1]).all() or not 0 < anomaly_ratio < 100):
        raise ValueError('Original complete finite streams and binary labels required')
    thresholds = dict(author_pooled_train_test=float(np.percentile(np.concatenate([train,test]),100-anomaly_ratio)),
                      train_only_same_percentile=float(np.percentile(train,100-anomaly_ratio)))
    ranking = dict(auroc=float(roc_auc_score(labels,test)),average_precision=float(average_precision_score(labels,test)))
    rows = []
    for calibration, threshold in thresholds.items():
        raw = (test > threshold).astype(int)
        for adjusted in [False,True]:
            predictions = review.interval_pa(labels,raw) if adjusted else raw
            metrics,counts = review.count_metrics(labels,predictions)
            tn,fp,fn,tp = confusion_matrix(labels,predictions,labels=[0,1]).ravel().tolist()
            if counts != dict(tp=tp,fp=fp,fn=fn,tn=tn) or abs(metrics['f1']-float(f1_score(labels,predictions)))>1e-12:
                raise ValueError('Independent confusion/F1 calculation differs')
            rows.append(dict(calibration=calibration,label_dependent_PA=adjusted,threshold=threshold,
                             metrics=metrics,confusion=counts,alert_count=int(predictions.sum()),
                             test_score_distribution_used_for_threshold=calibration=='author_pooled_train_test',
                             test_labels_used_for_point_adjustment=adjusted,strict_TAB_result=False))
    return dict(thresholds=thresholds,ranking_metrics=ranking,controls=rows)


def audit(config):
    import numpy as np
    for pin in config['source_receipts']:
        if review.digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen actual completed-model evidence changed: '+pin['path'])
    queue = review.read(config['original_queue'])
    if len(queue['jobs']) != 1140 or queue['settings']['seeds'] != [42,1103,1104,1105,1106,1107]:
        raise ValueError('Full1140 original experiment and six-seed scope required')
    rows = []
    identities = set()
    for item in config['closed_cases']:
        proof = review.read(item['independent_proof'])
        registered = review.read(proof['configuration'])
        producer,completion,native,receipt,case,job,result = review.verify_registration(registered)
        independent_receipt = review.read(registered['state_root']+'/receipt.json')
        if (proof['canonical_id'] != item['canonical_id'] or job['id'] != item['canonical_id']
                or not proof['passed'] or independent_receipt['exit_code'] != 0
                or not independent_receipt['passed'] or independent_receipt['resource_guard_aborted']
                or proof['configuration_sha256'] != review.digest(proof['configuration'])
                or proof['completed_native_replay_report_sha256'] != review.digest(registered['completed_replay_report'])
                or proof['canonical_id'] in identities or job['dataset'] != 'PSM' or job['recipe'] != 'full'
                or proof['original_parameters'] != 1930289 or proof['full_native_budget']['epochs_completed'] != 3
                or proof['full_native_budget']['optimizer_minibatches'] != 1554
                or proof['strict_TAB_result'] or proof['author_equivalence_certified']):
            raise ValueError('Only unique actual guarded full-model/native/array certificates required')
        identities.add(proof['canonical_id'])
        source_path = ROOT / case['job']['output_directory'] / 'checkpoints/author_scores.npz'
        if review.digest(source_path) != proof['original_score_sha256']:
            raise ValueError('Every original source score archive byte must remain')
        with zipfile.ZipFile(source_path) as archive:
            if sum(a.file_size for a in archive.infolist()) > registered['maximum_uncompressed_score_bytes']:
                raise ValueError('Original full archive memory guard required')
        with np.load(source_path,allow_pickle=False) as arrays:
            if set(arrays.files) != set(proof['independently_rechecked_streams']):
                raise ValueError('Every original15 score stream must remain')
            train,test,labels = arrays['fused_train'],arrays['test_energy'],arrays['labels']
            controls = protocol_controls(train,test,labels,job['author_config']['data']['anomaly_ratio'])
            author_raw = next(r for r in controls['controls'] if r['calibration']=='author_pooled_train_test' and not r['label_dependent_PA'])
            author_PA = next(r for r in controls['controls'] if r['calibration']=='author_pooled_train_test' and r['label_dependent_PA'])
            if (author_raw['threshold'] != float(arrays['threshold'])
                    or author_raw['threshold'] != proof['source_threshold']
                    or not np.array_equal(arrays['raw_predictions'],(test>author_raw['threshold']).astype(int))
                    or not np.array_equal(arrays['pa_predictions'],review.interval_pa(labels,arrays['raw_predictions']))
                    or author_raw['confusion'] != proof['confusion_no_PA']
                    or author_PA['confusion'] != proof['confusion_author_PA']):
                raise ValueError('Original full saved predictions and independently closed metric counts must match')
            original = dict(author_PA['metrics'],pointwise_no_PA_same_author_threshold=dict(author_raw['metrics'],**controls['ranking_metrics']))
            review.same_numbers(original,proof['independently_recomputed_metrics'])
            additional_tp = author_PA['confusion']['tp']-author_raw['confusion']['tp']
            row = dict(canonical_id=job['id'],seed=job['seed'],branch=job['branch'],dataset='PSM',
                       original_full_native_budget=proof['full_native_budget'],original_parameters=1930289,
                       full_train_score_coordinates=len(train),full_scored_test_coordinates=len(test),
                       original_dropped_test_tail_points=completion['dropped_test_tail_points'],
                       original_anomaly_ratio_percent=job['author_config']['data']['anomaly_ratio'],
                       **controls,PA_added_true_positive_points=additional_tp,
                       PA_added_fraction_of_reported_true_positives=additional_tp/author_PA['confusion']['tp'],
                       checkpoint_sha256=native['original_checkpoint_sha256'],
                       source_score_sha256=proof['original_score_sha256'],closed_proof_sha256=review.digest(item['independent_proof']),
                       source_test_selected_early_stopping=result['test_used_for_early_stopping'],
                       original_training_resource_evidence_available=completion.get('original_training_resource_evidence_available',False),
                       native_full_model_and15stream_independent_replay_verified=True,strict_TAB_result=False)
            rows.append(row)
    groups = []
    for branch in sorted({r['branch'] for r in rows}):
        members = [r for r in rows if r['branch']==branch]
        if len({r['checkpoint_sha256'] for r in members}) != len(members):
            raise ValueError('Distinct training seeds require distinct actual model checkpoints')
        for calibration in ['author_pooled_train_test','train_only_same_percentile']:
            for PA in [False,True]:
                values = [next(c for c in row['controls'] if c['calibration']==calibration and c['label_dependent_PA']==PA)['metrics']['f1'] for row in members]
                groups.append(dict(branch=branch,seeds=[r['seed'] for r in members],n=len(values),required_original_seeds=queue['settings']['seeds'],
                                   calibration=calibration,label_dependent_PA=PA,F1_mean=float(np.mean(values)),
                                   F1_min=min(values),F1_max=max(values),F1_sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                   full_seed_cohort_complete=False,confidence_interval_certified=False))
    return dict(captured_utc=datetime.now(timezone.utc).isoformat(),records=rows,branch_seed_protocol_summaries=groups,
                original_required_jobs=1140,actual_closed_models_audited=len(rows),all_original_jobs_complete=False,
                original_scope=config['original_scope'],extra_model_inference=0,extra_optimizer_updates=0,
                threshold_percentile_frozen_to_original_author_anomaly_ratio=True,
                test_labels_or_metrics_used_to_select_percentile=False,
                checkpoint_still_test_selected=True,strict_TAB_result=False,author_equivalence_certified=False,
                checkpoint_train_only_calibration_does_not_remove_prior_test_selection=True,
                conclusions_limited_to_registered_PSM_branches_and_closed_seeds=True,
                all_original_experiments_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=review.read(args.config)
    destination=ROOT/config['output_report']
    if destination.exists():raise FileExistsError('Preserve frozen calibration controls')
    report=audit(config);write(destination,report)
    print(dict(closed_models=len(report['records']),protocol_summaries=report['branch_seed_protocol_summaries']))


if __name__=='__main__':main()
