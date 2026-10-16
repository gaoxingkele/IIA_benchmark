"""Full frozen-score PA/calibration controls; no model fitting or inference.

Recompute all original ten test-pool diagnostic ratios independently and add
same-threshold validation PA controls without changing any primary threshold.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as saved
from scripts.flow_matching.audit_crossad_full139_independent_metrics_v1 import percentile,equal_fields
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.audit_industrial_full466_PA_calibration_controls_v1'
TEST='tests/test_fm_industrial_full466_PA_calibration_controls_v1.py'
CONFIG='configs/reproducibility/fm_industrial_full466_PA_calibration_controls.v1.json'
PACKET='projects/flow_matching_research/results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1'
RATIOS=[.1,.5,1,2,3,5,10,15,20,25]


def registration():
    oc=read(ROOT/saved.CONFIG)
    paths=[MODULE.replace('.','/')+'.py',TEST,saved.CONFIG,saved.MODULE.replace('.','/')+'.py',
        PACKET+'/packet_integrity.json',PACKET+'/all466_proof_source_and_guard_bindings.json',
        'scripts/flow_matching/audit_crossad_full139_independent_metrics_v1.py']
    return dict(source_score_audit_packet=PACKET,required_complete_original_CPU_conditions=466,
        required_full_original_CPU_scope=550,required_full_original_GPU_scope=850,original_global_scope=oc['original_global_scope'],
        original_score_audit_configuration=saved.CONFIG,settings=oc['settings'],resource_lock=oc['resource_lock'],
        state_root='experiments/runs/fm_industrial_full466_PA_calibration_controls_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_full466_PA_calibration_controls_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        strict_validation_alarm_percent=[.1,1,5],test_pool_diagnostic_alarm_percent=RATIOS,
        numerical_metric_tolerance=oc['numerical_metric_tolerance'],test_label_selected_best_is_diagnostic_only=True,
        original_primary_threshold_and_score_direction_unchanged=True,new_training_seed_slots=0,new_optimizer_updates=0,
        new_neural_forwards=0,new_complete_model_results=0,full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Complete frozen scope, all thirteen threshold settings and unchanged primary rules required')
    bindings=read(ROOT/PACKET/'all466_proof_source_and_guard_bindings.json')['records']
    if len(bindings)!=466:raise ValueError('Every previously verified original condition remains required')
    q,manifest=saved.verify(read(ROOT/saved.CONFIG))
    return bindings,q,manifest


def measures(tp,fp,fn):
    return dict(true_positives=tp,false_positives=fp,false_negatives=fn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def paired_counts(entities,threshold):
    """Credit detected truth runs only inside each original entity boundary."""
    if not math.isfinite(threshold) or not entities:raise ValueError('Finite fixed threshold and original entities required')
    tp=fp=positive=adjusted_tp=events=detected=0
    for scores,labels in entities:
        if len(scores)!=len(labels) or not scores:raise ValueError('Full aligned entity scores and labels required')
        length=0;hit=False
        for score,label in zip(scores,labels):
            if not math.isfinite(score) or label not in (0,1):raise ValueError('Finite scores and binary truth required')
            label=int(label);prediction=bool(score>threshold)
            tp+=bool(prediction and label);fp+=bool(prediction and not label);positive+=label
            if label:
                length+=1;hit=hit or prediction
            elif length:
                events+=1;detected+=hit;adjusted_tp+=length if hit else 0;length=0;hit=False
        if length:
            events+=1;detected+=hit;adjusted_tp+=length if hit else 0
    return dict(no_PA=measures(tp,fp,positive-tp),with_PA=measures(adjusted_tp,fp,positive-adjusted_tp),
        ground_truth_events=events,detected_events=detected,PA_uses_test_ground_truth_events=True)


def audit(c,bindings,q,manifest,cid):
    binding=bindings[cid]
    source_path=ROOT/binding['source_result_path'];source=read(source_path);job=source['job']
    if (source['id']!=binding['actual_execution_id'] or sha(source_path)!=binding['source_result_sha256']
            or source['scores_sha256']!=binding['scores_sha256'] or sha(source_path.parent/'scores.npz')!=binding['scores_sha256']):
        raise ValueError('Complete already-verified original score execution identity required')
    if binding['separate_recovery']:
        path=Path(binding['independent_report_container'])
        if sha(path)!=binding['independent_report_container_sha256']:raise ValueError('Immutable recovery proof changed')
        previous=read(path)['records'][cid]['independent_report']
    else:
        path=Path(binding['independent_report_path'])
        if sha(path)!=binding['independent_report_sha256']:raise ValueError('Immutable full score audit changed')
        previous=read(path)
    dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    payload={}
    with zipfile.ZipFile(source_path.parent/'scores.npz') as archive:
        for role,g in saved.roles(dataset):
            name=g['array_prefix']+'_'+role
            h,b=saved.archive_array(archive,name)
            payload[name]=saved.vector(h,b,g['points'],'<f8')
            if hashlib.sha256(b).hexdigest()!=previous['all_array_bindings'][name]['sha256']:raise ValueError('Full score payload changed')
            if role=='test':
                name=g['array_prefix']+'_labels';h,b=saved.archive_array(archive,name)
                payload[name]=saved.vector(h,b,g['test_points'],'|u1')
                if hashlib.sha256(b).hexdigest()!=previous['all_array_bindings'][name]['sha256']:raise ValueError('Full original test labels changed')
    entities=[(payload[g['array_prefix']+'_test'],payload[g['array_prefix']+'_labels']) for g in dataset['entities']]
    validation=sorted(x for g in dataset['validation_groups'] for x in payload[g['array_prefix']+'_validation'])
    pool=sorted([x for g in dataset['training_groups'] for x in payload[g['array_prefix']+'_train']]+[x for s,y in entities for x in s])
    strict={}
    for ratio in c['strict_validation_alarm_percent']:
        key=str(ratio);threshold=percentile(validation,ratio)
        control=paired_counts(entities,threshold)
        equal_fields(dict(threshold=threshold),source['metrics']['strict'][key]['calibration'],c['numerical_metric_tolerance'])
        equal_fields(control['no_PA'],previous['independent_strict_metrics'][key]['micro'],c['numerical_metric_tolerance'])
        strict[key]=dict(threshold=threshold,threshold_split='validation',threshold_uses_test_scores=False,
            threshold_selected_using_test_labels=False,**control)
    grid=source['metrics']['tab_core_diagnostic']['ratio_grid']
    if [r['ratio_percent'] for r in grid]!=RATIOS:raise ValueError('All ten original diagnostic ratios in frozen order required')
    diagnostic=[]
    for ratio,old in zip(RATIOS,grid):
        threshold=percentile(pool,ratio);control=paired_counts(entities,threshold)
        equal_fields(dict(threshold=threshold,f_score=control['no_PA']['f1'],adjust_f_score=control['with_PA']['f1']),old,c['numerical_metric_tolerance'])
        diagnostic.append(dict(ratio_percent=ratio,threshold=threshold,threshold_split='pooled_train_and_test_scores',
            threshold_uses_test_scores=True,threshold_selected_using_test_labels=False,primary_performance=False,**control))
    best_plain=max(r['no_PA']['f1'] for r in diagnostic);best_PA=max(r['with_PA']['f1'] for r in diagnostic)
    equal_fields(dict(best_grid_f_score=best_plain,best_grid_adjust_f_score=best_PA),source['metrics']['tab_core_diagnostic'],c['numerical_metric_tolerance'])
    return dict(passed=True,canonical_id=cid,actual_execution_id=job['id'],dataset=job['dataset'],seed=job['seed'],model_config=job['model_config'],
        source_result_sha256=sha(source_path),source_score_archive_sha256=binding['scores_sha256'],
        original_full_test_points=sum(len(y) for s,y in entities),original_test_entities=len(entities),
        strict_validation_controls=strict,all_ten_test_pool_diagnostic_controls=diagnostic,
        test_label_selected_best_diagnostic=dict(no_PA_f1=best_plain,with_PA_f1=best_PA,test_labels_used_for_selection=True,primary_performance=False),
        source_all_ten_diagnostic_threshold_and_PA_formulas_independently_verified=True,
        original_primary_threshold_and_score_direction_unchanged=True,new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,new_complete_model_results=0,
        full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--case')
    args=parser.parse_args()
    if args.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);bindings,q,manifest=verify(c);target=Path(c['artifact_root'])
    if args.probe_only:
        print(json.dumps(dict(passed=True,required_conditions=len(bindings),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,numerical_workers_started=0)));return
    if args.case:
        report=audit(c,bindings,q,manifest,args.case);path=target/'conditions'/(args.case+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
        return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Complete PA/calibration audit already has an owner')
    if (base/'status.json').exists():raise RuntimeError('Preserve previous terminal attempts; no automatic retry')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,verified_ids=[],failed_preserved={},
        required_original_conditions=466,all_original_experiments_complete=False)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='independent_full_score_PA_and_calibration_controls'
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        for cid in sorted(bindings):
            with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                attempt=base/'attempts'/cid;attempt.mkdir(parents=True,exist_ok=False)
                command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--case',cid]
                with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    update(dict(state='independent_full_score_PA_and_calibration_controls',active_pid=child.pid,
                        active_create_time=psutil.Process(child.pid).create_time(),active_condition=cid,actual_command=command))
                    code,usage=capped_wait(child,c,api,update)
                proof=target/'conditions'/(cid+'.json')
                passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                receipt=dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None)
                write(attempt/'receipt.json',receipt)
                if passed:state['verified_ids'].append(cid)
                else:state['failed_preserved'][cid]=receipt
                update(dict(state='waiting_next_original_condition',active_pid=None))
        update(dict(state='all466_control_attempts_terminal',active_pid=None))
    finally:owner.close()


if __name__=='__main__':main()
