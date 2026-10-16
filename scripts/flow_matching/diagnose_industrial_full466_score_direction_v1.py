"""Prespecified high/low-tail controls on all466 complete frozen score cases.

Both orientations are reported. Test labels only measure outcomes; they never
select a direction, threshold, checkpoint, primary result or preferred method.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import zipfile

from scripts.flow_matching import replay_industrial_full550_checkpoints_v1 as native
from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as saved
from scripts.flow_matching.audit_crossad_full139_independent_metrics_v1 import percentile,ranking,at_threshold,equal_fields
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.diagnose_industrial_full466_score_direction_v1'
TEST='tests/test_fm_industrial_full466_score_direction_v1.py'
CONFIG='configs/reproducibility/fm_industrial_full466_score_direction.v1.json'
RATIOS=['0.1','1','5']
DIRECTIONS=[1,-1]
METRICS=['precision','recall','f1','auroc','average_precision','macro_f1','normal_false_positive_fraction']


def registration():
    pc=read(ROOT/saved.CONFIG)
    pins=[MODULE.replace('.','/')+'.py',TEST,native.CONFIG,native.MODULE.replace('.','/')+'.py',saved.CONFIG,
        saved.MODULE.replace('.','/')+'.py',native.PRIOR+'/all466_proof_source_and_guard_bindings.json',
        native.PRIOR+'/packet_integrity.json','scripts/flow_matching/audit_crossad_full139_independent_metrics_v1.py']
    return dict(required_complete_source_cases=466,required_original_CPU_cases=550,required_original_GPU_cases=850,
        original_scope=pc['original_global_scope'],source_native_configuration=native.CONFIG,
        state_root='experiments/runs/fm_industrial_full466_score_direction_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_full466_score_direction_v1',
        settings=pc['settings'],resource_lock=pc['resource_lock'],directions=DIRECTIONS,validation_alarm_percent=RATIOS,
        numerical_metric_tolerance=pc['numerical_metric_tolerance'],paired_metrics=METRICS,
        required_training_seeds=[1103,1104,1105,1106,1107],required_original_groups=110,
        complete_groups=89,partial_groups=6,empty_groups=15,
        direction_selected_using_test_labels=False,both_orientations_must_be_reported=True,
        opposite_orientation_primary=False,original_primary_scores_and_thresholds_unchanged=True,
        training_optimizer_updates=0,neural_forwards=0,new_training_seed_slots=0,new_model_training_results=0,
        native_checkpoint_replay=False,full_TAB_equivalence=False,paper_equivalence=False,
        citations=pc['citations'],source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('All466sources,predeclared both directions,all thresholds and unchanged primary rules required')
    q,manifest,jobs,executions=native.verify(read(ROOT/native.CONFIG))
    bindings=read(ROOT/native.PRIOR/'all466_proof_source_and_guard_bindings.json')['records']
    if len(bindings)!=466:raise ValueError('Complete fixed source snapshot required')
    return q,manifest,jobs,executions,bindings


def vector_hash(values):
    return hashlib.sha256(struct.pack('<'+'d'*len(values),*values)).hexdigest()


def controls(validation,entities):
    """Pure counts/ranks on every entity; no sign selection or PA."""
    if not validation or not entities or any(not math.isfinite(x) for x in validation):
        raise ValueError('Complete finite validation scores and original entities required')
    if len({name for name,s,y in entities})!=len(entities):raise ValueError('Distinct native entity boundaries required')
    for name,scores,labels in entities:
        if (not scores or len(scores)!=len(labels) or not all(math.isfinite(x) for x in scores)
                or any(y not in (0,1) for y in labels)):
            raise ValueError('Full finite entity scores and aligned binary labels required')
    rows={};ranks={}
    for sign in DIRECTIONS:
        val=[sign*x for x in validation];ordered=sorted(val)
        transformed=[(name,[sign*x for x in scores],labels) for name,scores,labels in entities]
        pooled_scores=[s for name,scores,labels in transformed for s in scores]
        pooled_labels=[int(y) for name,scores,labels in transformed for y in labels]
        pooled_ranks=ranking(pooled_scores,pooled_labels)
        per_ranks={name:ranking(scores,labels) for name,scores,labels in transformed};ranks[sign]=pooled_ranks
        for ratio in RATIOS:
            threshold=percentile(ordered,float(ratio))
            if not math.isfinite(threshold):raise ValueError('Finite complete validation threshold required')
            pooled=at_threshold(pooled_scores,pooled_labels,threshold)['micro']
            pooled.update(pooled_ranks);per_entity={};events=[]
            for name,scores,labels in transformed:
                measured=at_threshold(scores,labels,threshold);measured['micro'].update(per_ranks[name])
                per_entity[name]=measured;events.append(measured['event_metrics'])
            normal=len(pooled_labels)-sum(pooled_labels);total_events=sum(e['events'] for e in events)
            hits=sum(e['detected_events'] for e in events)
            values=dict(**pooled,macro_f1=statistics.mean(e['micro']['f1'] for e in per_entity.values()),
                normal_false_positive_fraction=pooled['false_positives']/normal if normal else None)
            row=dict(score_multiplier=sign,validation_alarm_percent=ratio,transformed_score_threshold=threshold,
                equivalent_original_score_threshold=sign*threshold,
                original_score_comparison='strict_greater' if sign==1 else 'strict_less',
                calibration_split='validation',calibration_points=len(val),calibration_score_payload_sha256=vector_hash(val),
                threshold_uses_test_scores=False,direction_selected_using_test_labels=False,point_adjustment=False,
                actual_primary_evaluation=sign==1 and ratio=='1',opposite_orientation_primary=False,
                micro_and_macro=values,per_entity=per_entity,
                event_metrics=dict(events=total_events,detected_events=hits,event_recall=hits/total_events if total_events else None,
                    mean_detection_delay_points=math.fsum(e['mean_detection_delay_points']*e['detected_events'] for e in events if e['detected_events'])/hits if hits else None,
                    false_alarms_per_1000_normal_points=1000*pooled['false_positives']/normal if normal else None))
            rows[sign,ratio]=row
    a,b=ranks[1]['auroc'],ranks[-1]['auroc']
    if (a is None)!=(b is None) or a is not None and not math.isclose(a+b,1.,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Full tied-rank AUROC must obey the exact opposite-direction identity')
    return rows


def paired_summary(source,opposite):
    if len(source)!=5 or len(opposite)!=5:raise ValueError('Only all five paired model seeds may be summarized')
    if any(x is None for x in source+opposite):
        return dict(n=5,status='undefined_some_paired_seeds',source_mean=None,opposite_mean=None,paired_difference_mean=None,
            paired_difference_sample_SD=None,paired_difference_SE=None,paired_difference_t95_low=None,paired_difference_t95_high=None)
    if any(not math.isfinite(x) for x in source+opposite):raise ValueError('Finite paired metrics required')
    differences=[b-a for a,b in zip(source,opposite)];mean=statistics.mean(differences);sd=statistics.stdev(differences)
    se=sd/math.sqrt(5);half=2.7764451051977987*se
    return dict(n=5,status='complete_five_seed_paired_diagnostic',source_mean=statistics.mean(source),opposite_mean=statistics.mean(opposite),
        paired_difference_mean=mean,paired_difference_sample_SD=sd,paired_difference_SE=se,
        paired_difference_t95_low=mean-half,paired_difference_t95_high=mean+half)


def case(c,manifest,jobs,executions,cid):
    ready=native.source_binding(read(ROOT/native.CONFIG),jobs,executions,cid)
    if ready is None:raise ValueError('All actual source training and independent score prerequisites required')
    job,source,previous,binding=ready;dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    if sha(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Complete original data changed')
    payload={};arrays={}
    with zipfile.ZipFile(ROOT/job['output_directory']/'scores.npz') as archive:
        for role,g in saved.roles(dataset):
            name=g['array_prefix']+'_'+role;h,b=saved.archive_array(archive,name)
            if hashlib.sha256(b).hexdigest()!=previous['all_array_bindings'][name]['sha256']:raise ValueError('Full score coordinate payload changed')
            payload[name]=saved.vector(h,b,g['points'],'<f8');arrays[name]=previous['all_array_bindings'][name]
            if role=='test':
                name=g['array_prefix']+'_labels';h,b=saved.archive_array(archive,name)
                if hashlib.sha256(b).hexdigest()!=previous['all_array_bindings'][name]['sha256']:raise ValueError('Complete native test labels changed')
                payload[name]=saved.vector(h,b,g['test_points'],'|u1');arrays[name]=previous['all_array_bindings'][name]
    validation=[x for g in dataset['validation_groups'] for x in payload[g['array_prefix']+'_validation']]
    entities=[(g['entity_id'],payload[g['array_prefix']+'_test'],payload[g['array_prefix']+'_labels']) for g in dataset['entities']]
    measured=controls(validation,entities)
    for ratio in RATIOS:
        actual=measured[1,ratio];old=previous['independent_strict_metrics'][ratio]
        equal_fields(dict(threshold=actual['transformed_score_threshold']),old,c['numerical_metric_tolerance'])
        equal_fields(old['micro'],actual['micro_and_macro'],c['numerical_metric_tolerance'])
        equal_fields(actual['event_metrics'],old['event_metrics'],c['numerical_metric_tolerance'])
        equal_fields(dict(macro_f1=actual['micro_and_macro']['macro_f1']),old,c['numerical_metric_tolerance'])
    return dict(passed=True,canonical_id=cid,actual_execution_id=job['id'],dataset=job['dataset'],model_config=job['model_config'],seed=job['seed'],
        complete_source_binding=binding,all_original_score_and_label_array_bindings=arrays,
        full_native_test_points=sum(len(y) for name,s,y in entities),full_native_entities=len(entities),
        controls=[measured[sign,ratio] for sign in DIRECTIONS for ratio in RATIOS],
        original_primary_scores_thresholds_and_direction_unchanged=True,direction_selected_using_test_labels=False,
        opposite_direction_is_diagnostic_only=True,source_checkpoint_hashed_not_neurally_replayed=True,
        new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,new_model_results=0,
        all_original_experiments_complete=False)


def worker(c):
    q,manifest,jobs,executions,bindings=verify(c);out=Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False)
    records={};flat=[];entities=[]
    for i,cid in enumerate(bindings):
        report=case(c,manifest,jobs,executions,cid);path=out/'conditions'/(cid+'.json');write(path,report)
        records[cid]=dict(path=str(path),sha256=sha(path))
        for r in report['controls']:
            identity=dict(canonical_id=cid,actual_execution_id=report['actual_execution_id'],model_config=report['model_config'],
                dataset=report['dataset'],seed=report['seed'],score_multiplier=r['score_multiplier'],
                validation_alarm_percent=r['validation_alarm_percent'],actual_primary_evaluation=r['actual_primary_evaluation'],
                opposite_orientation_primary=False,direction_selected_using_test_labels=False)
            flat.append(dict(**identity,transformed_score_threshold=r['transformed_score_threshold'],
                equivalent_original_score_threshold=r['equivalent_original_score_threshold'],
                original_score_comparison=r['original_score_comparison'],**r['micro_and_macro'],
                **r['event_metrics']))
            for name,e in r['per_entity'].items():entities.append(dict(**identity,entity=name,**e['micro'],**e['event_metrics']))
        write(out/'progress.json',dict(verified_sources=i+1,required_complete_sources=466,new_neural_forwards=0,new_optimizer_updates=0))
    groups=defaultdict(list)
    for j in q['jobs']:groups[j['model_config'],j['dataset']].append(j)
    index={(r['canonical_id'],r['score_multiplier'],r['validation_alarm_percent']):r for r in flat}
    summaries=[];group_status=[]
    for (model,dataset),required in sorted(groups.items()):
        if sorted(j['seed'] for j in required)!=c['required_training_seeds']:raise ValueError('Original complete seed roster required')
        done=[j for j in required if j['id'] in records]
        group_status.append(dict(model_config=model,dataset=dataset,completed_seeds=[j['seed'] for j in done],required_seeds=c['required_training_seeds'],
            complete=len(done)==5,partial_statistics_computed=False))
        if len(done)!=5:continue
        for ratio in RATIOS:
            for metric in METRICS:
                a=[index[j['id'],1,ratio][metric] for j in done];b=[index[j['id'],-1,ratio][metric] for j in done]
                summaries.append(dict(model_config=model,dataset=dataset,validation_alarm_percent=ratio,metric=metric,
                    score_direction_selected_using_test_labels=False,opposite_direction_primary=False,**paired_summary(a,b)))
    if (len(records)!=466 or len(flat)!=2796 or len(group_status)!=110 or len(summaries)!=1869
            or sum(x['complete'] for x in group_status)!=89
            or sum(0<len(x['completed_seeds'])<5 for x in group_status)!=6
            or sum(not x['completed_seeds'] for x in group_status)!=15):raise ValueError('All466 complete sources and89 full five-seed groups required')
    for name,data in [('all466_both_orientation_fixed_validation_controls.csv',flat),
            ('all466_both_orientation_full_entity_controls.csv',entities),('all89_complete_five_seed_paired_orientation_summaries.csv',summaries)]:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(out/'all466_source_and_direction_proof_bindings.json',dict(records=records))
    write(out/'all110_original_group_completion.json',dict(groups=group_status))
    write(out/'report.json',dict(passed=True,source_cases=466,orientation_controls=2796,entity_controls=len(entities),
        complete_five_seed_groups=89,paired_metric_summaries=1869,required_original_CPU_cases=550,required_original_GPU_cases=850,
        original_scope=c['original_scope'],both_prespecified_directions_reported=True,test_labels_used_for_direction_selection=False,
        primary_scores_thresholds_and_direction_unchanged=True,opposite_orientation_primary=False,
        no_point_adjustment=True,all_thresholds_calibrated_on_validation_only=True,
        new_neural_forwards=0,new_optimizer_updates=0,new_training_seed_slots=0,new_model_results=0,
        full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);verify(c)
    if a.worker:worker(c);return
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('All466 direction diagnostic already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(v):
        v=dict(v)
        if v.get('state')=='replaying_full_checkpoints':v['state']='measuring_both_frozen_score_directions_without_neural_inference'
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        if (base/'worker.stdout.log').exists():raise RuntimeError('Preserve prior diagnostic attempt;no automatic retry')
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
            command=[sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(state='all466_prespecified_direction_controls',active_pid=child.pid,
                    active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,c,api,update)
            proof=Path(c['artifact_root'])/'report.json';passed=code==0 and not usage['resource_guard_aborted'] and proof.is_file() and read(proof)['passed']
            write(base/'receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None))
            if not passed:raise RuntimeError('Terminal all466 direction diagnostic failure preserved')
        update(dict(state='complete',active_pid=None,passed=True))
    except Exception as error:
        update(dict(state='terminal_failure_preserved',active_pid=None,error=str(error)));raise
    finally:owner.close()


if __name__=='__main__':main()
