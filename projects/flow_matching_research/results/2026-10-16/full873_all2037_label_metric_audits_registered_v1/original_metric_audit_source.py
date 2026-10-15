"""Evaluate every full873 saved-score case under all seven raw label views.

Source guard, actual full109 preparation and all frozen inputs are required.
No fitting, model reinference, test-selected label policy or numerical retry.
Independent model reload/reinference and full TAB remain separate requirements.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.flow_matching import run_original_full_raw873_experiments_v2 as source

ROOT,read,sha,write=source.ROOT,source.read,source.sha,source.write
MODULE='scripts.flow_matching.audit_original_full873_label_metrics_v1'
CONFIG='configs/reproducibility/fm_original_full873_all2037_label_metrics.2026-10-16_v1.json'
TEST='tests/test_fm_original_full873_label_metrics_v1.py'
POLICY='configs/reproducibility/fm_timesnet_all5_NASA_label_protocol.2026-10-16_v1.json'
REPORT='all_label_view_metrics.json'


def registration():
    # The producer queues use both Python3.12 and the MOMENT author Python3.10.
    # Read their already frozen metadata without regenerating a different
    # producer registration in the scalar worker's environment.
    original=read(ROOT/source.CONFIG)
    material_path=Path(original['artifact_root'])/'materialization.json';material=read(material_path)
    if not material['passed'] or material['configuration_sha256']!=sha(ROOT/source.CONFIG) or material['required_cases']!=873:
        raise ValueError('Actual full873 queue materialization required')
    policy=read(ROOT/POLICY);cases=[];pins={source.CONFIG,source.preparation.CONFIG,POLICY,
        MODULE.replace('.','/')+'.py',TEST,'scripts/flow_matching/run_original_full_raw873_experiments_v2.py',
        'scripts/flow_matching/prepare_original_full_entity_tsad_v2.py','scripts/flow_matching/run_tsad_experiment.py',
        'scripts/flow_matching/run_moment_pretrained_job.py','scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'scripts/flow_matching/audit_timesnet_NASA_label_protocol_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',str(material_path)}
    for family in original['families']:
        item=next(q for q in material['queues'] if q['family']==family['name'])
        if sha(item['path'])!=item['sha256']:raise ValueError('Actual frozen full873 queue changed')
        queue=read(item['path']);source.validate_clone(read(ROOT/family['source_queue']),queue,family,material['full_data_binding'])
        pins.add(item['path'])
        for job in queue['jobs']:
            record=material['full_data_binding']['records'][job['dataset']]
            cases.append(dict(id=job['id'],family=family['name'],dataset=job['dataset'],seed=job['seed'],window=job['window'],
                queue_path=item['path'],queue_sha256=item['sha256'],worker_python=queue['python'],
                required_label_policies=list(record['labels_only_controls']),source_guard=str(ROOT/original['state_root']/'families'/family['name']/'receipts'/(job['id']+'.json'))))
    views=Counter(c['dataset'] for c in cases for _ in c['required_label_policies'])
    if len(cases)!=873 or len({c['id'] for c in cases})!=873 or dict(views)!=dict(smap=1164,msl=582,smd=291):
        raise ValueError('Every873case and all2037raw label views required')
    return dict(schema_version=1,source_execution=source.CONFIG,actual_materialization=str(material_path),
        required_cases=873,required_label_view_results=2037,required_view_counts=dict(views),cases=cases,
        settings=policy['settings'],resource_lock=policy['resource_lock'],controller_python=original['controller_python'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        paper_and_data_citations=read(ROOT/source.preparation.CONFIG)['paper_citations'],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full873_all2037_label_metrics_v1',
        state_root='experiments/runs/fm_original_full873_all2037_label_metrics_v1',
        source_metrics_exact_recomputation_required=True,independent_confusion_ranking_and_threshold_grid_required=True,
        test_based_label_policy_selection_allowed=False,all_original_validation_quantiles_bootstrap_seeds_and_repeats_retained=True,
        independent_model_reload_or_reinference_performed=False,full_TAB_certified=False,
        original_scope=dict(papers=45,method_baseline_ablation_records=681,data_tasks=273,paper_scalar_targets=702),
        boundary='All873model/dataset/seed cases;2037views=291*(4SMAP+2MSL+1SMD). '
            'Require actual full109data and successful original source execution guard. All predictions/checkpoints unchanged. '
            'Recompute full source point/event/bootstrap/grid metrics exactly,then evaluate every immutable label overlay '
            'with all original validation quantiles and bootstrap settings;independent confusion/ranking/PA/grid checks. '
            'No selection of the best test label policy or score orientation. Source failures/partial outputs preserved. '
            'Saved-score scalar review is not independent model checkpoint reload or reinference;both remain required. '
            'Five-seed paired statistics,preprocessing author equivalence and fullTAB/VUS/Affiliation remain separate. '
            'Readonly16/28/16GiB headroom and3/6GiB tree caps with existing audit lock unchanged.')


def verify(config):
    if config!=registration():raise ValueError('Complete873cases/all2037views and unchanged audit policy required')
    return config['cases']


@lru_cache(maxsize=7)
def cached_queue(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen source case queue changed')
    return read(path)


def resolve_case(case,rehash=False):
    if rehash and sha(case['queue_path'])!=case['queue_sha256']:raise ValueError('Frozen source case queue changed')
    queue=cached_queue(case['queue_path'],case['queue_sha256']);jobs=[j for j in queue['jobs'] if j['id']==case['id']]
    if len(jobs)!=1:raise ValueError('Exact unique fullraw source job required')
    job=jobs[0]
    if any(job[k]!=case[k] for k in ['id','dataset','seed','window']):raise ValueError('Original scientific identity differs')
    return queue,job


def validate_source_guard(case,guard,success_required=True):
    gpu=case['family']=='neural'
    if (guard.get('original_job' if gpu else 'job_id')!=case['id']
            or guard.get('full_original_budget_retained' if gpu else 'full_registered_budget_retained') is not True
            or guard.get('queue_sha256')!=case['queue_sha256']):
        raise ValueError('Successful exact original source guard and frozen scientific budget required')
    command=guard['command']
    if (command[command.index('--job-id')+1]!=case['id']
            or (ROOT/command[command.index('--queue')+1]).resolve()!=(ROOT/case['queue_path']).resolve()):
        raise ValueError('Actual guarded producer command belongs to another source case')
    if success_required and (guard['exit_code']!=0 or guard['resource_guard_aborted']):
        raise ValueError('Successful original source execution required')


def validate_source_result(case,job,result,guard):
    validate_source_guard(case,guard)
    purpose='full_frozen_pretrained_zero_shot_tsad' if case['family']=='moment' else 'full_local_tsad_benchmark'
    if (result.get('status')!='completed' or result.get('id')!=case['id'] or result.get('job')!=job
            or result.get('purpose')!=purpose or result.get('dataset')!=case['dataset'] or result.get('seed')!=case['seed']
            or result.get('model_config')!=job['model_config']
            or result.get('experiment_sha256')!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()):
        raise ValueError('Actual complete unchanged producer result required')


def source_status(case):
    queue,job=resolve_case(case);base=ROOT/job['output_directory'];receipt=Path(case['source_guard'])
    if not receipt.exists():return None
    guard=read(receipt)
    validate_source_guard(case,guard,success_required=False)
    if guard['exit_code']!=0 or guard['resource_guard_aborted']:
        return dict(kind='source_failed_preserved',source_guard=str(receipt),sha256=sha(receipt),
            exit_code=guard['exit_code'],resource_guard_aborted=guard['resource_guard_aborted'],metrics_available=False)
    path=base/'result.json'
    if not path.exists():raise ValueError('Successful source exit lacks complete source result;preserve for diagnosis')
    result=read(path);validate_source_result(case,job,result,guard)
    return dict(kind='source_complete',result_path=str(path),result_sha256=sha(path),source_guard_sha256=sha(receipt))


def source_binding(config,case):
    state=source_status(case)
    if state is None or state['kind']!='source_complete':raise ValueError('Actual completed source and successful guard required')
    queue,job=resolve_case(case,rehash=True);base=ROOT/job['output_directory'];result=read(base/'result.json')
    full=source.ready_data(read(ROOT/config['source_execution']))
    if full is None:raise ValueError('Actual complete109data and successful preparation guard required')
    dataset=full['records'][case['dataset']]
    if list(dataset['labels_only_controls'])!=case['required_label_policies']:raise ValueError('Every raw label policy required')
    for pin in job['frozen_files']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen source model/data dependency changed')
    for filename,key in [('model.pt','checkpoint_sha256'),('scores.npz','scores_sha256')]:
        if sha(base/filename)!=result[key]:raise ValueError('Complete source checkpoint or scores changed')
    return queue,job,result,dataset,dict(**state,queue_sha256=case['queue_sha256'],
        scores_sha256=result['scores_sha256'],checkpoint_sha256=result['checkpoint_sha256'],
        full_data_proof_sha256=full['full_data_proof_sha256'],full_data_guard_sha256=full['full_data_guard_sha256'],
        input_sha256=dataset['input_sha256'],manifest_sha256=full['manifest_sha256'])


def collect_arrays(inputs,saved,dataset):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    entities={};training=[];validation=[];coverage=[];digests={};expected=set()
    for entity in dataset['entities']:
        prefix=entity['array_prefix']
        for role in ['train','validation','test']:
            key=prefix+'_'+role;expected.add(key);values=saved[key]
            if values.ndim!=1 or len(values)!=len(inputs[key]) or not np.isfinite(values).all():
                raise ValueError('Every original native source point requires a finite saved score')
            digests[key]=array_digest(values);coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values)))
            if role=='train':training.append(values)
            elif role=='validation':validation.append(values)
        key=prefix+'_labels';expected.add(key);require_identical(saved[key],inputs[key])
        labels=saved[key]
        if labels.ndim!=1 or len(labels)!=len(saved[prefix+'_test']) or not np.isin(labels,[0,1]).all():
            raise ValueError('Every source target label must be binary and aligned')
        digests[key]=array_digest(labels);entities[entity['entity_id']]=(labels,saved[prefix+'_test'])
    if set(saved.files)!=expected:raise ValueError('Every saved array must belong to the full declared source score/label layout')
    return entities,np.concatenate(validation),np.concatenate(training),coverage,digests


def evaluate_views(entities,validation,training,overlays,dataset,result,evaluation,seed):
    import numpy as np
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import independent_controls
    from scripts.flow_matching.audit_timesnet_NASA_label_protocol_v1 import protocol_diagnostics
    original=evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],evaluation,seed)
    if original!=result['metrics']:raise ValueError('All original metrics/events/bootstrap/grid must recompute exactly')
    policies=dataset['labels_only_controls']
    if set(overlays)!=set(policies):raise ValueError('Every predeclared immutable label overlay required')
    expected_keys={e['array_prefix']+'_labels' for e in dataset['entities']};views={}
    for policy,labels in overlays.items():
        if set(labels)!=expected_keys:raise ValueError('Every original entity needs a full labels-only control')
        actual={};changed=0
        for entity in dataset['entities']:
            name=entity['entity_id'];y=np.asarray(labels[entity['array_prefix']+'_labels']);old,scores=entities[name]
            if y.shape!=old.shape or not np.isin(y,[0,1]).all():raise ValueError('Complete binary test-only label controls required')
            changed+=int((y!=old).sum());actual[name]=(y,scores)
        metrics=evaluate_saved(actual,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],evaluation,seed)
        if any(metrics['strict'][r]['calibration']!=original['strict'][r]['calibration'] for r in original['strict']):
            raise ValueError('Same model scores and original validation thresholds required for all label views')
        if policy==dataset['primary_label_policy'] and (changed or metrics!=original):
            raise ValueError('Primary raw label view must exactly recover all original targets and metrics')
        controls=independent_controls(actual,validation,training,metrics,evaluation)
        views[policy]=dict(metrics=metrics,independent_confusion_ranking_PA_and_all_grid_controls=controls,
            primary_protocol_diagnostics=protocol_diagnostics(actual,metrics['strict'][str(evaluation['primary_false_alarm_percent'])]['calibration']['threshold']),
            changed_target_points=changed,source_scores_sha256=result['scores_sha256'],label_policy_selected_by_TEST_performance=False)
    return views


def worker(config,case):
    import numpy as np
    queue,job,result,dataset,binding=source_binding(config,case)
    actual=dict(python=platform.python_version(),numpy=np.__version__,torch=importlib.metadata.version('torch'))
    if any(actual[k]!=result['environment'][k] for k in actual):raise ValueError('Exact source Python/NumPy/Torch producer versions required')
    output=Path(config['artifact_root'])/case['id'];output.mkdir(parents=True,exist_ok=False)
    evaluation=dict(queue['evaluation'])
    if case['family']=='moment':evaluation['bootstrap_block_points']=max(evaluation['bootstrap_block_points'],case['window'])
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as scores:
        entities,validation,training,coverage,digests=collect_arrays(inputs,scores,dataset)
    overlays={};label_bindings={}
    for policy,record in dataset['labels_only_controls'].items():
        if sha(record['path'])!=record['sha256']:raise ValueError('Immutable complete raw labels-only overlay changed')
        with np.load(record['path'],allow_pickle=False) as labels:overlays[policy]={k:labels[k] for k in labels.files}
        label_bindings[policy]=dict(path=record['path'],sha256=record['sha256'])
    views=evaluate_views(entities,validation,training,overlays,dataset,result,evaluation,job['seed'])
    _,_,_,_,after=source_binding(config,case)
    if binding!=after:raise ValueError('All original scores,checkpoint,data and source receipts must remain unchanged')
    for record in label_bindings.values():
        if sha(record['path'])!=record['sha256']:raise ValueError('Original label controls changed during evaluation')
    write(output/REPORT,dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        source_environment=actual,source_metrics_exactly_recomputed=True,all_original_arrays_checked=True,
        full_positional_saved_score_coverage=coverage,source_array_digests=digests,label_bindings=label_bindings,
        required_label_view_results=len(views),label_view_results=views,source_evaluation=evaluation,
        all_validation_thresholds_and_predictions_unchanged=True,training_or_model_inference_performed=False,
        independent_model_checkpoint_reload_and_reinference_complete=False,conditional_entity_uncertainty_not_training_seed_uncertainty=True,
        full_TAB_or_author_equivalence_certified=False,all_original_experiments_complete=False))


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('All873label-metric audit owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every existing audit attempt')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_cases=873,required_label_view_results=2037,active_pid=None,all_original_experiments_complete=False)
    view_counts={c['id']:len(c['required_label_policies']) for c in cases}
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_cases=list(completed),failed_audits_preserved=dict(failed),
            completed_label_view_results=sum(view_counts[c] for c in completed),
            source_failures_preserved=dict(source_failed),captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    try:
        while len(completed)+len(failed)+len(source_failed)<len(cases):
            for case in cases:
                identity=case['id']
                if identity in completed or identity in failed or identity in source_failed:continue
                try:
                    source_state=source_status(case)
                    if source_state is None:continue
                    if source_state['kind']=='source_failed_preserved':source_failed[identity]=source_state;continue
                    attempt=base/'attempts'/identity;output=Path(config['artifact_root'])/identity
                    if attempt.exists() or output.exists():raise FileExistsError('Preserve prior full/partial metric audit;no retry')
                    with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                        verify(read(ROOT/CONFIG));source_binding(config,case);attempt.mkdir(parents=True,exist_ok=False)
                        command=[str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',MODULE,'--job-id',identity]
                        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            write(attempt/'launch.json',handle);update(state='auditing_every_fullraw_label_view',active_pid=child.pid,active_job=identity)
                            code,usage=capped_wait(child,config,api,update)
                        proof=output/REPORT
                        ok=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
                        if not ok:raise ValueError('Full label-metric audit failed;preserve source and attempt without retry')
                    completed.append(identity);update(state='between_complete_label_metric_audits',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='audit_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_full873_source_results_and_successful_guards',active_pid=None,active_job=None);time.sleep(10)
        update(state='all873_metric_audit_outcomes_terminal_other_research_requirements_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');args=parser.parse_args()
    if args.register:
        source.verify(read(ROOT/source.CONFIG))
        write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=873,label_views=2037)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:
        states=[source_status(c) for c in cases]
        print(json.dumps(dict(passed=True,cases=873,label_views=2037,
            source_completed=sum(r is not None and r['kind']=='source_complete' for r in states),
            source_failed_preserved=sum(r is not None and r['kind']=='source_failed_preserved' for r in states),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,model_inference_performed=False)))
    elif args.job_id:worker(config,next(c for c in cases if c['id']==args.job_id))
    else:controller(config,cases)


if __name__=='__main__':main()
