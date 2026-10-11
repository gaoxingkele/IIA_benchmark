"""Complete point, model and independent replay review for all139 CrossAD jobs."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT, read, digest
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

GUARDS = ('python','cpu_threads','settings','resource_lock')
SCOPE = dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
SCORE_TOLERANCE = dict(rtol=1e-6,atol=1e-7)
METRIC_TOLERANCE = dict(rtol=1e-12,atol=1e-12)


def validate_scope(queue, runtime):
    jobs={j['id']:j for j in queue['jobs']}
    groups=defaultdict(list)
    for job in jobs.values(): groups[job['author_recipe'],job['dataset']].append(job)
    counts=Counter(j['mode'] for j in jobs.values())
    if (len(jobs)!=139 or len(queue['jobs'])!=139 or len(groups)!=29
            or counts!=dict(release_checkpoint=7,fresh_train=132)
            or sum('ablation_row' in j for j in jobs.values())!=90
            or runtime['required_original_jobs']!=139 or runtime['original_scope']!=SCOPE
            or queue['settings']['evaluation']['strict_false_alarm_percent']!=[.1,1,5]):
        raise ValueError('All139 original jobs/29cohorts/7releases/132training runs/90ablations required')
    for (recipe,dataset),items in groups.items():
        expected=[2025] if recipe=='release_checkpoint' else [2025,1103,1104,1105,1106,1107]
        if sorted(j['seed'] for j in items)!=sorted(expected):
            raise ValueError('Every original independent training seed or fixed release evaluation must remain')
    return jobs


def complete_result(job,result):
    if (result['status']!='completed' or result['id']!=job['id'] or result['seed']!=job['seed']
            or result['mode']!=job['mode'] or result['dataset']!=job['dataset']
            or result['experiment_sha256']!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
            or result['paper_equivalence'] or result['strict_TAB_result']
            or result['details']['declared_train_parameters']!=job['train_parameters']):
        raise ValueError('Only unchanged complete original execution may enter full replay')


def native_terminal(config,job):
    report=read(ROOT/config['artifact_directory']/'full_native_model_replay.json')
    receipt=read(ROOT/config['state_root']/'receipt.json')
    result=read(ROOT/job['output_directory']/'result.json')
    complete_result(job,result)
    if (receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or not receipt['full_native_model_replay_passed']
            or not report['full_original_model_reinference_passed']
            or not report['full_original_author_metric_reexecution_passed']
            or not report['original_SPOT_init_model_inference_and_threshold_reexecution_passed']
            or not report['original_full_state_dictionary_loaded_strictly']
            or report['job_id']!=job['id'] or report['original_seed']!=job['seed']
            or report['source_result_sha256']!=digest(ROOT/job['output_directory']/'result.json')
            or report['cuda_context_initialized'] or report['additional_training_seed_slots']!=0
            or report['full_score_points']!=result['details']['native_points']
            or report['full_model_parameters']!=result['details']['parameter_count']
            or report['actual_cpu_threads']!=config['cpu_threads']
            or report['original_inference_score_tolerance']!=SCORE_TOLERANCE
            or report['original_metric_reexecution_tolerance']!=METRIC_TOLERANCE
            or report['original_range_parameters']!=dict(slidingWindow=100,thre=250,version='opt')):
        raise ValueError('Successful actual full native model, metrics, SPOT, all points and original tolerances required')
    for name,expected in report['replay_outputs'].items():
        if digest(ROOT/config['artifact_directory']/name)!=expected:
            raise ValueError('Completed full native replay artifact changed')
    if job['mode']=='release_checkpoint' and (not report['fixed_release_checkpoint_evaluation']
            or report['completed_training_seed_slots']!=0):
        raise ValueError('Fixed released weights cannot count as original training seeds')
    return report,receipt


def verify(runtime):
    for pin in runtime['source_receipts']:
        if digest(pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full original CrossAD continuation source changed: '+pin['path'])
    queue=read(runtime['original_queue']);jobs=validate_scope(queue,runtime)
    template=read(runtime['native_replay_template'])
    if (template['queue']!=runtime['original_queue'] or any(runtime[k]!=template[k] for k in GUARDS)
            or runtime['score_tolerance']!=SCORE_TOLERANCE or runtime['metric_tolerance']!=METRIC_TOLERANCE):
        raise ValueError('Preserve all original native guards, CPU environment and numeric tolerances')
    covered={}
    for item in runtime['preserved_verified_jobs']:
        job=jobs[item['job_id']]
        if job['id'] in covered: raise ValueError('Preserved original executions must be unique')
        config=read(item['native_configuration'])
        proof=read(config['independent_point_completion'])
        native,receipt=native_terminal(config,job)
        if (not proof['passed'] or proof['job_id']!=job['id'] or not proof['full_raw_labels_and_source_rows_checked']
                or proof['original_source_result']!=read(ROOT/job['output_directory']/'result.json')
                or native['configuration_sha256']!=digest(item['native_configuration'])):
            raise ValueError('Preserved independent full raw-point and native-model evidence must match')
        covered[job['id']]=item
    if len(covered)!=3: raise ValueError('Preserve all three completed full native and point-reviewed original executions')
    return jobs,template,covered


def live_stage_jobs():
    ids=set()
    modules=('replay_crossad_registered_model','replay_crossad_full_native_model','replay_crossad_full_release_model',
        'run_crossad_completion_replay_queue')
    for process in psutil.process_iter(['name']):
        if 'python' not in (process.info['name'] or '').lower(): continue
        try:
            command=process.cmdline()
            if '--config' not in command or not any(any(module in arg for module in modules) for arg in command): continue
            creation=process.create_time();config=read(command[command.index('--config')+1])
            if process.create_time()==creation and process.cmdline()==command and 'job_id' in config:
                ids.add(config['job_id'])
        except (psutil.NoSuchProcess,psutil.AccessDenied,OSError,ValueError,KeyError): pass
    return ids


def exclusive_configuration(path,config):
    path=ROOT/path
    if path.exists():
        if read(path)!=config: raise ValueError('Never replace a frozen original stage configuration')
    else: exclusive_json(path,config)


def point_configuration(runtime,runtime_path,job):
    result=read(ROOT/job['output_directory']/'result.json');complete_result(job,result)
    config={k:copy.deepcopy(runtime[k]) for k in (*GUARDS,'original_scope')}
    config.update(queue=runtime['original_queue'],author_manifest=runtime['author_manifest'],job_id=job['id'],
        full_model_parameters=result['details']['parameter_count'],phase='point',
        output_directory=runtime['proof_root']+'/'+job['id']+'/point',
        state_root=runtime['child_state_root']+'/'+job['id']+'/point',
        source_receipts=copy.deepcopy(runtime['source_receipts'])+[
            dict(path=runtime_path,sha256=digest(runtime_path)),
            dict(path=(Path(job['output_directory'])/'result.json').as_posix(),sha256=digest(ROOT/job['output_directory']/'result.json'))])
    path=(Path(runtime['configuration_root'])/(job['id']+'_point.json')).as_posix()
    exclusive_configuration(path,config)
    return path,config


def replay_configuration(runtime,template,job,point_path,point):
    config=copy.deepcopy(template)
    config.pop('failed_v1_attempt',None)
    config.update(job_id=job['id'],independent_point_completion=point['output_directory']+
        ('/independent_full_release_completion.json' if job['mode']=='release_checkpoint' else '/independent_full_native_completion.json'),
        state_root=runtime['child_state_root']+'/'+job['id']+'/native',
        artifact_directory=str(Path(runtime['artifact_root'])/job['id']),
        source_receipts=copy.deepcopy(point['source_receipts'])+[
            dict(path=point_path,sha256=digest(point_path))])
    config['source_receipts'].append(dict(path=config['independent_point_completion'],sha256=digest(config['independent_point_completion'])))
    path=(Path(runtime['configuration_root'])/(job['id']+'_native.json')).as_posix()
    exclusive_configuration(path,config)
    return path,config


def post_review(config,path):
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score
    from scripts.flow_matching.replay_crossad_full_native_model_v1 import compare_scores,compare_metrics
    native=read(config['native_configuration']);queue=read(native['queue'])
    job=next(j for j in queue['jobs'] if j['id']==native['job_id'])
    report,receipt=native_terminal(native,job)
    if report['configuration_sha256']!=digest(config['native_configuration']):
        raise ValueError('Actual complete native replay must bind the exact stage configuration')
    point=read(native['independent_point_completion']);result=read(ROOT/job['output_directory']/'result.json')
    if not point['passed'] or point['original_source_result']!=result:
        raise ValueError('Every full original raw point and source row must already be checked')
    for artifact in result['artifacts']:
        if digest(artifact['path'])!=artifact['sha256']: raise ValueError('Original complete source artifact changed')
    with np.load(ROOT/job['output_directory']/'native_scores.npz',allow_pickle=False) as original,np.load(
            ROOT/native['artifact_directory']/'all_native_replayed_scores.npz',allow_pickle=False) as replay:
        compare_scores(original,replay)
        for key in ('labels','source_rows'):
            if not np.array_equal(original[key],replay[key]): raise ValueError('Every full raw-label and row identity must match')
        reviewed={}
        if set(result['metrics']['native_validation_control']['strict'])!={'0.1','1','5'}:
            raise ValueError('All three original complete validation operating points required')
        for ratio,recorded in result['metrics']['native_validation_control']['strict'].items():
            threshold=float(np.percentile(np.asarray(replay['validation'],dtype=np.float64),100-float(ratio),method='linear'))
            y=replay['labels'];prediction=replay['test']>threshold
            tp=int(np.sum(prediction&(y==1)));fp=int(np.sum(prediction&(y==0)));fn=int(np.sum(~prediction&(y==1)))
            values=dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,auroc=float(roc_auc_score(y,replay['test'])),
                average_precision=float(average_precision_score(y,replay['test'])))
            if (recorded['point_adjustment'] or not math.isclose(threshold,recorded['calibration']['threshold'],**METRIC_TOLERANCE)
                    or any(not math.isclose(value,recorded['micro'][key],**METRIC_TOLERANCE) for key,value in values.items())):
                raise ValueError('Every complete independently recomputed replayed validation-only metric must agree')
            reviewed[ratio]=dict(threshold=threshold,metrics=values)
    compare_metrics(result['metrics']['original_author'],report['original_author_metrics'])
    exclusive_json(ROOT/config['output_directory']/'independent_completed_replay_review.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=path,configuration_sha256=digest(path),
        job_id=job['id'],original_seed=job['seed'],mode=job['mode'],registered_author_recipe=job['author_recipe'],
        registered_ablation_row=job.get('ablation_row'),reconstructed_ablation=job['reconstructed_ablation'],
        full_native_model_replay_passed=True,all_original_raw_point_and_row_identities_independently_reviewed=True,
        all_full_original_and_replayed_scores_independently_compared=True,all_original_author_metrics_reexecuted_and_compared=True,
        replayed_validation_only_metrics_independently_recomputed=reviewed,full_point_counts=result['details']['native_points'],
        original_training_seed_slots_completed=0 if job['mode']=='release_checkpoint' else 1,
        required_original_cohort_training_seeds=0 if job['mode']=='release_checkpoint' else 6,
        native_terminal_receipt=receipt,native_proof_sha256=digest(ROOT/native['artifact_directory']/'full_native_model_replay.json'),
        additional_training_seed_slots=0,original_scope=config['original_scope'],author_equivalence_certified=False,
        unpublished_author_ablation_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False))


def stage_worker(config,path):
    for pin in config['source_receipts']:
        if digest(pin['path'])!=pin['sha256']: raise ValueError('Frozen full stage source changed')
    if config['phase']=='post': return post_review(config,path)
    queue=read(config['queue']);job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    if job['mode']=='release_checkpoint':
        from scripts.flow_matching.audit_crossad_full_release_completion_v1 import audit
    else:
        from scripts.flow_matching.audit_crossad_full_native_completion_v1 import audit
    audit(path)


def stage_controller(config,path):
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('This exact independent CrossAD stage already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values): state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_crossad_completion_replay_queue_v1',
                '--config',path,'--stage-worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),
                        MKL_NUM_THREADS=str(config['cpu_threads']),OPENBLAS_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
            code,usage=capped_wait(child,config,api,update)
        if config['phase']=='post': proof_name='independent_completed_replay_review.json'
        else:
            job=next(j for j in read(config['queue'])['jobs'] if j['id']==config['job_id'])
            proof_name='independent_full_'+('release' if job['mode']=='release_checkpoint' else 'native')+'_completion.json'
        proof_path=ROOT/config['output_directory']/proof_name
        passed=bool(code==0 and not usage['resource_guard_aborted'] and proof_path.exists() and read(proof_path)['passed'])
        write(base/'receipt.json',dict(exit_code=code,independent_stage_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
        if not passed: raise RuntimeError('Full independent stage failed; preserve original attempt')
    finally: lane.close()


def controller(runtime,path):
    jobs,template,verified=verify(runtime);api=resource_api();base=ROOT/runtime['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('This complete original139 CrossAD queue already has an owner')
    old=read(base/'status.json') if (base/'status.json').exists() else {};failed=old.get('failed_preserved',{})
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration_sha256=digest(path),active_pid=None,
        required_original_jobs=139,verified_original_ids=sorted(verified),failed_preserved=failed,all_original_experiments_complete=False)
    def update(values): state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def run_stage(stage_path,module,phase,identity):
        config=read(stage_path);receipt_path=ROOT/config['state_root']/'receipt.json'
        if receipt_path.exists():
            receipt=read(receipt_path)
            if receipt['exit_code']!=0 or receipt['resource_guard_aborted']: raise ValueError('Preserve failed existing stage; do not retry silently')
            return
        logs=base/'logs'/identity;logs.mkdir(parents=True,exist_ok=True)
        command=[runtime['python'],'-X','utf8','-u','-m',module,'--config',stage_path]
        if phase!='native': command.append('--stage-controller')
        with (logs/(phase+'.stdout.log')).open('xb') as out,(logs/(phase+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        handle=psutil.Process(child.pid);creation,actual=handle.create_time(),handle.cmdline()
        update(dict(state='waiting_actual_full_'+phase,active_job=identity,active_pid=child.pid,active_create_time=creation,actual_command=actual))
        while child.poll() is None:
            try:
                current=psutil.Process(child.pid)
                if current.create_time()!=creation or current.cmdline()!=actual: raise ValueError('Actual original full-stage identity changed')
                update(dict(actual_child_cpu_seconds=sum(current.cpu_times()[:2])))
            except psutil.NoSuchProcess:
                if child.poll() is None: raise
            time.sleep(runtime['poll_seconds'])
        if child.wait()!=0: raise RuntimeError('Original full stage failed; all actual inputs and attempts preserved')
        update(dict(active_pid=None,active_job=None))
    try:
        while len(verified)<139:
            live=live_stage_jobs()
            for identity,job in jobs.items():
                if identity in verified or identity in failed or identity in live: continue
                result_path=ROOT/job['output_directory']/'result.json'
                if not result_path.exists(): continue
                try: result=read(result_path)
                except (OSError,ValueError): continue
                if result.get('status')!='completed': continue
                try:
                    point_path,point=point_configuration(runtime,path,job)
                    run_stage(point_path,'scripts.flow_matching.run_crossad_completion_replay_queue_v1','point',identity)
                    native_path,native=replay_configuration(runtime,template,job,point_path,point)
                    run_stage(native_path,'scripts.flow_matching.replay_crossad_registered_model_v3','native',identity)
                    native_terminal(native,job)
                    post={k:copy.deepcopy(point[k]) for k in (*GUARDS,'original_scope','job_id')}
                    post.update(phase='post',queue=runtime['original_queue'],native_configuration=native_path,
                        output_directory=runtime['proof_root']+'/'+identity+'/post',state_root=runtime['child_state_root']+'/'+identity+'/post',
                        source_receipts=copy.deepcopy(native['source_receipts'])+[
                            dict(path=native_path,sha256=digest(native_path)),
                            dict(path=native['state_root']+'/receipt.json',sha256=digest(native['state_root']+'/receipt.json')),
                            dict(path=str(Path(native['artifact_directory'])/'full_native_model_replay.json'),
                                sha256=digest(ROOT/native['artifact_directory']/'full_native_model_replay.json'))])
                    post_path=(Path(runtime['configuration_root'])/(identity+'_post.json')).as_posix()
                    exclusive_configuration(post_path,post)
                    run_stage(post_path,'scripts.flow_matching.run_crossad_completion_replay_queue_v1','post',identity)
                    proof=read(ROOT/post['output_directory']/'independent_completed_replay_review.json')
                    if not proof['passed'] or proof['job_id']!=identity: raise ValueError('Complete independent full replay proof required')
                    verified[identity]=dict(job_id=identity,post_review_configuration=post_path,
                        independent_completed_replay_review=post['output_directory']+'/independent_completed_replay_review.json')
                    exclusive_json(base/'completed'/(identity+'.json'),verified[identity])
                except Exception as error:
                    failed[identity]=dict(error_type=type(error).__name__,error=str(error),failed_attempt_preserved=True)
                update(dict(verified_original_ids=sorted(verified),failed_preserved=failed,active_pid=None,active_job=None))
            update(dict(state='waiting_actual_original_full_execution_completion',verified_original_ids=sorted(verified),
                pending_original_jobs=139-len(verified),failed_preserved=failed,live_other_original_full_stage_jobs=sorted(live)))
            time.sleep(runtime['poll_seconds'])
        update(dict(state='all139_original_full_native_and_independent_executions_verified'))
    finally: lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage-worker',action='store_true');parser.add_argument('--stage-controller',action='store_true')
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args();config=read(args.config)
    if args.stage_worker: stage_worker(config,args.config)
    elif args.stage_controller: stage_controller(config,args.config)
    elif args.probe_only:
        jobs,template,verified=verify(config)
        print(json.dumps(dict(required_original_jobs=len(jobs),preserved_complete_native_and_independent_jobs=len(verified),
            pending_original_jobs=len(jobs)-len(verified),torch_imported='torch' in __import__('sys').modules,
            numpy_imported='numpy' in __import__('sys').modules,all_original_experiments_complete=False)))
    else: controller(config,args.config)
