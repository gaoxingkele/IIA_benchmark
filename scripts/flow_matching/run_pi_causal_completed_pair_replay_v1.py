"""Fully audit and replay the two already-trained original Pi causal PSM seeds."""
import argparse
import ast
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching import audit_pi_original_recovery_completion_v1 as completion
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import resource_api,capped_wait
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

ROOT,read,digest=completion.ROOT,completion.read,completion.digest
IDS=['pi_author__causal_axis_repaired__psm__full__seed1103','pi_author__causal_axis_repaired__psm__full__seed1104']
GUARDS=('python','cpu_threads','settings','resource_lock')


def historical_training_usage(base,job,case):
    path=base/'resource_usage.json'
    if path.exists():
        usage=read(path)
        if usage.get('resource_guard_aborted') is not False:
            raise ValueError('A preserved resource-aborted or unknown recovery receipt cannot certify completion')
        return dict(usage,original_training_resource_evidence_available=True)
    if job!=case['job'] or case['kind']!='readonly_original_canonical_binding':
        raise ValueError('Only historical exact canonical outputs may explicitly lack a resource receipt')
    return dict(original_training_resource_evidence_available=False,original_training_guard_verified=False,
        resource_guard_aborted=None,
        boundary='Original full canonical training result has no persisted resource usage receipt; its guard history remains unproven. New full point/native/independent replay stages are guarded.')


def completion_ast(source):
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    expected=ast.parse("usage = read(base/'resource_usage.json')").body[0]
    matches=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and ast.dump(n)==ast.dump(expected)]
    if len(matches)!=1:raise ValueError('Exactly one unchanged historical resource receipt boundary required')
    matches[0].value=ast.parse('historical_training_usage(base,job,case)',mode='eval').body
    return ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))


def point_worker(config,path):
    for pin in config['source_receipts']:
        if digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen complete original Pi source changed')
    namespace=dict(vars(completion),historical_training_usage=historical_training_usage)
    source=Path(completion.__file__).read_text(encoding='utf-8')
    exec(compile(completion_ast(source),'<unchanged original full point audit with explicit historical receipt disclosure>','exec'),namespace)
    namespace['audit'](config,path)


def verify(config):
    for pin in config['source_receipts']:
        if digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen full canonical Pi pair source changed: '+pin['path'])
    queue=read(config['original_queue'])
    if (len(queue['jobs'])!=1140 or len({j['id'] for j in queue['jobs']})!=1140
            or queue['settings']['seeds']!=[42,1103,1104,1105,1106,1107]
            or config['required_original_jobs']!=1140 or [c['canonical_id'] for c in config['cases']]!=IDS
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('Preserve all1140 original jobs, full six-seed scope and both ready original models')
    native_template=read(config['native_template']);review_template=read(config['review_template'])
    for case in config['cases']:
        point=read(case['point_configuration']);native=read(case['native_configuration'])
        if (any(native[k]!=native_template[k] for k in GUARDS)
                or any(point[k]!=review_template[k] for k in ('python','settings','resource_lock'))
                or point['cpu_threads']!=native_template['cpu_threads']
                or native['score_tolerance']!=native_template['score_tolerance']
                or native['independent_completion']!=point['output_directory']+'/independent_full_recovery_completion.json'):
            raise ValueError('Keep exact original native and independent guards, CPU settings and tolerances')
        bound,job,result=completion.binding(point)
        canonical=next(j for j in queue['jobs'] if j['id']==case['canonical_id'])
        if (job!=canonical or bound['job']!=canonical or bound['kind']!='readonly_original_canonical_binding'
                or bound['settings']!=queue['settings'] or job['branch']!='causal_axis_repaired'
                or job['dataset']!='PSM' or job['recipe']!='full'
                or job['seed']!=int(case['canonical_id'].rsplit('seed',1)[1])
                or result['parameter_count']!=1930289 or len(result['epochs'])!=3
                or any(e['steps']!=518 for e in result['epochs'])
                or job['author_config']['data']['batch_size']!=256
                or job['author_config']['model']['d_model']!=512
                or job['author_config']['training']['n_epochs']!=3
                or native['recovery_job_id']!=job['id'] or native['recovery_queue']!=point['recovery_queue']):
            raise ValueError('Exactly the two complete original full-size canonical models required')
    return queue,native_template,review_template


def point_controller(config,path):
    completion.binding(config)
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    api=resource_api();lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This complete point audit already has an actual owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        if (base/'receipt.json').exists():raise FileExistsError('Preserve completed and failed audit attempts')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_pi_causal_completed_pair_replay_v1',
                '--config',path,'--point-worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
                        PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            p=psutil.Process(child.pid);update(dict(state='auditing_all_original_points',active_pid=p.pid,
                active_create_time=p.create_time(),actual_command=p.cmdline()))
            code,usage=capped_wait(child,config,api,update)
        proof_path=ROOT/config['output_directory']/'independent_full_recovery_completion.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and proof_path.exists() and read(proof_path)['passed'])
        write(base/'receipt.json',dict(exit_code=code,full_original_point_audit_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
        if not passed:raise RuntimeError('Full original point audit failed; all evidence preserved')
    finally:lane.close()


def successful_stage(stage,flag):
    receipt=read(stage['state_root']+'/receipt.json')
    if receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or not receipt[flag]:
        raise ValueError('Only the actual successful nonaborted full stage may advance')
    return receipt


def controller(config,path):
    _,_,review_template=verify(config)
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=resource_api()['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('The two complete canonical model replay pipeline already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        configuration_sha256=digest(path),original_scope=config['original_scope'],required_original_jobs=1140,
        original_training_seeds=[42,1103,1104,1105,1106,1107],selected_ready_original_ids=IDS,
        verified_ready_original_ids=[],all_original_experiments_complete=False)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def execute(stage_path,module,phase,identity,flag):
        stage=read(stage_path)
        if (ROOT/stage['state_root']/'receipt.json').exists():successful_stage(stage,flag);return
        command=[config['python'],'-X','utf8','-u','-m',module,'--config',stage_path]
        if phase=='point':command.append('--point-controller')
        logs=base/'logs'/identity;logs.mkdir(parents=True,exist_ok=True)
        with (logs/(phase+'.stdout.log')).open('xb') as out,(logs/(phase+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        p=psutil.Process(child.pid);creation,actual=p.create_time(),p.cmdline()
        update(dict(state='waiting_actual_full_'+phase,active_job=identity,active_pid=p.pid,active_create_time=creation,actual_command=actual))
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid)
                if p.create_time()!=creation or p.cmdline()!=actual:raise ValueError('Actual full stage process identity changed')
                update(dict(actual_child_cpu_seconds=sum(p.cpu_times()[:2])))
            except psutil.NoSuchProcess:
                if child.poll() is None:raise
            time.sleep(config['poll_seconds'])
        if child.wait()!=0:raise RuntimeError('Full original stage failed; preserve the attempt and actual evidence')
        successful_stage(stage,flag);update(dict(active_pid=None,active_job=None))
    try:
        # Audit both already-completed budgets and all saved streams first;
        # native inference retains the original larger admission requirements.
        for case in config['cases']:
            execute(case['point_configuration'],'scripts.flow_matching.run_pi_causal_completed_pair_replay_v1',
                'point',case['canonical_id'],'full_original_point_audit_passed')
        for case in config['cases']:
            identity=case['canonical_id'];native=read(case['native_configuration'])
            point=read(case['point_configuration'])
            native['source_receipts']+= [dict(path=p,sha256=digest(p)) for p in (
                case['native_configuration'],case['point_configuration'],native['independent_completion'],
                point['state_root']+'/receipt.json',path)]
            native_path=config['configuration_root']+'/'+identity+'_native_ready.json'
            if (ROOT/native_path).exists():
                if read(native_path)!=native:raise ValueError('Preserve frozen complete native replay configuration')
            else:exclusive_json(ROOT/native_path,native)
            execute(native_path,'scripts.flow_matching.replay_pi_original_full_checkpoint_v1',
                'native',identity,'full_native_checkpoint_reinference_passed')
            review=copy.deepcopy(review_template)
            review.update(completed_replay_configuration=native_path,
                completed_replay_report=str(Path(native['artifact_directory'])/'full_original_checkpoint_replay.json'),
                output_directory=config['proof_root']+'/'+identity+'/independent',
                state_root=config['child_state_root']+'/'+identity+'/independent',
                source_receipts=copy.deepcopy(config['source_receipts'])+[
                    dict(path=native_path,sha256=digest(native_path)),
                    dict(path=native['independent_completion'],sha256=digest(native['independent_completion'])),
                    dict(path=native['state_root']+'/receipt.json',sha256=digest(native['state_root']+'/receipt.json')),
                    dict(path=str(Path(native['artifact_directory'])/'full_original_checkpoint_replay.json'),
                        sha256=digest(ROOT/native['artifact_directory']/'full_original_checkpoint_replay.json'))])
            review_path=config['configuration_root']+'/'+identity+'_independent.json'
            if (ROOT/review_path).exists():
                if read(review_path)!=review:raise ValueError('Preserve every frozen independent review configuration')
            else:exclusive_json(ROOT/review_path,review)
            execute(review_path,'scripts.flow_matching.verify_pi_original_full_replay_completion_v1','independent',identity,'passed')
            proof=read(review['output_directory']+'/independent_completed_native_replay.json')
            if (not proof['passed'] or proof['canonical_id']!=identity or not proof['full_original_model_checkpoint_reinference_completed']
                    or not proof['all_fifteen_full_saved_and_replay_streams_independently_reread']
                    or proof['strict_TAB_result'] or proof['author_equivalence_certified']
                    or proof['original_training_seed_slots']!=1 or proof['additional_training_seed_slots']!=0):
                raise ValueError('Actual complete original native replay and independent full evidence required')
            state['verified_ready_original_ids'].append(identity);update(dict(active_pid=None))
        exclusive_json(base/'completion.json',dict(passed=True,verified_ready_original_ids=state['verified_ready_original_ids'],
            required_original_jobs=1140,original_training_seeds=[42,1103,1104,1105,1106,1107],
            original_scope=config['original_scope'],additional_training_seed_slots=0,all_original_experiments_complete=False))
        update(dict(state='both_ready_original_full_models_and_independent_arrays_verified'))
    except Exception as error:
        write(base/'failure.json',dict(error_type=type(error).__name__,error=str(error),failed_attempt_preserved=True))
        update(dict(state='failed_preserved',active_pid=None));raise
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--point-worker',action='store_true');parser.add_argument('--point-controller',action='store_true')
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args();config=read(args.config)
    if args.point_worker:point_worker(config,args.config)
    elif args.point_controller:point_controller(config,args.config)
    elif args.probe_only:
        q,_,_=verify(config)
        print(json.dumps(dict(required_original_jobs=len(q['jobs']),ready_complete_models=IDS,
            full_native_checkpoint_reinference_passed=False,all_original_experiments_complete=False,
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules)))
    else:controller(config,args.config)


if __name__=='__main__':main()
