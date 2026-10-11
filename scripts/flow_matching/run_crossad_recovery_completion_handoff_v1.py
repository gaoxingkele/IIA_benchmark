"""Review exact CrossAD recovery outputs and union them with the original139.

The live original controller and every training recipe remain immutable. Only
the four preserved failed-output routes are dispatched here; retries never add
training seeds or replace a failed attempt's artifacts.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching import run_crossad_completion_replay_queue_v1 as original
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json

ROOT, read, digest = original.ROOT, original.read, original.digest
EXPECTED = {
    'crossad_author__release_checkpoint__psm__seed2025',
    'crossad_author__release_checkpoint__smap__seed2025',
    'crossad_author__fresh_train__gecco__seed2025',
    'crossad_author__fresh_train__psm__seed2025',
}


def exact_route(route, canonical, queue, base_queue):
    candidates = [j for j in queue['jobs'] if j['id'] == route['recovery_job_id']]
    if len(candidates) != 1 or route['canonical_id'] != canonical['id']:
        raise ValueError('Exactly one frozen original-to-recovery identity required')
    job = candidates[0]
    science = lambda j: {k:v for k,v in j.items() if k not in ('id','output_directory')}
    if (science(job) != science(canonical) or job['output_directory'] == canonical['output_directory']
            or job['id'] == canonical['id'] and route['kind'] != 'selected_full_retry'):
        raise ValueError('Only registered recovery identity/output may differ, never scientific fields')
    settings = copy.deepcopy(base_queue['settings'])
    if route['kind'] == 'selected_full_retry':
        settings['output_root'] = route['selected_state_root']
    if queue['settings'] != settings:
        raise ValueError('All original data, environment, evaluator and admission settings must remain')
    return job


def canonical_union(original_ids, verified, recovery_ids):
    if len(set(original_ids)) != 139 or set(recovery_ids) != EXPECTED:
        raise ValueError('Preserve all139 original jobs and exactly four known recovery obligations')
    if not set(verified).issubset(original_ids):
        raise ValueError('Retries/unknown IDs cannot become additional original executions')
    return dict(required_original_jobs=139, verified_original_ids=sorted(verified),
        verified_original_jobs=len(verified), pending_original_jobs=139-len(verified),
        all139_original_full_native_and_independent_executions_verified=len(verified)==139,
        all_original_experiments_complete=False, additional_training_seed_slots=0)


def verify(config):
    for pin in config['source_receipts']:
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen exact recovery handoff source changed: '+pin['path'])
    base = read(config['original_completion_configuration'])
    jobs, template, covered = original.verify(base)
    if (config['required_original_jobs'] != 139 or config['original_scope'] != original.SCOPE
            or any(config[k] != base[k] for k in original.GUARDS)
            or config['score_tolerance'] != original.SCORE_TOLERANCE
            or config['metric_tolerance'] != original.METRIC_TOLERANCE
            or len(config['recovery_routes']) != 4
            or {r['canonical_id'] for r in config['recovery_routes']} != EXPECTED):
        raise ValueError('Preserve full scope, original guards and numeric tolerances')
    parent = read(base['original_queue'])
    routes = {}
    for route in config['recovery_routes']:
        identity = route['canonical_id']
        job = exact_route(route, jobs[identity], read(route['native_queue']), parent)
        if digest(route['original_failure_path']) != route['original_failure_sha256']:
            raise ValueError('Keep each original preserved failure immutable')
        registration = read(route['recovery_registration'])
        if route['kind'] == 'selected_full_retry':
            from scripts.flow_matching.run_selected_native_full_retry_v1 import full_queue_retry
            canonical, retry = full_queue_retry(parent, read(route['native_queue']), registration)
            if canonical != jobs[identity] or retry != job:
                raise ValueError('Exact selected full registry retry must bind the original job')
        else:
            cases = [c for c in registration['jobs'] if c['original_id'] == identity]
            if (len(cases)!=1 or cases[0]['job']!=job or cases[0]['original_job']!=jobs[identity]
                    or cases[0]['original_queue']!=base['original_queue']
                    or cases[0]['original_queue_sha256']!=digest(base['original_queue'])
                    or cases[0]['settings']!=parent['settings']):
                raise ValueError('Exact frozen original recovery registration required')
        if (ROOT/jobs[identity]['output_directory']/'result.json').exists():
            raise ValueError('Inspect unexpected canonical completion before a duplicate recovery review')
        routes[identity] = (route, job)
    return base, jobs, template, covered, routes


def completed_review(post_path, expected_job, guards):
    post = read(post_path)
    for pin in post['source_receipts']:
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Frozen completed recovery review source changed')
    if any(post[k] != guards[k] for k in original.GUARDS):
        raise ValueError('All original guarded native/independent stage settings required')
    native = read(post['native_configuration'])
    if any(native[k] != guards[k] for k in original.GUARDS):
        raise ValueError('Native replay guard settings changed')
    queue = read(native['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == native['job_id'])
    if job != expected_job:
        raise ValueError('Full replay must match the exact canonical or registered recovery job')
    report, receipt = original.native_terminal(native, job)
    point = read(native['independent_point_completion'])
    if (not point['passed'] or point['job_id']!=job['id']
            or not point['full_raw_labels_and_source_rows_checked']
            or point['original_source_result']!=read(ROOT/job['output_directory']/'result.json')):
        raise ValueError('Full original raw labels, source rows and complete recovery result must agree')
    if report['configuration_sha256'] != digest(post['native_configuration']):
        raise ValueError('Actual full native replay configuration hash must agree')
    proof_path = post['output_directory']+'/independent_completed_replay_review.json'
    proof = read(proof_path)
    stage_receipt = read(post['state_root']+'/receipt.json')
    if (stage_receipt['exit_code']!=0 or stage_receipt['resource_guard_aborted']
            or not stage_receipt['independent_stage_passed'] or not proof['passed']
            or proof['configuration_sha256']!=digest(post_path) or proof['job_id']!=job['id']
            or proof['original_seed']!=job['seed'] or proof['mode']!=job['mode']
            or set(proof['replayed_validation_only_metrics_independently_recomputed'])!={'0.1','1','5'}
            or proof['full_point_counts']!=report['full_score_points']
            or proof['original_training_seed_slots_completed']!=(0 if job['mode']=='release_checkpoint' else 1)
            or proof['required_original_cohort_training_seeds']!=(0 if job['mode']=='release_checkpoint' else 6)
            or proof['additional_training_seed_slots']!=0 or proof['strict_TAB_result']
            or proof['author_equivalence_certified'] or proof['all_original_experiments_complete']
            or proof['native_proof_sha256']!=digest(ROOT/native['artifact_directory']/'full_native_model_replay.json')
            or any(not proof[k] for k in ('full_native_model_replay_passed',
                'all_original_raw_point_and_row_identities_independently_reviewed',
                'all_full_original_and_replayed_scores_independently_compared',
                'all_original_author_metrics_reexecuted_and_compared'))):
        raise ValueError('Actual complete native replay and successful full independent review required')
    return dict(post_review_configuration=post_path, independent_completed_replay_review=proof_path,
        independent_completed_replay_review_sha256=digest(proof_path), native_terminal_receipt=receipt,
        completed_original_training_seed_slots=proof['original_training_seed_slots_completed'])


def stage_paths(config, config_path, base, template, route, job):
    effective = copy.deepcopy(base)
    for key in ('source_receipts','proof_root','child_state_root','configuration_root','artifact_root'):
        effective[key] = copy.deepcopy(config[key])
    effective['original_queue'] = route['native_queue']
    point_path, point = original.point_configuration(effective, config_path, job)
    replay_template = copy.deepcopy(template)
    replay_template['queue'] = route['native_queue']
    return effective, point_path, point, replay_template


def run_stage(config, stage_path, module, phase, identity, update):
    stage = read(stage_path)
    receipt_path = ROOT/stage['state_root']/'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['exit_code']!=0 or receipt['resource_guard_aborted']:
            raise ValueError('Preserve failed stage; no silent retry or overwrite')
        return
    logs = ROOT/config['state_root']/'logs'/identity
    logs.mkdir(parents=True,exist_ok=True)
    command = [config['python'],'-X','utf8','-u','-m',module,'--config',stage_path]
    if phase!='native': command.append('--stage-controller')
    with (logs/(phase+'.stdout.log')).open('xb') as out,(logs/(phase+'.stderr.log')).open('xb') as err:
        child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    handle = psutil.Process(child.pid); creation, actual = handle.create_time(),handle.cmdline()
    update(dict(state='waiting_actual_recovery_'+phase,active_job=identity,active_pid=child.pid,
        active_create_time=creation,actual_command=actual))
    while child.poll() is None:
        try:
            current = psutil.Process(child.pid)
            if current.create_time()!=creation or current.cmdline()!=actual:
                raise ValueError('Actual recovery stage process identity changed')
            update(dict(actual_child_cpu_seconds=sum(current.cpu_times()[:2])))
        except psutil.NoSuchProcess:
            if child.poll() is None: raise
        time.sleep(config['poll_seconds'])
    if child.wait()!=0:
        raise RuntimeError('Full recovery stage failed; original attempts and outputs preserved')
    update(dict(active_pid=None,active_job=None))


def controller(config, config_path):
    base, jobs, template, preserved, routes = verify(config)
    (ROOT/config['state_root']).mkdir(parents=True,exist_ok=True)
    lane = original.resource_api()['exclusive_lock'](ROOT/config['state_root']/'controller.lock')
    if lane is None: raise RuntimeError('This exact recovery handoff already has a live owner')
    state_path = ROOT/config['state_root']/'status.json'
    prior = read(state_path) if state_path.exists() else {}
    failed = prior.get('failed_preserved',{})
    covered = {k:dict(v,completion_origin='preserved_original_full_replay') for k,v in preserved.items()}
    recovered = {}
    for path in (ROOT/config['state_root']/'completed').glob('*.json'):
        item=read(path);identity=item['canonical_id']
        if identity not in routes or item['recovery_job_id']!=routes[identity][1]['id']:
            raise ValueError('Existing recovery completion must match one exact original obligation')
        completed_review(item['post_review_configuration'],routes[identity][1],config)
        recovered[identity]=item;covered[identity]=item
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        configuration_sha256=digest(config_path),original_scope=config['original_scope'])
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat())
        write(state_path,state)
    try:
        while len(covered)<139:
            # Status IDs alone are insufficient: every newly reported original
            # execution must have successful full native and post-review proof.
            directory = ROOT/base['state_root']/'completed'
            for path in directory.glob('*.json'):
                item = read(path); identity = item['job_id']
                if identity in covered: continue
                if identity not in jobs or identity in routes:
                    raise ValueError('Inspect an unexpected original/recovery dispatch overlap')
                covered[identity] = dict(completed_review(item['post_review_configuration'],jobs[identity],config),
                    canonical_id=identity,completion_origin='original_full139_queue')
            for identity,(route,job) in routes.items():
                if identity in covered or identity in failed: continue
                if (ROOT/jobs[identity]['output_directory']/'result.json').exists():
                    raise ValueError('Inspect unexpected canonical completion before duplicate recovered execution')
                if {identity,job['id']} & original.live_stage_jobs(): continue
                final = ROOT/job['output_directory']/'result.json'
                if not final.exists(): continue
                try: result = read(final)
                except (OSError,ValueError): continue
                if result.get('status')!='completed': continue
                try:
                    effective,point_path,point,replay_template = stage_paths(config,config_path,base,template,route,job)
                    run_stage(config,point_path,'scripts.flow_matching.run_crossad_completion_replay_queue_v1','point',identity,update)
                    native_path,native = original.replay_configuration(effective,replay_template,job,point_path,point)
                    run_stage(config,native_path,'scripts.flow_matching.replay_crossad_registered_model_v3','native',identity,update)
                    original.native_terminal(native,job)
                    post = {k:copy.deepcopy(point[k]) for k in (*original.GUARDS,'original_scope','job_id')}
                    post.update(phase='post',queue=route['native_queue'],native_configuration=native_path,
                        output_directory=config['proof_root']+'/'+job['id']+'/post',
                        state_root=config['child_state_root']+'/'+job['id']+'/post',
                        source_receipts=copy.deepcopy(native['source_receipts'])+[
                            dict(path=native_path,sha256=digest(native_path)),
                            dict(path=native['state_root']+'/receipt.json',sha256=digest(native['state_root']+'/receipt.json')),
                            dict(path=str(Path(native['artifact_directory'])/'full_native_model_replay.json'),
                                sha256=digest(ROOT/native['artifact_directory']/'full_native_model_replay.json'))])
                    post_path=(Path(config['configuration_root'])/(job['id']+'_post.json')).as_posix()
                    original.exclusive_configuration(post_path,post)
                    run_stage(config,post_path,'scripts.flow_matching.run_crossad_completion_replay_queue_v1','post',identity,update)
                    # Revalidate the science-to-canonical mapping after all
                    # original model/data/metric work, before assigning credit.
                    exact_route(route,jobs[identity],read(route['native_queue']),read(base['original_queue']))
                    evidence = dict(completed_review(post_path,job,config),canonical_id=identity,
                        recovery_job_id=job['id'],native_queue=route['native_queue'],
                        completion_origin='exact_preserved_failure_recovery',additional_training_seed_slots=0)
                    exclusive_json(ROOT/config['state_root']/'completed'/(identity+'.json'),evidence)
                    recovered[identity]=evidence;covered[identity]=evidence
                except Exception as error:
                    failed[identity]=dict(error_type=type(error).__name__,error=str(error),failed_attempt_preserved=True)
                update(dict(active_pid=None,active_job=None))
            union=canonical_union(jobs,covered,routes)
            update(dict(union,state='waiting_actual_exact_recovery_and_original_full139_completion',
                verified_exact_recovery_ids=sorted(recovered),failed_preserved=failed,
                pending_exact_recovery_ids=sorted(set(routes)-set(recovered))))
            if len(covered)<139: time.sleep(config['poll_seconds'])
        # A family completion is not completion of the45-paper objective.
        verify(config)
        for identity,item in covered.items():
            if 'post_review_configuration' in item:
                completed_review(item['post_review_configuration'],routes[identity][1] if identity in routes else jobs[identity],config)
        exclusive_json(ROOT/config['state_root']/'all139_verified_union.json',dict(
            canonical_union(jobs,covered,routes),captured_utc=datetime.now(timezone.utc).isoformat(),
            completed_original_executions=covered,original_scope=config['original_scope']))
        update(dict(state='all139_original_full_native_and_independent_executions_verified'))
    finally: lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(args.config)
    if args.probe_only:
        _,jobs,_,covered,routes=verify(config)
        print(json.dumps(dict(canonical_union(jobs,covered,routes),exact_recovery_routes=len(routes),
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules)))
    else: controller(config,args.config)


if __name__=='__main__': main()
