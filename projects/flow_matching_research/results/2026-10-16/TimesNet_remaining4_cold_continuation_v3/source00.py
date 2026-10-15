"""Keep all eight original TimesNet audits, release idle parent NN imports.

Numerical workers remain the frozen original v2 entry with identical arguments,
stage configurations, resource caps, checkpoints and bitwise score conditions.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType
from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as original
from scripts.flow_matching import run_idle_exact_command_cached_handoff_v3 as handoff

ROOT,read,write=original.ROOT,original.read,original.write
MODULE='scripts.flow_matching.continue_timesnet_remaining4_cold_v3'
CONFIG='configs/runtime/fm_timesnet_remaining4_cold_continuation.v3.json'
TEST='tests/test_fm_timesnet_remaining4_cold_continuation_v3.py'
BASE='experiments/runs/fm_timesnet_remaining4_cold_continuation_v3'
EXPECTED=dict(pid=33632,create_time=1792082282.488604,command=[
    'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe','-B','-X','utf8','-u','-m',original.MODULE])
COMPLETED=[dict(id=original.identity(1104),mode=m) for m in ['point','native']]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(65536),b''):value.update(b)
    return value.hexdigest()


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def worker_command(stage,path):
    return [stage['python'],'-B','-X','utf8','-u','-m',original.MODULE,'--worker','--config',str(path)]


def checked_stage(config,job,mode):
    stage=original.stage_config(config,job,mode);attempt=ROOT/stage['state_root'];output=Path(stage['output_directory'])
    path=attempt/'frozen_stage.json';guard=attempt/'receipt.json';proof=output/'independent_full_saved_score_verification.json'
    frozen,g,p=read(path),read(guard),read(proof)
    if (frozen!=stage or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted']
            or g['proof_sha256']!=sha(proof) or g['exact_child_handle']['command']!=worker_command(stage,path)
            or not p['passed'] or p['configuration_sha256']!=sha(path) or p['configuration']!=str(path)
            or p['canonical_id']!=job['id'] or p['mode']!=mode or p['parameters']!=28132633
            or not p['all_saved_streams_verified'] or not p['all_original_point_core_event_bootstrap_metrics_exact']
            or p['original_scope']!=config['original_scope'] or p['full_entities']!=51
            or p['full_test_points']!=408160 or p['full_validation_points']!=26684 or p['full_train_points']!=101565
            or p['additional_training_seed_slots']!=0 or p['additional_optimizer_updates']!=0):
        raise ValueError('Complete immutable original stage and exact successful guard required')
    source=ROOT/job['output_directory'];result=read(source/'result.json')
    if (sha(source/'result.json')!=p['original_result_sha256'] or result['job']!=job
            or sha(source/'model.pt')!=p['original_checkpoint_sha256'] or result['checkpoint_sha256']!=p['original_checkpoint_sha256']
            or sha(source/'scores.npz')!=p['original_scores_sha256'] or result['scores_sha256']!=p['original_scores_sha256']):
        raise ValueError('Every actual source result/checkpoint/score byte must remain identical')
    for pin in stage['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Original stage source/data changed')
    if mode=='native' and (not p['full_native_checkpoint_reinference_passed'] or not p['all_replayed_timestamp_scores_bitwise_equal']
            or sha(output/'replayed_scores.npz')!=p['replay_scores_sha256'] or p['replay_scores_sha256']!=p['original_scores_sha256']):
        raise ValueError('Every existing native score byte must remain exact')
    if mode=='point' and (p['full_native_checkpoint_reinference_passed'] or p['replay_scores_sha256'] is not None):
        raise ValueError('Point-only evidence cannot be relabeled as native replay')
    return dict(id=job['id'],mode=mode,proof=str(proof),proof_sha256=sha(proof),guard_sha256=sha(guard),
        configuration_sha256=sha(path),source_checkpoint_sha256=p['original_checkpoint_sha256'],source_scores_sha256=p['original_scores_sha256'])


def registration():
    c=read(ROOT/original.CONFIG);jobs=original.verify(c)
    reused=[checked_stage(c,jobs[r['id']],r['mode']) for r in COMPLETED]
    paths={original.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_idle_exact_command_cached_handoff_v3.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py'}
    return dict(original_configuration=original.CONFIG,original_configuration_sha256=sha(ROOT/original.CONFIG),
        original_scope=c['original_scope'],required_original_stages=8,required_original_seeds=c['required_original_seed_cohort'],
        previous_owner=EXPECTED,previous_state_path=c['state_root']+'/status.json',state_root=BASE,
        initial_completed=COMPLETED,reused_actual_stages=reused,settings=c['settings'],resource_lock=c['resource_lock'],
        cpu_threads=c['cpu_threads'],modes=c['modes'],jobs=c['jobs'],poll_seconds=10,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        citation=c['paper_citation'],data_citation=c['data_citation'],all_original_experiments_complete=False,
        boundary='All original8stages/fourremainingseeds plus existing1103 retained. Same51entitysubset,not canonical NASA full109. '
            'Original worker/config/arguments/precision/batch/threads/caps/lock/bitwise conditions unchanged. '
            'Rehash and reference existing1104point/native;no new replay or training credit. '
            'Only verified idle parent without numerical descendants is handed off. Full binding/numerical packages stay in original worker. '
            'Preserve any partial/failed attempt without retry. Full45/681/273/702research remains.')


def verify(c):
    if c!=registration():raise ValueError('Every original audit,resource policy,source and retained certificate required')
    return read(ROOT/original.CONFIG)


def idle_allowed(state,expected,descendants):
    return (state.get('state')=='waiting_actual_original_training' and state.get('completed')==COMPLETED
        and not state.get('failed_preserved')
        and handoff.idle_allowed(dict(state,state='waiting_for_memory'),expected,descendants))


def perform_handoff(config,base):
    role=dict(previous_owner=config['previous_owner'],previous_state_path=config['previous_state_path'])
    return FunctionType(handoff.handoff.__code__,dict(handoff.__dict__,idle_allowed=idle_allowed))(role,base,config['poll_seconds'])


def controller(config):
    import psutil
    c=verify(config);jobs=original.verify(c)
    from scripts.flow_matching.run_cached_resource_monitor_controller_v1 import install_cached_loader
    install_cached_loader(lambda value:None)
    api=original.original.resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing cold continuation owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior continuation attempt')
    oldlock=None;done=list(COMPLETED);failed={};actor=psutil.Process()
    state=dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),active_pid=None,required_stages=8,
        all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {},**kw,completed=list(done),failed_preserved=dict(failed),captured_utc=datetime.now(timezone.utc).isoformat())
        write(base/'status.json',state)
    try:
        update(state='verifying_exact_idle_handoff')
        perform_handoff(config,base)
        oldlock=api['exclusive_lock'](ROOT/c['state_root']/'controller.lock')
        if oldlock is None:raise RuntimeError('Another original audit owner appeared;never duplicate stages')
        verify(config)
        update(state='retained_two_original_successful_stages')
        for job in jobs.values():
            try:
                for mode in c['modes']:
                    identity=dict(id=job['id'],mode=mode)
                    if identity in done:continue
                    while not original.source_ready(c,job):
                        update(state='waiting_actual_original_training',job=job['id'],mode=mode,active_pid=None);time.sleep(config['poll_seconds'])
                    stage=original.stage_config(c,job,mode);attempt=ROOT/stage['state_root'];output=Path(stage['output_directory'])
                    if attempt.exists() or output.exists():raise FileExistsError('Preserve existing full/partial stage without retry')
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
                        verify(config);original.source_ready(c,job)
                        attempt.mkdir(parents=True,exist_ok=False);path=attempt/'frozen_stage.json';write(path,stage)
                        command=worker_command(stage,path)
                        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            write(attempt/'launch.json',handle);update(state='running_original_full_'+mode,active_pid=child.pid,exact_child_handle=handle)
                            code,usage=original.original.capped_wait(child,stage,api,update)
                        proof=output/'independent_full_saved_score_verification.json'
                        passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                        write(attempt/'receipt.json',dict(exit_code=code,passed=passed,**usage,exact_child_handle=handle,
                            proof_sha256=sha(proof) if proof.exists() else None))
                        if not passed:raise ValueError('Original full stage failed;preserve every attempt')
                        checked_stage(c,job,mode)
                    done.append(identity);update(state='between_original_full_audits',active_pid=None);time.sleep(2)
            except Exception as error:failed[job['id']]=str(error);update(state='failed_original_seed_preserved',active_pid=None)
        update(state='all_remaining_stages_terminal_other_research_remains',active_pid=None)
    finally:
        if oldlock is not None:oldlock.close()
        owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,required_stages=8,reused_original_stages=2)));return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:
        import psutil
        print(json.dumps(dict(passed=True,required_stages=8,reused_original_stages=2,numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,private_bytes=psutil.Process().memory_info().private)))
    else:controller(c)


if __name__=='__main__':main()
