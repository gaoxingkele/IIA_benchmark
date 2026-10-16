"""One explicit native epoch280 continuation; preserve all285 CFM conditions.

CFM-TS https://openreview.net/forum?id=Hqn4Aj7xrQ. Frozen paper-derived local
model; historical author equivalence unproved. No automatic numerical retry.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace
from scripts.flow_matching.run_cfm_checkpointed_seed1106_parent_v1 import ROOT,parent_namespace
from scripts.flow_matching.timegan_native_common_v1 import read,sha

MODULE='scripts.flow_matching.run_cfm_node1106_epoch280_resume_v1'
CONFIG='configs/runtime/fm_cfm_node1106_epoch280_resume.v1.json'
QUEUE='configs/experiments/fm_cfm_node1106_epoch280_resume_queue.v1.json'
REPLAY='configs/runtime/fm_cfm_node1106_epoch280_resume_full285_replay.v1.json'
PREVIOUS='configs/runtime/fm_cfm_node1106_complete_epoch_full_retry.v14.json'
PRIMARY='configs/runtime/fm_cfm_completion_replay_queue.v4.json'
TEST='tests/test_fm_cfm_node1106_epoch280_resume_v1.py'
CANONICAL='cfm_ts_original__main_text__node_e400__pendulum__seed1106'
BASE='experiments/runs/fm_cfm_node1106_epoch280_resume_v1'
ARTIFACT='F:/aicoding/IIA_Data/experiments/cfm_node1106_epoch280_resume_v1'


def exclusive(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def check_boundary(terminal,progress,ledger):
    if (terminal['exit_code']==0 or terminal['complete'] or terminal['full_original400_checkpointed_result_verified']
            or not terminal['resource_observations']['resource_guard_aborted']
            or progress['epoch']!=284 or progress['optimizer_updates']!=284 or progress['required_epochs']!=400
            or [r['completed_original_epochs'] for r in ledger]!=list(range(10,281,10))):
        raise ValueError('Exact terminal resource abort,284 partial progress and all28 complete epoch checkpoints required')
    for r in ledger:
        if (r['optimizer_updates']!=r['completed_original_epochs'] or r['required_original_epochs']!=400
                or not r['complete_original_epoch_boundary']
                or not r['all_native_model_optimizer_rng_gradients_history_and_order_values_reread_bitwise']
                or not r['no_training_state_or_random_state_changed_by_save'] or r['additional_optimizer_updates']!=0):
            raise ValueError('Every retained complete native epoch must remain bitwise verified')
    return ledger[-1]


def registration():
    import psutil
    old=read(ROOT/PREVIOUS);values=parent_namespace();queue,failed=values['verify_registration'](old)
    terminal_path=old['state_root']+'/receipt.json';progress_path=failed['output_directory']+'/progress.json'
    ledger_path=str(Path(failed['artifact_directory'])/'completed_full_epoch_checkpoints.json')
    terminal,progress,ledger=read(ROOT/terminal_path),read(ROOT/progress_path),read(ledger_path)
    checkpoint=check_boundary(terminal,progress,ledger)
    handles=[dict(pid=39920,create_time=1791829639.2058773),dict(pid=33832,create_time=1792097263.141405),
        dict(pid=36912,create_time=1792097263.1796665)]
    for h in handles:
        try:
            p=psutil.Process(h['pid'])
            if abs(p.create_time()-h['create_time'])<.01:raise ValueError('Prior numerical owner/worker is still live')
        except psutil.NoSuchProcess:pass
    if (ROOT/failed['output_directory']/'result.json').exists():raise ValueError('Never resume an already completed condition')
    sources=[PREVIOUS,old['queue'],terminal_path,progress_path,ledger_path,PRIMARY,
        MODULE.replace('.','/')+'.py',TEST,'scripts/flow_matching/run_cfm_metadata_permission_replay_queue_v1.py',
        'scripts/flow_matching/run_cfm_uncovered_recovery_completion_v1.py',
        'scripts/flow_matching/run_cfm_streaming_idle_handoff_v3.py']
    for r in ledger:
        if sha(r['full_state_path'])!=r['full_state_sha256']:raise ValueError('Preserved full native checkpoint changed')
        receipt=Path(r['full_state_path']).parent/'receipt.json'
        if read(receipt)!=r:raise ValueError('Exact native save receipt required')
        sources.extend([r['full_state_path'],str(receipt)])
    pins=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(sources))]
    q=deepcopy(queue);job=next(j for j in q['jobs'] if j['id']==old['job_id'])
    job.update(id=CANONICAL+'__epoch280_native_resume_v1',output_directory=BASE+'/job',artifact_directory=ARTIFACT+'/source')
    q.update(state_root=BASE,recovery_job_ids=[job['id']],source_receipts=q['source_receipts']+pins,
        recovery_selection='One explicit continuation from hashed complete epoch280. Discard unsaved281..284;restore every native state;unchanged full400 budget and500 test points. Prior failure preserved.')
    c=dict(old,queue=QUEUE,job_id=job['id'],state_root=BASE,
        resume_full_epoch_checkpoint=checkpoint['full_state_path'],resume_full_epoch_checkpoint_sha256=checkpoint['full_state_sha256'],
        source_receipts=old['source_receipts']+pins,
        previous_terminal=terminal_path,previous_progress=progress_path,previous_complete_epoch_ledger=ledger_path,
        full_saved_boundary_epoch=280,remaining_original_epochs=120,discarded_unsaved_epochs=[281,282,283,284],
        maximum_new_numerical_attempts=1,old_failed_handles=handles,all_original_experiments_complete=False)
    return c,q


def verify(c):
    expected,q=registration();expected['source_receipts'].append(dict(path=QUEUE,sha256=sha(ROOT/QUEUE)))
    if c!=expected or read(ROOT/QUEUE)!=q:raise ValueError('Exact native epoch280 restore,all285 slots and unchanged guards required')
    queue,job=parent_namespace()['verify_registration'](c)
    return [job],queue


def source_closed(c):
    path=ROOT/c['state_root']/'receipt.json'
    if not path.exists():return False
    terminal=read(path)
    if terminal['exit_code']!=0 or not terminal['complete'] or terminal['resource_observations']['resource_guard_aborted']:return False
    jobs,q=verify(c);job=jobs[0]
    result=read(ROOT/job['output_directory']/'result.json')
    restore=read(Path(job['artifact_directory'])/'complete_epoch_restore_verification.json')
    ledger=read(Path(job['artifact_directory'])/'completed_full_epoch_checkpoints.json')
    if (result['job']!=job or result['status']!='completed' or result['epochs_completed']!=400 or result['optimizer_updates']!=400
            or result['test_prediction_shape']!=[1,500,1] or restore['completed_original_epochs']!=280
            or restore['remaining_original_epochs']!=120 or not restore['full_model_optimizer_CPU_numpy_python_rng_and_order_restored_bitwise']
            or [r['completed_original_epochs'] for r in ledger]!=list(range(290,401,10))):
        raise ValueError('Complete actual restore plus400 updates and all500 final points required')
    for r in ledger:
        if sha(r['full_state_path'])!=r['full_state_sha256']:raise ValueError('Resumed full epoch checkpoint changed')
    return True


def replay_namespace():
    from scripts.flow_matching import run_cfm_metadata_permission_replay_queue_v1 as previous
    values=FunctionType(previous.namespace.__code__,dict(previous.__dict__,PRODUCER=CONFIG,
        producer=SimpleNamespace(verify_runtime=verify)))()
    selected=values['_selected_ready']
    values['_selected_ready']=lambda route,runtime:selected(route,runtime) and source_closed(read(ROOT/CONFIG))
    return values


def replay_registration(c,q):
    from scripts.flow_matching import run_cfm_uncovered_recovery_completion_v1 as supplement
    primary=read(ROOT/PRIMARY);value=deepcopy(primary)
    value.update(previous_full285_configuration=PRIMARY,producer_configuration=CONFIG,runtime_configuration=REPLAY,
        routes=supplement.expected_routes(primary,q),selected_canonical_ids=[CANONICAL],state_root=BASE+'/replay',child_state_root=BASE+'/replay/children',
        child_config_root='configs/reproducibility/fm_cfm_node1106_epoch280_resume_v1',artifact_root=ARTIFACT+'/replay',
        actual_source_guard_and_restore_gate=CONFIG,
        source_receipts=primary['source_receipts']+c['source_receipts']+[dict(path=CONFIG,sha256=sha(ROOT/CONFIG))],
        boundary='All285 routes retained;only epoch280 restored canonical NODE1106 selected for new dispatch. '
            'Historical optional terminal path retained as metadata by expected_routes;candidate gate instead requires '
            'the actual new source guard and bitwise280 restore plus complete400 updates and500 test points. '
            'Other actual primary certificates adopted without reexecution;full original native/independent numerical workers unchanged.')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--stage',choices=['replay']);a=p.parse_args()
    if a.register:
        c,q=registration();exclusive(QUEUE,q);c['source_receipts'].append(dict(path=QUEUE,sha256=sha(ROOT/QUEUE)))
        exclusive(CONFIG,c);verify(c);r=replay_registration(c,q);replay_namespace()['verify'](r);exclusive(REPLAY,r);return
    c=read(ROOT/CONFIG);verify(c)
    if a.probe_only:
        replay_namespace()['verify'](read(ROOT/REPLAY))
        print(json.dumps(dict(canonical_jobs=285,resume_epoch=280,remaining_epochs=120,required_epochs=400,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if a.stage=='replay':replay_namespace()['controller'](read(ROOT/REPLAY),REPLAY);return
    before=sys.argv
    try:sys.argv=[before[0],'--config',CONFIG];parent_namespace()['main']()
    finally:sys.argv=before


if __name__=='__main__':main()
