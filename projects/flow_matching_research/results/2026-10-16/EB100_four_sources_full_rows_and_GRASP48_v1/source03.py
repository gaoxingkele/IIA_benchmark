"""Bounded nonpreemptive preparation window using existing family locks only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import time
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive
from scripts.flow_matching import prepare_CITE13176_GEO_raw_backed_QC_v2 as preparation

MODULE='scripts.flow_matching.reserve_CITE_full_preprocessing_window_v1'
CONFIG='configs/runtime/fm_CITE13176_GEO_nonpreemptive_data_window.v1.json'
TEST='tests/test_fm_CITE_full_preprocessing_window_v1.py'


def registration():
    prep=read(ROOT/preparation.CONFIG);preparation.verify(prep)
    sources=['configs/experiments/fm_otcfm_singlecell_native_full120.v4.json',
        'configs/experiments/fm_otcfm_EB_CITE_native_full150.v2.json',
        'configs/experiments/fm_otcfm_EB100_code_referenced_native_full90.v1.json']
    replays=['configs/reproducibility/fm_otcfm_singlecell_full120_checkpoint_replay.v2.json',
        'configs/reproducibility/fm_otcfm_EB_CITE_full150_checkpoint_replay.v1.json',
        'configs/reproducibility/fm_otcfm_EB100_full90_checkpoint_replay.v1.json']
    training={read(ROOT/p)['resource_lock'] for p in sources};review={read(ROOT/p)['resource_lock'] for p in replays}
    if len(training)!=1 or len(review)!=1:raise ValueError('All existing source/replay family lock bindings must agree')
    locks=sorted(training|review)
    if len(locks)!=2 or prep['resource_lock'] in locks:raise ValueError('Never reserve the preparation own lock')
    handle_path=ROOT/prep['state_root']/'launch.json';handle=read(handle_path)
    paths=[preparation.CONFIG,str(handle_path),MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/run_tsad_queue.py',
        *sources,*replays]
    return dict(preparation_configuration=preparation.CONFIG,preparation_handle=handle,family_locks=locks,
        maximum_lock_acquisition_seconds=900,maximum_owned_window_seconds=900,initial_headroom_grace_seconds=120,
        acquisition_poll_seconds=.5,owned_poll_seconds=2,original_preparation_settings=prep['settings'],
        original_preparation_stable_samples=prep['stable_samples'],original_preparation_poll_seconds=prep['poll_seconds'],
        state_root='experiments/runs/fm_CITE13176_GEO_nonpreemptive_data_window_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],original_scope=prep['original_scope'],
        boundary='One bounded natural-release window;only acquire both actual existing training/replay locks when both idle. '
            'Release earlier lock immediately if second lock is busy;never wait while owning only one. '
            'Allow the existing CITE controller own unchanged6x5second stable admission and original32GiB gate. '
            'Keep family slots reserved through the existing full data worker until terminal guard or900seconds. '
            'Deadline/emergency/missing owner releases only metadata locks;all numerical children continue under their original guards. '
            'No process signals,source edits,resource snapshot fabrication,guard lowering,epoch/batch/data changes or retry.')


def verify(c):
    if c!=registration():raise ValueError('Exact original lock/admission bindings and bounded nonpreemptive window required')


def try_reserve(exclusive_lock,paths):
    handles=[]
    try:
        for p in sorted(paths,key=str):
            h=exclusive_lock(p)
            if h is None:
                for previous in reversed(handles):previous.close()
                return None
            handles.append(h)
        return handles
    except BaseException:
        for h in reversed(handles):h.close()
        raise


def window_action(terminal_guard,producer_live,worker_live,headroom,emergency_ok,elapsed,limit,grace):
    if terminal_guard:return 'release_actual_preparation_terminal_guard'
    if not producer_live:return 'release_missing_exact_preparation_owner'
    if not emergency_ok:return 'release_emergency_headroom_metadata_only'
    if elapsed>=limit:return 'release_window_deadline_numeric_worker_continues'
    if elapsed>=grace and not worker_live and not headroom:return 'release_no_original_headroom_after_grace'
    return 'hold_original_family_locks_for_full_preparation'


def observe(c):
    import psutil
    prep=read(ROOT/c['preparation_configuration']);base=ROOT/prep['state_root'];guard=base/'guard_receipt.json'
    terminal=guard.exists();parent=False;worker=False;child_handle=None
    if not terminal:
        try:
            p=psutil.Process(c['preparation_handle']['pid'])
            parent=p.is_running() and p.create_time()==c['preparation_handle']['create_time']
        except psutil.NoSuchProcess:pass
    if (base/'status.json').exists():
        state=read(base/'status.json');h=state.get('exact_child_handle')
        if h and state.get('active_pid'):
            try:
                p=psutil.Process(h['pid'])
                worker=p.is_running() and p.create_time()==h['create_time'] and p.cmdline()==h['command']
                if worker:child_handle=h
            except psutil.NoSuchProcess:pass
    return dict(actual_terminal_guard=terminal,actual_producer_live=parent,actual_worker_live=worker,actual_worker_handle=child_handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');a=parser.parse_args()
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    api=resource_api();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('One exact full preprocessing window already owned')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve earlier window attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),numerical_children_signalled=False,
        original_guards_changed=False,owned_family_locks=False,all_original_experiments_complete=False)
    def update(**v):state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    handles=None;started=time.monotonic();acquired=None
    try:
        while handles is None:
            observation=observe(c)
            if observation['actual_terminal_guard'] or not observation['actual_producer_live']:
                update(state='no_window_needed_or_source_owner_missing',**observation);return
            if time.monotonic()-started>=c['maximum_lock_acquisition_seconds']:
                update(state='acquisition_deadline_no_numeric_job_changed',**observation);return
            handles=try_reserve(api['exclusive_lock'],[ROOT/p for p in c['family_locks']])
            if handles is None:
                update(state='waiting_both_families_naturally_idle_no_lock_owned',**observation)
                time.sleep(c['acquisition_poll_seconds'])
        acquired=time.monotonic()
        while True:
            observation=observe(c);snapshot=api['resource_snapshot']();elapsed=time.monotonic()-acquired
            action=window_action(observation['actual_terminal_guard'],observation['actual_producer_live'],observation['actual_worker_live'],
                api['has_headroom'](snapshot,c['original_preparation_settings']),
                snapshot['commit_available_bytes']>=c['original_preparation_settings']['emergency_commit_headroom_bytes'],
                elapsed,c['maximum_owned_window_seconds'],c['initial_headroom_grace_seconds'])
            update(state=action,owned_family_locks=True,owned_window_elapsed_seconds=elapsed,**observation,**snapshot)
            if action.startswith('release_'):break
            time.sleep(c['owned_poll_seconds'])
    finally:
        if handles:
            for h in reversed(handles):h.close()
        if acquired is not None:
            exclusive(base/'window_receipt.json',dict(last_state=state,all_metadata_locks_released=True,
                owned_seconds=time.monotonic()-acquired,configuration_sha256=sha(ROOT/CONFIG),
                numerical_children_signalled=False,original_guards_changed=False))
            update(owned_family_locks=False,state='window_released_original_queues_continue')
        owner.close()


if __name__=='__main__':main()
