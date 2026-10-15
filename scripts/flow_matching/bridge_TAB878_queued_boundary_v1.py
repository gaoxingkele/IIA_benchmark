"""Queue for an entity release; briefly expose a safe handoff window.

No process is suspended or terminated here. Only the already configured audit
lock is queued. All source/metric work, limits and completed outcomes stay intact.
"""
import argparse
from datetime import datetime,timezone
import json
import os
import time
from scripts.flow_matching import continue_TAB878_fair_entities_v2 as successor
from scripts.flow_matching.win32_queued_audit_lock_v1 import acquire
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic

ROOT,read,sha=successor.ROOT,successor.read,successor.sha
MODULE='scripts.flow_matching.bridge_TAB878_queued_boundary_v1'
CONFIG='configs/runtime/fm_TAB878_queued_boundary_bridge.v1.json'
TEST='tests/test_fm_TAB878_queued_boundary_bridge_v1.py'
BASE='experiments/runs/fm_TAB878_queued_boundary_bridge_v1'


def registration():
    c=read(ROOT/successor.CONFIG);original=read(ROOT/successor.previous.original.CONFIG)
    paths={successor.CONFIG,successor.previous.original.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/win32_queued_audit_lock_v1.py','tests/test_fm_win32_queued_audit_lock_v1.py'}
    return dict(successor_configuration=successor.CONFIG,previous_owner=c['previous_owner'],
        successor_handle=read(ROOT/c['state_root']/'launch.json'),successor_state_root=c['state_root'],state_root=BASE,
        resource_lock=original['resource_lock'],settings=original['settings'],required_cases=878,required_label_views=2057,
        maximum_hold_seconds=2,successor_observation_seconds=3,poll_seconds=.05,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(paths)],
        original_scope=original['original_scope'],all_original_experiments_complete=False,
        documentation='https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex',
        boundary='Kernel queued byte0 lock interoperates with original msvcrt audit lock;no file byte written by queued helper. '
            'Wait for actual entity lock release,hold at most2seconds only to observe original parent blocked by this lock, '
            'then release for existing1second successor versus original10second retry. No process suspend/termination, '
            'numeric child action,budget/threshold/guard/metric change or new scientific repetition. '
            'If successor already handed off,exit;otherwise queue again rather than restart any calculation.')


def verify(c):
    if c!=registration():raise ValueError('Exact original audit lock,all878 scope,and bounded metadata-only window required')
    return successor.verify(read(ROOT/successor.CONFIG))


def exact_process(handle):
    import psutil
    p=psutil.Process(handle['pid'])
    if p.create_time()!=handle['create_time'] or p.cmdline()!=handle['command']:
        raise ValueError('Exact expected metadata process identity changed')
    return p


def window_ready(state,expected,descendants,since):
    return (state.get('pid')==expected['pid'] and state.get('create_time')==expected['create_time']
        and state.get('active_pid') is None and state.get('state')=='waiting_for_heavy_lock'
        and state.get('shared_lock_owned') is False
        and datetime.fromisoformat(state['captured_utc']).timestamp()>=since
        and all(successor.previous.allowed_console_child(d['name'],d['exe']) for d in descendants))


def controller(config,c):
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=resource_api()['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Queued boundary bridge already has an owner')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous bridge attempt')
    actor=psutil.Process();state=dict(pid=actor.pid,create_time=actor.create_time(),active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {},**kw,captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    target=ROOT/config['successor_state_root'];source=ROOT/c['state_root']/'status.json'
    def completed():return (target/'handoff.json').exists()
    def waiting(value):
        exact_process(config['successor_handle'])
        if completed():raise HandoffCompleted()
        exact_process(config['previous_owner']);update(value)
    count=0
    try:
        while not completed():
            exact_process(config['successor_handle']);exact_process(config['previous_owner'])
            lock=None
            try:
                lock=acquire(ROOT/config['resource_lock'],waiting)
                since=time.time();deadline=time.monotonic()+config['maximum_hold_seconds'];observed=False
                while time.monotonic()<deadline and not completed():
                    p=exact_process(config['previous_owner'])
                    descendants=[dict(name=v.name(),exe=v.exe()) for v in p.children(recursive=True)]
                    status=read(source)
                    if window_ready(status,config['previous_owner'],descendants,since):
                        observed=True;break
                    time.sleep(config['poll_seconds'])
                count+=1;update(state='releasing_bounded_window_for_original_waiters',windows=count,original_wait_observed=observed)
            except HandoffCompleted:break
            finally:
                if lock is not None:lock.close()
            deadline=time.monotonic()+config['successor_observation_seconds']
            while not completed() and time.monotonic()<deadline:time.sleep(.1)
        update(state='existing_fair_successor_handoff_observed',handoff_completed=completed(),windows=count)
    finally:owner.close()


class HandoffCompleted(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:successor.previous.exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,cases=878)));return
    config=read(ROOT/CONFIG);c=verify(config)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=878,labels=2057)))
    else:controller(config,c)


if __name__=='__main__':main()
