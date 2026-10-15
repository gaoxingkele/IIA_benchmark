"""Same full TAB878 fair continuation, atomic metadata status updates only."""
import argparse
import json
import sys
from types import FunctionType
from scripts.flow_matching import continue_TAB878_fair_entities_v1 as previous
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic

ROOT,read,sha=previous.ROOT,previous.read,previous.sha
MODULE='scripts.flow_matching.continue_TAB878_fair_entities_v2'
CONFIG='configs/runtime/fm_TAB878_fair_entity_continuation.v2.json'
TEST='tests/test_fm_TAB878_fair_entity_continuation_v2.py'
BASE='experiments/runs/fm_TAB878_fair_entity_continuation_v2'


def registration():
    import psutil
    old=read(ROOT/previous.CONFIG);base=ROOT/old['state_root'];h=read(base/'launch.json')
    try:
        if psutil.Process(h['pid']).create_time()==h['create_time']:raise ValueError('Previous metadata-only successor still alive')
    except psutil.NoSuchProcess:pass
    if (base/'handoff.json').exists() or (base/'handoff_identity.json').exists():raise ValueError('Prior version already handed off;inspect its actual state')
    if 'FileExistsError' not in (base/'controller.stderr.log').read_text(encoding='utf-8'):
        raise ValueError('Exact preserved metadata write failure required')
    c=previous.registration();pins={p['path'] for p in c['source_receipts']}
    pins.update([previous.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',str(base/'controller.stderr.log'),str(base/'launch.json'),str(base/'status.json')])
    return dict(c,state_root=BASE,source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        prior_metadata_failure=dict(handle=h,state_root=old['state_root'],handoff_performed=False,numerical_child_stopped=False),
        atomic_mutable_status_writer='scripts.flow_matching.audit_cfm_checkpoint_replay_v1.write',
        boundary=c['boundary']+' Preserve failedv1metadata-only attempt. Same full controller bytecode with atomic writer in private globals; '
            'original stage/metric workers stay unchanged. No scientific output or prior attempt overwritten.')


def verify(c):
    if c!=registration():raise ValueError('Full unchanged original TAB program and exact atomic-status correction required')
    source=read(ROOT/previous.original.CONFIG);previous.original.verify(source);return source


def namespace():
    return dict(previous.__dict__,CONFIG=CONFIG,MODULE=MODULE,verify=verify,write=atomic)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:previous.exclusive(ROOT/CONFIG,registration());print(json.dumps(dict(registered=True,cases=878,labels=2057)));return
    config=read(ROOT/CONFIG);c=verify(config)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=878,labels=2057,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    else:FunctionType(previous.controller.__code__,namespace())(config,c)


if __name__=='__main__':main()
