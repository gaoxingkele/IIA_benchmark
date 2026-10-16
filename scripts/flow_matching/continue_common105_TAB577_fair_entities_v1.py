"""Fair metadata continuation of the complete matched577 TAB numerical program.

Reuse the previously validated guarded entity-boundary handoff. The original
matched worker command, scientific files, checkpoints and metric bodies stay
unchanged; only the metadata scheduler yields the existing lock for20seconds.
"""
import argparse
import ast
from copy import deepcopy
import json
import sys
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import continue_TAB878_fair_entities_v1 as fair
from scripts.flow_matching import run_common105_matched577_TAB_v1 as matched
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic

ROOT,read,sha=fair.ROOT,fair.read,fair.sha
MODULE='scripts.flow_matching.continue_common105_TAB577_fair_entities_v1'
CONFIG='configs/runtime/fm_common105_TAB577_fair_entity_continuation.v1.json'
TEST='tests/test_fm_common105_TAB577_fair_entity_continuation_v1.py'
BASE='experiments/runs/fm_common105_TAB577_fair_entity_continuation_v1'
EXPECTED=dict(pid=32712,create_time=1792099591.5344648,command=[
    'C:\\Users\\10175\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
    '-B','-X','utf8','-u','-m',matched.MODULE])

def registration():
    c=read(ROOT/matched.CONFIG);matched.verify(c)
    pins={p['path'] for p in c['source_receipts']}
    pins.update([matched.CONFIG,fair.CONFIG, fair.MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/continue_TAB878_fair_entities_v2.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',MODULE.replace('.','/')+'.py',TEST])
    return dict(schema_version=1,original_configuration=matched.CONFIG,
        original_configuration_sha256=sha(ROOT/matched.CONFIG),state_root=BASE,previous_owner=deepcopy(EXPECTED),
        required_cases=577,required_label_views=1726,required_entity_label_views=77490,
        settings=c['settings'],resource_lock=c['resource_lock'],cpu_threads=c['CPU_threads'],
        handoff_poll_seconds=.25,inter_entity_yield_seconds=20,maximum_new_attempts_per_fragment=1,
        retain_all_old_successful_and_failed_outcomes=True,scientific_artifact_root=c['artifact_root'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        citations=c['citations'],original_scope=c['original_scope'],all_original_experiments_complete=False,
        boundary='All577matched cases/1726label views/77490entity-label views retained. '
            'Exact original worker/config/commands,all21metrics/201calls/threshold grids/VUS/cell journals, '
            'source/native bindings,16/28/16GiB3/6caps and original audit lock unchanged. '
            'Takeover requires actual audit lock ownership,twice checked exact parent PID/create-time/full command, '
            'no numerical descendants and terminal exact child guard. Never stops a numerical child. '
            'Revalidate and reuse existing successful fragments,preserve all negatives,no retry. '
            'Only metadata parent yields20seconds after new original complete entities;atomic status writer. '
            'All original45/681/273/702and873matched comparison scope remains;no new training or completed-case credit.')

def verify(c):
    if c!=registration():raise ValueError('All577matched cases and unchanged original numerical/resource contract required')
    original=read(ROOT/matched.CONFIG);matched.verify(original);return original

def source_api():
    return SimpleNamespace(ROOT=ROOT,CONFIG=matched.CONFIG,MODULE=matched.MODULE,REPORT=matched.REPORT,
        paths=matched.original.paths,worker_command=matched.worker_command,checked_fragment=matched.checked_fragment,
        ready_case=matched.ready_case,source_status=matched.source_status)

def namespace():
    ns=dict(fair.__dict__,CONFIG=CONFIG,MODULE=MODULE,BASE=BASE,EXPECTED=EXPECTED,
        original=source_api(),verify=verify,registration=registration,write=atomic)
    # Numerical stage/fragment/handoff code objects stay byte-identical; only
    # controller counts and their status text refer to the matched roster.
    for name in ['terminal_child','handoff','fragment_action','stage']:
        old=getattr(fair,name)
        ns[name]=FunctionType(old.__code__,ns,name,old.__defaults__,old.__closure__)
    tree=ast.parse(open(fair.__file__,encoding='utf-8').read())
    class Counts(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in (878,2057):
                return ast.copy_location(ast.Constant({878:577,2057:1726}[node.value]),node)
            if isinstance(node.value,str):
                return ast.copy_location(ast.Constant(node.value.replace('878','577').replace('2057','1726')),node)
            return node
    nodes=[Counts().visit(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['boundary_allowed','controller']]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(fair.__file__),'exec'),ns)
    return ns

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    if a.register:fair.exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);original=verify(c)
    if a.probe_only:
        print(json.dumps(dict(passed=True,cases=577,label_views=1726,entity_label_views=77490,
            original_worker_commands_unchanged=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    else:namespace()['controller'](c,original)

if __name__=='__main__':main()
