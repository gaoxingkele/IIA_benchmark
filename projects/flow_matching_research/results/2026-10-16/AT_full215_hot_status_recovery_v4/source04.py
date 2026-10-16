"""Restore the missing metadata handoff argument;retain frozen215 statistics."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import types
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v2 as previous

ROOT,read,sha,exclusive=previous.ROOT,previous.read,previous.sha,previous.exclusive
MODULE='scripts.flow_matching.recover_anomaly_transformer_full215_statistics_v3'
CONFIG='configs/runtime/fm_anomaly_transformer_full215_statistics_metadata_recovery.v3.json'
TEST='tests/test_fm_anomaly_transformer_full215_statistics_metadata_recovery_v3.py'
ERROR="TypeError: handoff() missing 1 required positional argument: 'poll_seconds'"


def failure_evidence():
    c=read(ROOT/previous.CONFIG);base=ROOT/c['state_root'];launch=read(base/'launch.json')
    error=base/'controller.stderr.log'
    if ERROR not in error.read_text(encoding='utf-8'):raise ValueError('Only proven pre-handoff call-signature failure may recover')
    import psutil
    try:
        p=psutil.Process(launch['pid'])
        if p.is_running() and p.create_time()==launch['create_time']:raise ValueError('Previous metadata process still live')
    except psutil.NoSuchProcess:pass
    if any((base/n).exists() for n in ['handoff.json','handoff_identity.json','previous_status_at_verified_handoff.json','attempts','status.json']):
        raise ValueError('Previous metadata ownership or numerical attempt exists;never replay it')
    if Path(c['artifact_root']).exists():raise ValueError('Previous numerical artifact exists')
    return dict(failed_metadata_launch=str(base/'launch.json'),failed_metadata_launch_sha256=sha(base/'launch.json'),
        failed_trace=str(error),failed_trace_sha256=sha(error),expected_error=ERROR,
        previous_handoff_performed=False,previous_numerical_worker_started=False)


def registration():
    c=previous.registration();evidence=failure_evidence()
    c.update(schema_version=3,state_root='experiments/runs/fm_anomaly_transformer_full215_statistics_metadata_recovery_v3',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_full215_statistics_metadata_recovery_v3',
        preserved_pre_handoff_failure=evidence,explicit_handoff_poll_seconds=10)
    c['source_receipts']+=[dict(path=p,sha256=sha(ROOT/p)) for p in [previous.CONFIG,MODULE.replace('.','/')+'.py',TEST,
        evidence['failed_metadata_launch'],evidence['failed_trace']]]
    c['boundary']+=' V3 changes only the metadata transfer call from two arguments to the original third poll_seconds=10;all v2 numerical/statistical code and resource rules remain. Preserve the pre-handoff TypeError,which did not stop any owner or start a numerical task.'
    return c


def verify(c):
    if c!=registration():raise ValueError('Exact complete215 metadata call recovery and preserved pre-handoff failure required')
    return previous.original.verify(c)


def controller_tree():
    tree=ast.parse(inspect.getsource(previous.controller));count=0
    class Fix(ast.NodeTransformer):
        def visit_Call(self,n):
            nonlocal count
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='transfer':
                if len(n.args)!=2:raise ValueError('Unexpected prior metadata transfer signature')
                n.args.append(ast.Constant(value=10));count+=1
            return n
    tree=Fix().visit(tree)
    if count!=1:raise ValueError('Only the proven metadata transfer call may change')
    return ast.fix_missing_locations(tree)


def private_controller():
    namespace=dict(previous.controller.__globals__,MODULE=MODULE)
    exec(compile(controller_tree(),__file__,'exec'),namespace)
    return namespace['controller']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--config',default=CONFIG);p.add_argument('--probe-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--task-id');a=p.parse_args()
    if a.config!=CONFIG:raise ValueError('Only the isolated complete215 v3configuration is allowed')
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);source,cases,contract,selected=verify(c);c['_config_path']=CONFIG
    if a.probe_only:
        import sys
        print(json.dumps(dict(passed=True,required_models=500,tasks=len(selected),
            task_kinds={k:sum(t['kind']==k for t in selected.values()) for k in ['cohort','paired','macro']},
            explicit_original_handoff_poll_seconds=10,original_statistics_worker_unchanged=True,
            preserved_initial_metadata_failures=len(c['recovered_initial_preworker_task_ids']),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,numerical_worker_started=False)))
    elif a.worker:previous.original.worker(c,source,cases,contract,selected[a.task_id],selected)
    else:private_controller()(c,source,selected)


if __name__=='__main__':main()
