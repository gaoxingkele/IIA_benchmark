"""Resume the completed metadata handoff with one cold facade initialization."""
import argparse
import ast
import inspect
import json
from pathlib import Path
from scripts.flow_matching import recover_anomaly_transformer_full215_statistics_v3 as previous

ROOT,read,sha,exclusive=previous.ROOT,previous.read,previous.sha,previous.exclusive
MODULE='scripts.flow_matching.recover_anomaly_transformer_full215_statistics_v4'
CONFIG='configs/runtime/fm_anomaly_transformer_full215_statistics_metadata_recovery.v4.json'
TEST='tests/test_fm_anomaly_transformer_full215_statistics_metadata_recovery_v4.py'
ERROR='RuntimeError: Install resource facades only in a fresh cold metadata interpreter'


def failure_evidence():
    c=read(ROOT/previous.CONFIG);base=ROOT/c['state_root'];launch=read(base/'launch.json')
    error=base/'controller.stderr.log';h=read(base/'handoff.json');hi=read(base/'handoff_identity.json')
    if (ERROR not in error.read_text(encoding='utf-8') or not h['waiting_controller_only_stopped']
            or h['original_native_child_stopped'] or not hi['no_numerical_child_present']):
        raise ValueError('Only the completed idle metadata handoff and proven duplicate-facade failure may resume')
    import psutil
    for handle in [launch,c['predecessor_owner']]:
        try:
            p=psutil.Process(handle['pid'])
            if p.is_running() and p.create_time()==handle['create_time']:raise ValueError('A previous exact metadata owner is live')
        except psutil.NoSuchProcess:pass
    if any((base/n).exists() for n in ['attempts','status.json']) or Path(c['artifact_root']).exists():
        raise ValueError('A prior statistics numerical attempt exists;preserve without retry')
    return dict(failed_launch=str(base/'launch.json'),failed_launch_sha256=sha(base/'launch.json'),
        failed_trace=str(error),failed_trace_sha256=sha(error),expected_error=ERROR,
        completed_metadata_handoff=str(base/'handoff.json'),completed_metadata_handoff_sha256=sha(base/'handoff.json'),
        checked_idle_identity=str(base/'handoff_identity.json'),checked_idle_identity_sha256=sha(base/'handoff_identity.json'),
        previous_statistics_worker_started=False)


def registration():
    c=previous.registration();evidence=failure_evidence()
    c.update(schema_version=4,state_root='experiments/runs/fm_anomaly_transformer_full215_statistics_metadata_recovery_v4',
        artifact_root='F:/aicoding/IIA_Data/experiments/anomaly_transformer_full215_statistics_metadata_recovery_v4',
        preserved_duplicate_facade_failure=evidence,resource_facades_initialized_once=True,completed_metadata_handoff_repeated=False)
    paths=[previous.CONFIG,MODULE.replace('.','/')+'.py',TEST,evidence['failed_launch'],evidence['failed_trace'],
        evidence['completed_metadata_handoff'],evidence['checked_idle_identity']]
    c['source_receipts']+=[dict(path=p,sha256=sha(ROOT/p)) for p in paths]
    c['boundary']+=' V4 retains the completed metadata-only handoff and resumes in a new cold process. Initialize the original resource facade once,then pass that same API to the unchanged215task controller. No owner is stopped again;all numerical/statistical worker code,readiness fix,budgets and source guards remain unchanged.'
    return c


def verify(c):
    if c!=registration():raise ValueError('Exact completed metadata handoff and all original215tasks required')
    return previous.previous.original.verify(c)


def controller_tree():
    original=previous.previous.original
    tree=ast.parse(inspect.getsource(original.controller));count=0
    class Fix(ast.NodeTransformer):
        def visit_Assign(self,n):
            nonlocal count
            self.generic_visit(n)
            if (any(isinstance(t,ast.Name) and t.id=='api' for t in n.targets)
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                    and n.value.func.id=='install_facades'):
                n.value=ast.Name(id='existing_api',ctx=ast.Load());count+=1
            return n
    tree=Fix().visit(tree)
    if count!=1:raise ValueError('Only the duplicate metadata API initialization may change')
    return ast.fix_missing_locations(tree)


def private_controller(api):
    prior=previous.previous
    namespace=dict(prior.original.controller.__globals__,MODULE=MODULE,readiness=prior.readiness,existing_api=api)
    exec(compile(controller_tree(),__file__,'exec'),namespace)
    return namespace['controller']


def controller(c,source,tasks):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    if (base/'status.json').exists():raise FileExistsError('Preserve previous full statistics attempt')
    old=read(ROOT/previous.previous.OLD_CONFIG)
    oldlock=api['exclusive_lock'](ROOT/old['state_root']/'controller.lock')
    if oldlock is None:raise RuntimeError('Another original metadata owner appeared;never duplicate tasks')
    try:private_controller(api)(c,source,tasks)
    finally:oldlock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--config',default=CONFIG);p.add_argument('--probe-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--task-id');a=p.parse_args()
    if a.config!=CONFIG:raise ValueError('Use only the full215 v4metadata recovery registration')
    if a.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);source,cases,contract,tasks=verify(c);c['_config_path']=CONFIG
    if a.probe_only:
        import sys
        print(json.dumps(dict(passed=True,required_models=500,tasks=len(tasks),
            task_kinds={k:sum(t['kind']==k for t in tasks.values()) for k in ['cohort','paired','macro']},
            original_statistics_worker_unchanged=True,numerical_worker_started=False,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            completed_metadata_handoff_repeated=False,resource_facades_initialized_once=True)))
    elif a.worker:previous.previous.original.worker(c,source,cases,contract,tasks[a.task_id],tasks)
    else:controller(c,source,tasks)


if __name__=='__main__':main()
