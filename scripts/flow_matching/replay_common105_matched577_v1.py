"""Original six native/review implementations bound to all577 matched cases."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import audit_common105_matched577_labels_v1 as scalar

ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write
MODULE='scripts.flow_matching.replay_common105_matched577_v1'
CONFIG='configs/reproducibility/fm_common105_matched577_native_components.2026-10-16_v1.json'
TEST='tests/test_fm_common105_matched577_native_components_v1.py'
SPECS=dict(
    fm=('replay_original_full165_fm_objectives_v1',('objectives','entropic'),110,330),
    neural=('replay_original_full510_neural_v1',('neural',),340,1020),
    baselines=('replay_original_full105_baselines_v1',('baselines',),65,190),
    reflow=('replay_original_full60_reflow_v1',('reflow',),40,120),
    tranad64=('replay_original_full15_tranad_v1',('tranad64',),10,30),
    moment=('replay_original_full18_moment_v1',('moment',),12,36))
CONFIGS={name:'configs/reproducibility/fm_common105_native577_'+name+'.2026-10-16_v1.json' for name in SPECS}


def component(family):return importlib.import_module('scripts.flow_matching.'+SPECS[family][0])


def family_registration(family):
    module=component(family);old=read(ROOT/module.CONFIG);source=read(ROOT/scalar.CONFIG)
    groups=SPECS[family][1];cases=[c for c in source['cases'] if c['family'] in groups]
    if len(cases)!=SPECS[family][2] or sum(len(c['required_label_policies']) for c in cases)!=SPECS[family][3]:
        raise ValueError('Every new matched case and every original label view required')
    pins={r['path'] for r in old['source_receipts']}
    pins.update([module.CONFIG,scalar.CONFIG,scalar.MODULE.replace('.','/')+'.py',MODULE.replace('.','/')+'.py',TEST])
    pins.update(r['path'] for r in source['source_receipts'])
    return dict(old,upstream_scalar_configuration=scalar.CONFIG,source_execution=source['source_execution'],cases=cases,
        native_family=family,required_family_native_cases=len(cases),
        required_family_label_views=sum(len(c['required_label_policies']) for c in cases),
        other_family_cases=873-len(cases),source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_native577_v1/'+family,
        state_root='experiments/runs/fm_common105_native577_v1/'+family,
        boundary='New matched105 data/source binding only;original complete native/review code bodies, '
            'numerical observers,capacity,epochs,MC,solver,precision,device,threads and every independent math tolerance unchanged. '
            'All577new cases/1726labels belong to comparison873/2037with5existingDCD+291unchangedSMDreferences. '
            'Require actual common105 preparation,successful source guard and complete new scalar proof before replay. '
            'All original resource policies/locks/VRAM/UUID/phasegates retained;no source edits,fit or retry. '
            'Original historical author-fidelity limits remain;fullTAB and paired5trainingseed comparison remain separate. '
            +old['boundary'])


def verify_family(family,c):
    if c!=family_registration(family):raise ValueError('Complete matched native family and unchanged original numerical/resource program required')
    return c['cases']


def registration():
    items=[]
    for name in SPECS:
        c=read(ROOT/CONFIGS[name]);verify_family(name,c)
        items.append(dict(family=name,configuration=CONFIGS[name],sha256=sha(ROOT/CONFIGS[name]),
            cases=c['required_family_native_cases'],label_views=c['required_family_label_views']))
    if sum(x['cases'] for x in items)!=577 or sum(x['label_views'] for x in items)!=1726:
        raise ValueError('All577new native cases and1726labels required')
    original=read(ROOT/scalar.CONFIG)
    return dict(schema_version=1,families=items,required_cases=577,required_label_views=1726,
        source_scalar_configuration=scalar.CONFIG,controller_python=original['controller_python'],
        state_root='experiments/runs/fm_common105_native577_v1/dispatcher',
        original_matched_comparison=original['original_matched_comparison'],original_scope=original['original_scope'],
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('Every one of the six complete matched-native families required')


def paths(c,case,phase):
    if phase not in ['native','review']:raise ValueError('Explicit native/review stage required')
    return ROOT/c['state_root']/'attempts'/case['id']/phase,Path(c['artifact_root'])/case['id']/phase


def worker_command(family,case,phase):
    if phase not in ['native','review']:raise ValueError('Explicit complete native/review phase required')
    return [str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',MODULE,
        '--family',family,'--job-id',case['id'],'--stage',phase]


def ready_scalar(c,case):
    previous=read(ROOT/scalar.CONFIG);path=Path(previous['artifact_root'])/case['id']/scalar.REPORT
    guard=ROOT/previous['state_root']/'attempts'/case['id']/'guard_receipt.json'
    if not path.exists() or not guard.exists():return None
    p,g=read(path),read(guard)
    if (not p['passed'] or p['case']!=case or p['configuration_sha256']!=sha(ROOT/scalar.CONFIG)
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=scalar.worker_command(case)
            or not p['source_metrics_exactly_recomputed'] or not p['all_original_arrays_checked']
            or set(p['label_view_results'])!=set(case['required_label_policies'])
            or not p['all_validation_thresholds_and_predictions_unchanged']):
        raise ValueError('Actual complete new matched scalar proof and exact successful guard required')
    binding=scalar.source_binding(c,case)[-1]
    if binding!=p['source_binding']:raise ValueError('Matched source/data/scalar lineage changed')
    return dict(path=str(path),sha256=sha(path),guard_sha256=sha(guard),source_binding=binding)


def namespace(family):
    module=component(family)
    api=SimpleNamespace(**dict(vars(scalar.original),CONFIG=scalar.CONFIG,source_binding=scalar.source_binding))
    ns=dict(vars(module),ROOT=ROOT,CONFIG=CONFIGS[family],MODULE=MODULE,scalar=api,paths=paths,ready_scalar=ready_scalar,
        worker_command=lambda case,phase:worker_command(family,case,phase),verify=lambda c:verify_family(family,c))
    for name in ['ready_native','native_worker','review_worker']:
        fn=getattr(module,name);ns[name]=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__)
    return ns


def bound_controller(family):
    module=component(family);ns=namespace(family);old=read(ROOT/module.CONFIG)
    mapping={old['required_family_native_cases']:SPECS[family][2],old['required_family_label_views']:SPECS[family][3]}
    node=next(n for n in ast.parse(Path(module.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class ScopeNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,int) and not isinstance(n.value,bool) and n.value in mapping:
                return ast.copy_location(ast.Constant(mapping[n.value]),n)
            if isinstance(n.value,str):
                value=n.value
                for a,b in mapping.items():value=value.replace(str(a),str(b))
                return ast.copy_location(ast.Constant(value),n)
            return n
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ScopeNames().visit(node)],type_ignores=[])),str(module.__file__),'exec'),ns)
    return ns['controller']


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing matched577 native dispatcher preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous dispatcher attempt')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=577,required_label_views=1726,
        family_handles=[],all_original_experiments_complete=False)
    try:
        verify(c)
        for item in c['families']:
            name=item['family'];path=base/name;path.mkdir(parents=True,exist_ok=False)
            command=[c['controller_python'],'-B','-X','utf8','-u','-m',MODULE,'--family',name]
            with (path/'stdout.log').open('xb') as out,(path/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            h=dict(family=name,pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(path/'launch.json',h);state['family_handles'].append(h)
            state.update(state='matched_native_metadata_controllers_launched_actual_replays_pending',captured_utc=datetime.now(timezone.utc).isoformat())
            atomic(base/'status.json',state)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--family',choices=list(SPECS));p.add_argument('--job-id');p.add_argument('--stage',choices=['native','review']);a=p.parse_args()
    if a.register:
        for f in SPECS:write(ROOT/CONFIGS[f],family_registration(f))
        write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=577,label_views=1726)));return
    if a.family:
        c=read(ROOT/CONFIGS[a.family]);cases=verify_family(a.family,c)
        if a.probe_only:print(json.dumps(dict(passed=True,family=a.family,cases=len(cases),label_views=c['required_family_label_views'],
            numpy_imported='numpy' in __import__('sys').modules,torch_imported='torch' in __import__('sys').modules)))
        elif a.job_id:
            if a.stage is None:raise ValueError('Explicit full native/review stage required')
            namespace(a.family)['native_worker' if a.stage=='native' else 'review_worker'](c,next(x for x in cases if x['id']==a.job_id))
        else:bound_controller(a.family)(c,cases)
    else:
        c=read(ROOT/CONFIG);verify(c)
        if a.probe_only:print(json.dumps(dict(passed=True,cases=577,label_views=1726,families=6,
            numpy_imported='numpy' in __import__('sys').modules,torch_imported='torch' in __import__('sys').modules)))
        else:controller(c)


if __name__=='__main__':main()
