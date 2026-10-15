"""Repair the proven pre-worker facade import defect in an isolated process.

All v1 label/score/metric worker code remains unchanged. Admit only v1 metadata
failures whose attempt/output never existed. Never retry a numerical attempt.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
from scripts.flow_matching import audit_timesnet_NASA_label_protocol_v1 as original

ROOT,read,sha=original.ROOT,original.read,original.sha
MODULE='scripts.flow_matching.audit_timesnet_NASA_label_protocol_v2'
CONFIG='configs/reproducibility/fm_timesnet_all5_NASA_label_protocol.2026-10-16_v2.json'
TEST='tests/test_fm_timesnet_NASA_label_protocol_v2.py'
EXPECTED_ERROR="cannot import name 'complete_result' from 'scripts.flow_matching.run_tsad_queue' (unknown location)"


def metadata_failure(config,job):
    old=read(ROOT/config['predecessor_runtime']);base=ROOT/old['state_root']
    state=base/'status.json'
    if not state.exists():return None
    failure=read(state)['failed_preserved'].get(job['id'])
    if failure is None:return None
    if failure!=EXPECTED_ERROR:raise ValueError('Only the exact proven pre-worker facade import failure may enter v2')
    if (base/'attempts'/job['id']).exists() or (Path(old['artifact_root'])/job['id']).exists():
        raise ValueError('A numerical attempt/output exists;preserve without retry')
    return dict(predecessor_runtime=config['predecessor_runtime'],case_failure=failure,
        stable_failure_sha256=hashlib.sha256(json.dumps({job['id']:failure},sort_keys=True).encode()).hexdigest(),
        no_original_worker_or_output_created=True)


def install_metadata_complete_result():
    """Supply only the missing pure source function to this new local facade."""
    facade=sys.modules['scripts.flow_matching.run_tsad_queue']
    if hasattr(facade,'complete_result'):raise ValueError('Require the actual diagnosed missing facade function')
    path=ROOT/'scripts/flow_matching/run_tsad_queue.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='complete_result')
    namespace=dict(Path=Path,json=json,sha=sha,hashlib=hashlib)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    facade.complete_result=namespace['complete_result']


def private_module():
    source=Path(original.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
    controller=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    changes=0
    for index,node in enumerate(controller.body):
        if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Name) and node.value.func.id=='install_facades'):
            controller.body.insert(index+1,ast.parse('_install_metadata_complete_result()').body[0]);changes+=1;break
    if changes!=1:raise ValueError('Require exactly one metadata facade installation point')
    module=types.ModuleType('_timesnet_label_protocol_facade_recovery_v2');module.__file__=str(Path(original.__file__))
    exec(compile(ast.fix_missing_locations(tree),str(Path(original.__file__)),'exec'),module.__dict__)
    raw_source_native=module.source_native
    def gated_source_native(config,job):
        source=raw_source_native(config,job)
        if source is None:return None
        proof=metadata_failure(config,job)
        return None if proof is None else dict(**source,pre_worker_recovery=proof)
    module.source_native=gated_source_native
    module._install_metadata_complete_result=install_metadata_complete_result
    module.MODULE,module.CONFIG,module.TEST=MODULE,CONFIG,TEST
    module.registration=registration
    return module


def registration():
    c=original.registration()
    c.update(schema_version=2,predecessor_runtime=original.CONFIG,predecessor_runtime_sha256=sha(ROOT/original.CONFIG),
        state_root='experiments/runs/fm_timesnet_all5_NASA_label_protocol_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/timesnet_all5_NASA_label_protocol_v2',
        correction='Only add pinned pure complete_result to the new process-local metadata facade after installation',
        expected_pre_worker_failure=EXPECTED_ERROR,numerical_worker_body_changed=False)
    pins={p['path']:p['sha256'] for p in c['source_receipts']}
    for path in [original.CONFIG,MODULE.replace('.','/')+'.py',TEST,'scripts/flow_matching/run_tsad_queue.py']:
        pins[path]=sha(ROOT/path)
    c['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    c['boundary']+=' v2 admits only preserved pre-worker metadata import failures with no old attempt/output;no numerical retry.'
    return c


def verify(config):
    if config!=registration():raise ValueError('Exact isolated facade correction,all5models and unchanged scientific settings required')
    return config['jobs']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.register:original.write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,models=5)));return
    config=read(ROOT/CONFIG);jobs=verify(config);private=private_module()
    if args.probe_only:
        print(json.dumps(dict(passed=True,models=5,preserved_pre_worker_failures=[j['seed'] for j in jobs if metadata_failure(config,j)],
            numerical_body_changed=False,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.worker:
        job=next(j for j in jobs if j['seed']==args.seed);private.worker(config,job,Path(config['artifact_root'])/job['id'])
    else:private.controller(config,jobs)


if __name__=='__main__':main()
