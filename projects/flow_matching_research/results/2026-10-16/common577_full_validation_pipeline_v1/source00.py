"""All577 new matched105 saved-score cases and all1726 label views."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
from types import FunctionType
from scripts.flow_matching import run_common105_matched577_v1 as source
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as original

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.audit_common105_matched577_labels_v1'
CONFIG='configs/reproducibility/fm_common105_matched577_all1726_labels.2026-10-16_v1.json'
TEST='tests/test_fm_common105_matched577_labels_v1.py'
REPORT=original.REPORT


def registration():
    src=read(ROOT/source.CONFIG);proof=source.verify_materialization(src)
    previous=read(ROOT/original.CONFIG);cases=[];pins={source.CONFIG,original.CONFIG,
        MODULE.replace('.','/')+'.py',TEST,source.MODULE.replace('.','/')+'.py',
        original.MODULE.replace('.','/')+'.py',str(Path(src['artifact_root'])/'materialization.json')}
    pins.update(r['path'] for r in previous['source_receipts'])
    pins.update(r['path'] for r in src['source_receipts'])
    for item in proof['queues']:
        queue=read(item['path']);pins.add(item['path'])
        for job in queue['jobs']:
            record=proof['full_data_binding']['records'][job['dataset']]
            cases.append(dict(id=job['id'],family=item['family'],dataset=job['dataset'],seed=job['seed'],window=job['window'],
                queue_path=item['path'],queue_sha256=item['sha256'],worker_python=queue['python'],
                required_label_policies=list(record['labels_only_controls']),
                source_guard=str(ROOT/src['state_root']/'families'/item['family']/'receipts'/(job['id']+'.json'))))
    if len(cases)!=577 or len({c['id'] for c in cases})!=577 or sum(len(c['required_label_policies']) for c in cases)!=1726:
        raise ValueError('All577new matched models and1726label views required')
    return dict(previous,source_execution=source.CONFIG,actual_materialization=str(Path(src['artifact_root'])/'materialization.json'),
        cases=cases,required_cases=577,required_label_view_results=1726,required_view_counts=dict(smap=1144,msl=582),
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_matched577_all1726_labels_v1',
        state_root='experiments/runs/fm_common105_matched577_all1726_labels_v1',
        original_matched_comparison=dict(cases=873,label_views=2037,unchanged_SMD_references=291,existing_DCD_common105=5),
        boundary='All577new matched105 actual guarded producers and1726label views;not original V2 inputs. '
            'Unchanged complete original scalar-worker code with privately bound new source data/guards. '
            'All source thresholds,metrics,events,bootstrap200 and grids exactly recomputed,all arrays/labels checked. '
            'No model training/reinference,test label selection,source edits or automatic retry. '
            'Same original16/28/16GiB3/6caps/auditlock and exact producer Python/NumPy/Torch. '
            'Native checkpoint reload,TAB and paired5trainingseed reviews remain separate; '
            'cohort873/2037includes5existingDCD+291unchangedSMDreferences,never pseudo-independent duplicates. '
            'All full45paper/681method/273data/702scalar original requirements remain.')


def verify(c):
    if c!=registration():raise ValueError('All577new matched sources/1726labels and unchanged original audit policy required')
    return c['cases']


def source_binding(c,case):
    return FunctionType(original.source_binding.__code__,dict(original.source_binding.__globals__,source=source))(c,case)


def worker(c,case):
    return FunctionType(original.worker.__code__,dict(original.worker.__globals__,CONFIG=CONFIG,source_binding=source_binding))(c,case)


def bound_controller():
    node=next(n for n in ast.parse(Path(original.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class ScopeNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,int) and not isinstance(n.value,bool) and n.value in [873,2037]:
                return ast.copy_location(ast.Constant({873:577,2037:1726}[n.value]),n)
            if isinstance(n.value,str) and ('873' in n.value or '2037' in n.value):
                return ast.copy_location(ast.Constant(n.value.replace('873','577').replace('2037','1726')),n)
            return n
    ns=dict(original.controller.__globals__,CONFIG=CONFIG,MODULE=MODULE,verify=verify,source_binding=source_binding)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ScopeNames().visit(node)],type_ignores=[])),str(original.__file__),'exec'),ns)
    return ns['controller']


def worker_command(case):
    return [str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',MODULE,'--job-id',case['id']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true');p.add_argument('--job-id');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=577,label_views=1726)));return
    c=read(ROOT/CONFIG);cases=verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=577,label_views=1726,
        numpy_imported='numpy' in __import__('sys').modules,torch_imported='torch' in __import__('sys').modules)))
    elif a.job_id:worker(c,next(case for case in cases if case['id']==a.job_id))
    else:bound_controller()(c,cases)


if __name__=='__main__':main()
