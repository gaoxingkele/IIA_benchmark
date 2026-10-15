"""Original TAB21 metric suite for all577 native-closed matched105 cases."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as original
from scripts.flow_matching import recover_TAB_resource_cells_v2 as durable
from scripts.flow_matching import replay_common105_matched577_v1 as native

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.run_common105_matched577_TAB_v1'
CONFIG='configs/reproducibility/fm_common105_matched577_TAB21_durable_metrics.2026-10-16_v1.json'
TEST='tests/test_fm_common105_matched577_TAB21_v1.py'
REPORT=original.REPORT
COMPONENT='original_entity_metric_component.json'


def registration():
    previous=read(ROOT/original.CONFIG);root=read(ROOT/native.CONFIG);native.verify(root)
    entries=[];components={};pins={original.CONFIG,durable.CONFIG,native.CONFIG,
        MODULE.replace('.','/')+'.py',TEST,native.MODULE.replace('.','/')+'.py',durable.MODULE.replace('.','/')+'.py'}
    pins.update(p['path'] for p in previous['source_receipts'])
    for item in root['families']:
        family=item['family'];c=read(ROOT/item['configuration']);pins.add(item['configuration'])
        pins.update(p['path'] for p in c['source_receipts'])
        components[family]=dict(module=native.MODULE,configuration=item['configuration'],configuration_sha256=item['sha256'],
            required_native_cases=item['cases'])
        for case in c['cases']:
            entries.append(dict(key=original.key(case),case=case,component=family,track='common105_matched577'))
    if len(entries)!=577 or len({e['key'] for e in entries})!=577 or sum(len(e['case']['required_label_policies']) for e in entries)!=1726:
        raise ValueError('All577matched native cases and1726label views required')
    return dict(previous,entries=entries,native_components=components,required_cases=577,required_label_views=1726,
        required_entity_label_views=77490,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_matched577_TAB21_durable_v1',
        state_root='experiments/runs/fm_common105_matched577_TAB21_durable_v1',
        source_native_configuration=native.CONFIG,source_scalar_configuration=native.scalar.CONFIG,
        original_matched_comparison=root['original_matched_comparison'],
        boundary='All577new matched cases/1726labels/77490entity-label views after actual source/native/fresh-review closure. '
            'Original21TABfunctions/all201calls per entity-label/all13original threshold levels/fullVUS windows and250thresholds. '
            'Unchanged original worker/controller/aggregate numerical bodies with new native/source bindings. '
            'Atomic per-metric exact argument/function/source-bound journal;not a changed metric or new independent repetition. '
            'Original16/28/16GiB3/6caps/auditlock/CPU2 retained;any resource abort is preserved without automatic retry. '
            'Per-entity events remain separate;undefined/errors preserved;complete macro and200entityCI require allfinite. '
            '291SMDreferences and5DCDcontrols remain in their own original/compatible queues to cover873/2037,never rebindingV2SMAP/MSL. '
            'All original source historical-fidelity limits,paired5trainingseed/fullTABtrainingharness obligations and entire45/681/273/702scope remain.')


def verify(c):
    if c!=registration():raise ValueError('All577matched21metric suites and unchanged full original numerical/resource definitions required')
    return c['entries']


def component(entry):
    family=entry['component']
    if family not in native.SPECS:raise ValueError('Exact matched native family required')
    module=native.component(family);ns=native.namespace(family)
    return SimpleNamespace(CONFIG=native.CONFIGS[family],REPORTS=module.REPORTS,paths=native.paths,
        ready_native=ns['ready_native'],worker_command=lambda case,phase:native.worker_command(family,case,phase))


def ready_case(c,entry):
    if entry['track']!='common105_matched577':raise ValueError('Explicit matched105source track required')
    api=SimpleNamespace(source_binding=native.scalar.source_binding)
    return FunctionType(original.ready_case.__code__,dict(original.ready_case.__globals__,component=component,scalar=api))(c,entry)


def worker_command(c,entry,prefix):
    return [c['metric_python'],'-B','-X','utf8','-u','-m',MODULE,'--case-key',entry['key'],'--entity-prefix',prefix]


def source_status(entry):return native.scalar.original.source_status(entry['case'])


def worker(c,entry,prefix):
    bound=ready_case(c,entry)
    if bound is None:raise ValueError('Actual matched native/review closure required')
    _,output=original.paths(c,entry,prefix);journal={}
    def evaluate(config,scores,views,result,metrics,progress):
        rows,closed=durable.evaluate_checkpointed(config,entry,prefix,scores,views,result,metrics,
            bound['lineage'],output.parent/(prefix+'_cell_bank'),progress)
        journal.update(closed);return rows
    body=FunctionType(original.worker.__code__,dict(original.worker.__globals__,ROOT=ROOT,CONFIG=CONFIG,REPORT=COMPONENT,
        ready_case=ready_case,evaluate_entity=evaluate))
    body(c,entry,prefix)
    p=read(output/COMPONENT)
    write(output/REPORT,dict(p,original_component_sha256=sha(output/COMPONENT),durable_cell_journal=journal,
        all_original_metric_calls_freshly_executed_this_attempt=journal['checkpoint_cells']==0,
        original_metric_definition_or_grid_changed=False,checkpoint_cells_are_new_independent_scientific_repetitions=False))


def checked_fragment(c,entry,prefix):
    # The full durable-fragment checker also verifies every referenced saved
    # cell equals the metric result and its hash. Original-V2 fallback has no
    # matching new case key;actual configuration/command guards remain strict.
    body=FunctionType(durable.checked_fragment.__code__,dict(durable.checked_fragment.__globals__,ROOT=ROOT,CONFIG=CONFIG,
        REPORT=REPORT,paths=original.paths,worker_command=worker_command))
    path,proof,kind=body(c,entry,prefix)
    if kind!='new_checkpointed_guarded_fragment':raise ValueError('New matched metric proof required;no originalV2 data rebinding')
    return path,proof


def namespace():
    ns=dict(original.controller.__globals__,CONFIG=CONFIG,MODULE=MODULE,verify=verify,source_status=source_status,
        ready_case=ready_case,worker=worker,checked_fragment=checked_fragment,worker_command=worker_command)
    ns['summary_worker']=FunctionType(original.summary_worker.__code__,ns)
    node=next(n for n in ast.parse(Path(original.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class ScopeNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,int) and not isinstance(n.value,bool) and n.value in [878,2057]:
                return ast.copy_location(ast.Constant({878:577,2057:1726}[n.value]),n)
            if isinstance(n.value,str):return ast.copy_location(ast.Constant(n.value.replace('878','577').replace('2057','1726')),n)
            return n
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ScopeNames().visit(node)],type_ignores=[])),str(original.__file__),'exec'),ns)
    return ns


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--case-key');p.add_argument('--entity-prefix');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=577,label_views=1726)));return
    c=read(ROOT/CONFIG);entries=verify(c)
    if a.probe_only:print(json.dumps(dict(passed=True,cases=577,label_views=1726,entity_label_views=77490,
        numpy_imported='numpy' in __import__('sys').modules,torch_imported='torch' in __import__('sys').modules)))
    elif a.case_key:
        entry=next(e for e in entries if e['key']==a.case_key)
        if a.entity_prefix=='summary':namespace()['summary_worker'](c,entry)
        elif a.entity_prefix:worker(c,entry,a.entity_prefix)
        else:raise ValueError('Explicit complete original entity required')
    else:namespace()['controller'](c,entries)


if __name__=='__main__':main()
