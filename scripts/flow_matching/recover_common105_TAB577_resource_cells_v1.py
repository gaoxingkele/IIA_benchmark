"""One explicit cell-preserving recovery per proven matched577 resource gap.

Keep the live fair controller, every577case/1726label view and all201original
metric calls. No model retraining or metric/source change; no automatic retry.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import FunctionType,SimpleNamespace
from scripts.flow_matching import recover_TAB_resource_cells_v2 as durable
from scripts.flow_matching import run_common105_matched577_TAB_v1 as matched
from scripts.flow_matching import continue_common105_TAB577_fair_entities_v1 as fair

ROOT,read,sha,write=durable.ROOT,durable.read,durable.sha,durable.write
MODULE='scripts.flow_matching.recover_common105_TAB577_resource_cells_v1'
CONFIG='configs/reproducibility/fm_common105_TAB577_resource_cell_recovery.2026-10-16_v1.json'
TEST='tests/test_fm_common105_TAB577_resource_cell_recovery_v1.py'
REPORT,COMPONENT,SUMMARY=durable.REPORT,durable.COMPONENT,durable.SUMMARY
paths=durable.paths


def registration():
    old=read(ROOT/matched.CONFIG);matched.verify(old)
    reference=read(ROOT/durable.CONFIG);durable.verify(reference)
    settings=deepcopy(old['settings']);settings['minimum_commit_headroom_bytes']=32*2**30
    pins={r['path'] for r in old['source_receipts']}
    pins.update([matched.CONFIG,fair.CONFIG,durable.CONFIG,durable.MODULE.replace('.','/')+'.py',
        MODULE.replace('.','/')+'.py',TEST,'scripts/flow_matching/cached_resource_snapshot_v1.py'])
    return dict(old,original_metric_configuration=matched.CONFIG,original_resource_settings=old['settings'],settings=settings,
        stable_admission_samples=6,stable_poll_seconds=5,maximum_new_attempts_per_entity=1,
        original_failed_registry=fair.BASE+'/status.json',inter_entity_yield_seconds=20,
        original_partial_cell_contract_configuration=durable.original.CONFIG,
        artifact_root='F:/aicoding/IIA_Data/experiments/common105_TAB577_resource_cell_recovery_v1',
        state_root='experiments/runs/fm_common105_TAB577_resource_cell_recovery_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        boundary='All577matched cases/1726label views/77490entity-label views retained;live fair controller stays. '
            'Require actual failed original resource guard,exact command,dead exact child and unchanged source/native/labels. '
            'One explicit recovery per missing entity;reuse guarded original entities. Preserve original failed banks; '
            'byte-copy complete saved cells to a new bank,verify original arguments/function/lineage/outcome hashes on reuse, '
            'and execute every remaining original201metric cell per label unchanged. '
            'Only complete new guarded entities count;partial cells are checkpoints,not independent runs. '
            '32GiB admission6stable5second snapshots;original16GiBphysical/16GiBemergency/3GiBRSS/6GiBprivate/CPU2/lock retained. '
            'Yield20seconds between recovered entities. Further resource or numerical failures preserved without retry. '
            'Full original873/878 queues and45papers/681methods/273data/702targets remain required.',
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('All577matched cases and unchanged metric/cell/source/resource contract required')
    return c['entries']


def resource_abort(c,entry):
    import psutil
    status=read(ROOT/c['original_failed_registry'])
    if entry['key'] not in status['failed_cases_preserved']:return None
    old=read(ROOT/matched.CONFIG);base=ROOT/old['state_root']/'attempts'/entry['key'];evidence=[]
    for path in sorted(base.glob('*/guard_receipt.json')):
        guard=read(path)
        if not guard.get('resource_guard_aborted'):continue
        prefix=path.parent.name;expected=matched.worker_command(old,entry,prefix)
        if (guard['passed'] or guard['exit_code']==0 or guard['exact_child_handle']['command']!=expected
                or not (guard['abort_snapshot']['commit_available_bytes']<old['settings']['emergency_commit_headroom_bytes']
                    or guard['peak_tree_rss_bytes']>old['settings']['maximum_worker_rss_bytes']
                    or guard['peak_tree_private_bytes']>old['settings']['maximum_worker_private_bytes'])):
            raise ValueError('Exact actual original resource-aborted terminal and worker command required')
        h=guard['exact_child_handle']
        try:
            p=psutil.Process(h['pid'])
            if p.create_time()==h['create_time']:
                if p.cmdline()!=h['command']:raise ValueError('Old child identity unexpectedly changed')
                return None
        except psutil.NoSuchProcess:pass
        evidence.append(dict(prefix=prefix,guard_path=str(path),guard_sha256=sha(path),exact_terminal_handle=h,
            abort_snapshot=guard['abort_snapshot']))
    if not evidence:return None
    bound=matched.ready_case(c,entry)
    if bound is None:return None
    return dict(original_failed_guards=evidence,source_lineage=bound['lineage'],
        failed_guard_is_algorithm_accuracy_failure=False,original_controller_entry_terminal=True)


def seed_partial_bank(old,entry,prefix,lineage,target):
    bank=Path(old['artifact_root'])/'cases'/entry['key']/(prefix+'_cell_bank')
    target=Path(target);target.mkdir(parents=True,exist_ok=False);copied=[]
    for source in sorted(bank.glob('*.json')):
        item=read(source);identity=item['identity']
        if (identity['case']!=entry['case'] or identity['entity_prefix']!=prefix or identity['source_lineage']!=lineage
                or identity['original_configuration_sha256']!=sha(ROOT/durable.original.CONFIG)
                or item['identity_sha256']!=durable.digest(identity) or item['outcome_sha256']!=durable.digest(item['outcome'])
                or source.name!=durable.digest(dict(policy=identity['policy'],metric=identity['metric']))+'.json'):
            raise ValueError('Original partial cell must retain exact source/argument/function/outcome identity')
        dest=target/source.name
        with source.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
        if sha(source)!=sha(dest):raise ValueError('Original bank copy must remain byte-identical')
        copied.append(dict(original_path=str(source),original_sha256=sha(source),new_path=str(dest),new_sha256=sha(dest)))
    return copied


def worker(c,entry,prefix):
    evidence=resource_abort(c,entry)
    if evidence is None:raise ValueError('Exact terminal matched resource gap required')
    bound=matched.ready_case(c,entry);old=read(ROOT/matched.CONFIG);_,output=paths(c,entry,prefix)
    bank=output.parent/(prefix+'_cell_bank');copied=seed_partial_bank(old,entry,prefix,bound['lineage'],bank);journal={}
    def evaluate(config,scores,views,result,metrics,progress):
        rows,closed=durable.evaluate_checkpointed(config,entry,prefix,scores,views,result,metrics,bound['lineage'],bank,progress)
        journal.update(closed);return rows
    body=FunctionType(matched.original.worker.__code__,dict(matched.original.worker.__globals__,CONFIG=CONFIG,
        REPORT=COMPONENT,paths=paths,ready_case=matched.ready_case,evaluate_entity=evaluate))
    body(c,entry,prefix);proof=read(output/COMPONENT)
    if proof['source_lineage']!=evidence['source_lineage'] or journal['checkpoint_cells']!=len(copied):
        raise ValueError('Every seeded partial cell must actually match the complete original calculation')
    write(output/REPORT,dict(proof,original_resource_abort=evidence,original_component_sha256=sha(output/COMPONENT),
        durable_cell_journal=journal,seeded_partial_cell_bank=copied,old_partial_cell_bank_unchanged=True,
        all_original_metric_calls_freshly_executed_this_attempt=journal['checkpoint_cells']==0,
        checkpoint_cells_are_new_independent_scientific_repetitions=False,original_metric_definition_or_grid_changed=False,
        recovery_attempts_in_this_version=1,all_original_experiments_complete=False))


def worker_command(c,entry,prefix):
    return [c['metric_python'],'-B','-X','utf8','-u','-m',MODULE,'--case-key',entry['key'],'--entity-prefix',prefix]


def namespace():
    api=SimpleNamespace(CONFIG=matched.CONFIG,REPORT=matched.REPORT,paths=matched.original.paths,ready_case=matched.ready_case,
        checked_fragment=lambda c,e,p:matched.checked_fragment(c,e,p)[:2],aggregate=matched.original.aggregate)
    values=dict(durable.__dict__,CONFIG=CONFIG,MODULE=MODULE,original=api,verify=verify,resource_abort=resource_abort,
        paths=paths,worker=worker,worker_command=worker_command)
    for name in ['checked_fragment','summary_worker']:
        f=getattr(durable,name);values[name]=FunctionType(f.__code__,values,name,f.__defaults__,f.__closure__)
    tree=ast.parse(Path(durable.__file__).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    class MatchedScope(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value) is int and n.value==878:return ast.copy_location(ast.Constant(577),n)
            return n
        def visit_Expr(self,n):
            result=self.generic_visit(n)
            if (isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='update'
                    and any(k.arg=='state' and isinstance(k.value,ast.Constant) and k.value.value=='between_checkpointed_entities' for k in n.value.keywords)):
                return [result,ast.parse("time.sleep(c['inter_entity_yield_seconds'])").body[0]]
            return result
    exec(compile(ast.fix_missing_locations(ast.Module(body=[MatchedScope().visit(node)],type_ignores=[])),__file__,'exec'),values)
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--case-key');p.add_argument('--entity-prefix');a=p.parse_args()
    if a.register:durable.atomic_new(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);entries=verify(c);ns=namespace()
    if a.probe_only:
        eligible=[e['key'] for e in entries if resource_abort(c,e) is not None]
        print(json.dumps(dict(cases=577,label_views=1726,actual_resource_gaps=eligible,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if a.case_key:
        entry=next(e for e in entries if e['key']==a.case_key)
        if a.entity_prefix=='summary':ns['summary_worker'](c,entry)
        elif a.entity_prefix:worker(c,entry,a.entity_prefix)
        else:raise ValueError('Complete entity or summary required')
    else:ns['controller'](c,entries)


if __name__=='__main__':main()
