"""Checkpoint exact TAB metric cells for proven original resource-abort gaps.

The original878 controller keeps running. Successful old fragments are retained;
one explicit recovery attempt is allowed per missing entity in this version.
No scientific source training or metric definition is changed.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType
import uuid
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as original

ROOT,read,sha,write=original.ROOT,original.read,original.sha,original.write
MODULE='scripts.flow_matching.recover_TAB_resource_cells_v2'
CONFIG='configs/reproducibility/fm_TAB878_resource_recovery_durable_cells.2026-10-16_v2.json'
TEST='tests/test_fm_TAB_resource_recovery_durable_cells_v2.py'
REPORT='entity_checkpointed_TAB_metrics.json'
COMPONENT='original_entity_metric_component.json'
SUMMARY='complete_case_checkpointed_TAB_metrics.json'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def registration():
    old=read(ROOT/original.CONFIG);settings=deepcopy(old['settings'])
    settings['minimum_commit_headroom_bytes']=32*2**30
    pins={r['path'] for r in old['source_receipts']}
    pins.update([original.CONFIG,MODULE.replace('.','/')+'.py',TEST])
    return dict(old,original_metric_configuration=original.CONFIG,settings=settings,
        original_resource_settings=old['settings'],stable_admission_samples=6,stable_poll_seconds=5,
        maximum_new_attempts_per_entity=1,
        artifact_root='F:/aicoding/IIA_Data/experiments/TAB878_resource_durable_recovery_v2',
        state_root='experiments/runs/fm_TAB878_resource_durable_recovery_v2',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        boundary=old['boundary']+' Resource-abort recovery only,not model/numerical failure retry. '
            'Actual failed original guard plus dead exact child and unchanged native lineage required. '
            'Strengthen admission28to32GiB with6stable5second snapshots;keep original physical/emergency/RSS/private/auditlock. '
            'All original6score+15label functions and201cells per entity-label unchanged. '
            'Reuse guarded original entity fragments;persist individual successful/error/nonfinite cells atomically '
            'with exact argument/function/source hashes for resumability. Partial cells are checkpoints, '
            'not independently guarded benchmark results until a complete new entity exits successfully. '
            'One new attempt per missing entity;any new abort/failure preserved,never automatically repeated. '
            'Original878/2057and complete45/681/273/702scope retained.')


def verify(c):
    if c!=registration():raise ValueError('Full878scope/exact original metric programs and strengthened original resource policy required')
    return c['entries']


def resource_abort(c,entry):
    """Observe a terminal old guard,not a lock file or merely stale status."""
    import psutil
    old=read(ROOT/original.CONFIG);base=ROOT/old['state_root']
    status=read(base/'status.json')
    if entry['key'] not in status['failed_cases_preserved']:return None
    attempts=base/'attempts'/entry['key'];evidence=[]
    for path in sorted(attempts.glob('*/guard_receipt.json')):
        g=read(path)
        if not g.get('resource_guard_aborted'):continue
        prefix=path.parent.name;expected=original.worker_command(old,entry,prefix)
        if (g['passed'] or g['exit_code']==0 or g['exact_child_handle']['command']!=expected
                or not (g['abort_snapshot']['commit_available_bytes']<old['settings']['emergency_commit_headroom_bytes']
                    or g['peak_tree_rss_bytes']>old['settings']['maximum_worker_rss_bytes']
                    or g['peak_tree_private_bytes']>old['settings']['maximum_worker_private_bytes'])):
            raise ValueError('Actual original resource-triggered failed guard required')
        h=g['exact_child_handle']
        try:
            p=psutil.Process(h['pid'])
            if p.create_time()==h['create_time']:
                if p.cmdline()!=h['command']:raise ValueError('Original worker identity changed unexpectedly')
                return None
        except psutil.NoSuchProcess:pass
        evidence.append(dict(prefix=prefix,guard_path=str(path),guard_sha256=sha(path),exact_terminal_handle=h,
            abort_snapshot=g['abort_snapshot']))
    if not evidence:return None
    bound=original.ready_case(c,entry)
    if bound is None:return None
    return dict(original_failed_guards=evidence,source_lineage=bound['lineage'],
        failed_guard_is_algorithm_accuracy_failure=False,original_controller_entry_terminal=True)


def paths(c,entry,prefix):
    if prefix!='summary' and (not prefix.startswith('e') or not prefix[1:].isdigit()):raise ValueError('Original complete entity prefix required')
    return (ROOT/c['state_root']/'attempts'/entry['key']/prefix,
        Path(c['artifact_root'])/'cases'/entry['key']/prefix)


def atomic_new(path,value):
    """Atomic durable single-file checkpoint;never overwrite an existing cell."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Existing metric checkpoint preserved')
    # Keep temporary names independent of the64-character cell key so deeply
    # nested Windows workspaces do not exceed MAX_PATH during the atomic save.
    temporary=path.parent/('.pending-'+uuid.uuid4().hex)
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if path.exists():raise FileExistsError('Concurrent checkpoint writer refused')
    temporary.rename(path)


def evaluate_checkpointed(c,entry,prefix,scores,views,result,metrics,source_lineage,bank,progress=lambda value:None):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    bank=Path(bank);bank.mkdir(parents=True,exist_ok=True)
    counts=dict(fresh_cells=0,checkpoint_cells=0);records=[];current={}
    contract=dict(original_configuration_sha256=sha(ROOT/original.CONFIG),case=entry['case'],
        entity_prefix=prefix,source_lineage=source_lineage)
    def observed(value):current.update(value);progress(value)
    def evaluate(function,actual,predicted):
        identity=dict(contract,policy=current['policy'],metric=current['metric'],
            actual_argument=array_digest(np.asarray(actual,dtype=np.float64)),
            predicted_argument=array_digest(np.asarray(predicted,dtype=np.float64)),
            source_function=dict(module=function.__module__,name=function.__name__,qualified_name=function.__qualname__,
                first_line=function.__code__.co_firstlineno,file=str(Path(function.__code__.co_filename).resolve()),
                file_sha256=sha(function.__code__.co_filename)))
        token=digest(identity);path=bank/(digest(dict(policy=current['policy'],metric=current['metric']))+'.json')
        if path.exists():
            stored=read(path)
            if stored['identity']!=identity or stored['identity_sha256']!=token or stored['outcome_sha256']!=digest(stored['outcome']):
                raise ValueError('Metric checkpoint source,argument,function or outcome changed')
            outcome=stored['outcome'];counts['checkpoint_cells']+=1
        else:
            outcome=original.evaluate_cell(function,actual,predicted)
            atomic_new(path,dict(identity=identity,identity_sha256=token,outcome=outcome,outcome_sha256=digest(outcome),
                producing_pid=os.getpid(),checkpoint_is_independent_guarded_result=False))
            counts['fresh_cells']+=1
        records.append(dict(policy=current['policy'],metric=current['metric'],path=str(path),sha256=sha(path)))
        return outcome
    body=FunctionType(original.evaluate_entity.__code__,dict(original.evaluate_entity.__globals__,evaluate_cell=evaluate))
    rows=body(c,scores,views,result,metrics,observed)
    original.validate_cells(c,rows,views)
    if len(records)!=len(views)*201 or counts['fresh_cells']+counts['checkpoint_cells']!=len(records):
        raise ValueError('Every original cell must be computed or exactly checkpoint-bound')
    return rows,dict(**counts,cell_records=records,numerical_contract=contract,
        complete_original_evaluate_entity_code_reused=True,partial_checkpoints_are_independent_repetitions=False)


def worker(c,entry,prefix):
    evidence=resource_abort(c,entry)
    if evidence is None:raise ValueError('Actual terminal resource-abort gap required before new recovery')
    bound=original.ready_case(c,entry);attempt,output=paths(c,entry,prefix)
    journal={}
    def evaluate(config,scores,views,result,metrics,progress):
        rows,closed=evaluate_checkpointed(config,entry,prefix,scores,views,result,metrics,
            bound['lineage'],output.parent/(prefix+'_cell_bank'),progress)
        journal.update(closed);return rows
    body=FunctionType(original.worker.__code__,dict(original.worker.__globals__,CONFIG=CONFIG,REPORT=COMPONENT,
        paths=paths,evaluate_entity=evaluate))
    body(c,entry,prefix)
    proof=read(output/COMPONENT)
    if proof['source_lineage']!=evidence['source_lineage']:raise ValueError('Every old and recovered metric must use the same exact source')
    write(output/REPORT,dict(proof,original_resource_abort=evidence,
        original_component_sha256=sha(output/COMPONENT),durable_cell_journal=journal,
        all_original_metric_calls_freshly_executed_this_attempt=journal['checkpoint_cells']==0,
        checkpoint_cells_are_new_independent_scientific_repetitions=False,
        original_metric_definition_or_grid_changed=False,recovery_attempts_in_this_version=1,
        all_original_experiments_complete=False))


def worker_command(c,entry,prefix):
    return [c['metric_python'],'-B','-X','utf8','-u','-m',MODULE,'--case-key',entry['key'],'--entity-prefix',prefix]


def checked_fragment(c,entry,prefix):
    old=read(ROOT/original.CONFIG);a,o=original.paths(old,entry,prefix)
    if (o/original.REPORT).exists() and (a/'guard_receipt.json').exists() and read(a/'guard_receipt.json')['passed']:
        path,p=original.checked_fragment(old,entry,prefix)
        return path,p,'original_guarded_fragment'
    attempt,output=paths(c,entry,prefix);path=output/REPORT;guard=attempt/'guard_receipt.json'
    p,g=read(path),read(guard);original.validate_cells(c,p['metric_results'],entry['case']['required_label_policies'])
    if (not p['passed'] or p['case']!=entry['case'] or p['array_prefix']!=prefix or p['configuration_sha256']!=sha(ROOT/CONFIG)
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(c,entry,prefix)
            or p['original_metric_definition_or_grid_changed']):
        raise ValueError('Complete exact recovered entity and successful actual guard required')
    records=p['durable_cell_journal']['cell_records']
    expected={(policy,name) for policy,cells in p['metric_results'].items() for name in cells}
    if len(records)!=len(expected) or {(r['policy'],r['metric']) for r in records}!=expected:
        raise ValueError('All exact original metric cells need durable evidence')
    for cell in records:
        if sha(cell['path'])!=cell['sha256']:raise ValueError('A completed durable cell changed')
        record=read(cell['path'])
        if record['outcome']!=p['metric_results'][cell['policy']][cell['metric']]:raise ValueError('Exact saved metric outcome required')
    return path,p,'new_checkpointed_guarded_fragment'


def summary_worker(c,entry):
    evidence=resource_abort(c,entry);bound=original.ready_case(c,entry)
    if evidence is None or bound is None:raise ValueError('Complete actual unchanged source and preserved resource abort required')
    fragments=[];pins=[]
    for entity in bound['dataset']['entities']:
        path,proof,kind=checked_fragment(c,entry,entity['array_prefix'])
        if proof['entity']!=entity['entity_id'] or proof['source_lineage']!=bound['lineage']:
            raise ValueError('Full original grouped dataset at the same source thresholds required')
        fragments.append(proof);pins.append(dict(path=str(path),sha256=sha(path),kind=kind))
    _,output=paths(c,entry,'summary');output.mkdir(parents=True,exist_ok=False)
    write(output/SUMMARY,dict(passed=True,case=entry['case'],configuration_sha256=sha(ROOT/CONFIG),source_lineage=bound['lineage'],
        original_resource_abort=evidence,entities=len(fragments),fragments=pins,
        metric_calls=sum(p['metric_calls'] for p in fragments),macro_metrics=original.aggregate(c,fragments,entry['case']['seed']),
        original_metric_definition_or_grid_changed=False,new_training_or_model_inference=False,
        all_original_experiments_complete=False))


def stable_headroom(c,api,update):
    count=0
    while count<c['stable_admission_samples']:
        snapshot=api['resource_snapshot']()
        enough=(snapshot['physical_available_bytes']>=c['settings']['minimum_available_memory_bytes']
            and snapshot['commit_available_bytes']>=c['settings']['minimum_commit_headroom_bytes'])
        count=count+1 if enough else 0
        update(dict(state='waiting_stable_strengthened_memory',stable_samples=count,**snapshot))
        if count<c['stable_admission_samples']:time.sleep(c['stable_poll_seconds'])


def controller(c,entries):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing durable TAB recovery owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every previous recovery attempt')
    done=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        original_case_scope=878,completed_recovered_entities={},reused_original_entities={},all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_recovery_case_keys=list(done),failed_recoveries_preserved=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    def stage(entry,prefix):
        attempt,output=paths(c,entry,prefix)
        if attempt.exists() or output.exists():raise FileExistsError('One explicit new recovery attempt only;preserve any failure')
        stable_headroom(c,api,update)
        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
            verify(read(ROOT/CONFIG));attempt.mkdir(parents=True,exist_ok=False);command=worker_command(c,entry,prefix)
            with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                h=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                write(attempt/'launch.json',h);update(state='running_checkpointed_original_TAB_metrics',active_pid=child.pid,
                    active_case_key=entry['key'],active_entity_prefix=prefix)
                code,usage=capped_wait(child,c,api,update)
            report=output/(SUMMARY if prefix=='summary' else REPORT)
            ok=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
            write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=h,proof_sha256=sha(report) if report.exists() else None))
            if not ok:raise ValueError('New recovery attempt failed;preserve checkpoints without automatic retry')
    try:
        while True:
            for entry in entries:
                k=entry['key']
                if k in done or k in failed:continue
                try:
                    evidence=resource_abort(c,entry)
                    if evidence is None:continue
                    bound=original.ready_case(c,entry);old=read(ROOT/original.CONFIG)
                    for entity in bound['dataset']['entities']:
                        prefix=entity['array_prefix'];a,o=original.paths(old,entry,prefix)
                        if (o/original.REPORT).exists() and (a/'guard_receipt.json').exists() and read(a/'guard_receipt.json')['passed']:
                            checked_fragment(c,entry,prefix);state['reused_original_entities'][k]=state['reused_original_entities'].get(k,0)+1
                            continue
                        stage(entry,prefix);checked_fragment(c,entry,prefix)
                        state['completed_recovered_entities'][k]=state['completed_recovered_entities'].get(k,0)+1
                        update(state='between_checkpointed_entities',active_pid=None)
                    stage(entry,'summary');done.append(k);update(state='complete_resource_gap_case_closed',active_pid=None)
                except Exception:
                    failed[k]=__import__('traceback').format_exc();update(state='recovery_failure_preserved',active_pid=None)
            update(state='waiting_actual_original_resource_abort_gaps',active_pid=None);time.sleep(10)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--case-key');p.add_argument('--entity-prefix');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,original_cases=878)));return
    c=read(ROOT/CONFIG);entries=verify(c)
    if a.probe_only:
        eligible=[e['key'] for e in entries if resource_abort(c,e) is not None]
        print(json.dumps(dict(passed=True,original_case_scope=878,actual_resource_gap_case_keys=eligible,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif a.case_key:
        entry=next(e for e in entries if e['key']==a.case_key)
        if a.entity_prefix=='summary':summary_worker(c,entry)
        elif a.entity_prefix:worker(c,entry,a.entity_prefix)
        else:raise ValueError('Explicit full original entity required')
    else:controller(c,entries)


if __name__=='__main__':main()
