"""All5 full-SMAP common105 checkpoints: scalar audit, native reload, review.

Reuse the pinned full873 metric and full105 baseline numerical bodies. Source
completion and actual common105 data are independently bound before each step.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace

from scripts.flow_matching import run_full_smap_DCD105_min105_controls_v1 as producer
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as metric
from scripts.flow_matching import replay_original_full105_baselines_v1 as baseline

ROOT,read,sha,write=producer.ROOT,producer.read,producer.sha,producer.write
MODULE='scripts.flow_matching.replay_full5_DCD105_min105_controls_v1'
CONFIG='configs/reproducibility/fm_full5_DCD105_min105_native_and20_labels.2026-10-16_v1.json'
TEST='tests/test_fm_full5_DCD105_min105_native_and_labels_v1.py'
REPORTS=dict(scalar=metric.REPORT,**baseline.REPORTS)


def registration():
    source=read(ROOT/producer.CONFIG);producer.verify(source)
    material_path=Path(source['artifact_root'])/'materialization.json';material=read(material_path)
    data=producer.binding(source)
    if (not material['passed'] or material['configuration_sha256']!=sha(ROOT/producer.CONFIG)
            or material['full_data_binding']!=data or sha(material['queue'])!=material['queue_sha256']):
        raise ValueError('Actual frozen5source queues and common105 data required')
    queue=read(material['queue']);producer.validate_clone(source,queue,data)
    cases=[];models={};policies=list(data['records']['smap']['labels_only_controls'])
    native_policy=read(ROOT/baseline.POLICY);scalar_policy=read(ROOT/metric.POLICY)
    for job in queue['jobs']:
        cases.append(dict(id=job['id'],family='baselines',dataset=job['dataset'],seed=job['seed'],window=job['window'],
            queue_path=material['queue'],queue_sha256=material['queue_sha256'],worker_python=queue['python'],
            required_label_policies=policies,
            source_guard=str(ROOT/source['state_root']/'receipts'/(job['id']+'.json'))))
        models[job['model_config']]=read(ROOT/job['model_config'])
    if len(cases)!=5 or len(policies)!=4 or {c['seed'] for c in cases}!=set(range(1103,1108)):
        raise ValueError('All5source seeds and all20SMAP label views required')
    pins={producer.CONFIG,producer.MODULE.replace('.','/')+'.py',str(material_path),material['queue'],
        MODULE.replace('.','/')+'.py',TEST,baseline.MODULE.replace('.','/')+'.py',
        metric.MODULE.replace('.','/')+'.py',baseline.POLICY,metric.POLICY,baseline.MATH,
        'scripts/flow_matching/legacy_baseline_checkpoint_math_v1.py',
        'scripts/flow_matching/reflow_checkpoint_evidence_v1.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/audit_timesnet_NASA_label_protocol_v1.py',
        'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
        'src/iia_benchmark/models/fm_legacy_window_adapter.py','src/iia_benchmark/models/mtsad_detectors.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py'}
    pins.update(r['path'] for r in source['source_receipts']);pins.update(models)
    return dict(schema_version=1,source_execution=producer.CONFIG,actual_materialization=str(material_path),cases=cases,
        required_cases=5,required_family_native_cases=5,required_family_label_views=20,
        source_model_configurations=models,expected_source_model_entrypoint='iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter',
        settings=native_policy['settings'],resource_lock=native_policy['resource_lock'],
        scalar_settings=scalar_policy['settings'],scalar_resource_lock=scalar_policy['resource_lock'],
        controller_python=sys.executable,math_tolerance=read(ROOT/baseline.MATH)['math_tolerance'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        paper_and_data_citations=dict(models={path:m.get('citations',m.get('citation')) for path,m in models.items()},
            data=read(ROOT/producer.prep.CONFIG)['paper_citations']),
        original_scope=dict(papers=45,method_baseline_ablation_records=681,data_tasks=273,paper_scalar_targets=702,
            original_full_raw_cases=873,original_label_views=2037),
        artifact_root='F:/aicoding/IIA_Data/experiments/full5_DCD105_min105_native_and20_labels_v1',
        state_root='experiments/runs/fm_full5_DCD105_min105_native_and20_labels_v1',
        boundary='All5complete54entitySMAP common105 source controls;all4label policies/20views. '
            'Exact producer guard and complete data proof before pure scalar audit. Restore every trained module/buffer; '
            'no fitting;native all-array dtype/shape/content bytes and serialized reload;fresh-process all-label metrics/events/bootstrap200/grid. '
            'Unchanged full105 baseline numerical bodies and original2CPUthreads,requires_grad flags, '
            'FP32per-head score program and independentFP64primitives at2e-6. IdealFP64end-to-end mismatches remain diagnostics. '
            'Scalar16/28/16GiB with3/6caps/auditlock;native/review16/32/16GiB with10/12caps/strictlock. '
            'Original localDCD3epochs/256width/3layers/105window/[3,5,7]patches/batch256 retained;not paper training certification. '
            'No retry/old result rebinding/V2cross-protocol pooling/source modification. Other-method matched105 controls, '
            'paired5seed statistics,fullTAB/historicalauthor equivalence and entire original research scope remain separate.')


def verify(c):
    if c!=registration():raise ValueError('All5actual common105 sources/all20labels/original math and memory contracts required')
    return c['cases']


def paths(c,case,phase):
    if phase not in REPORTS:raise ValueError('Explicit scalar/native/review phase required')
    output=Path(c['artifact_root'])/('scalar/'+case['id'] if phase=='scalar' else case['id']+'/'+phase)
    return ROOT/c['state_root']/'attempts'/case['id']/phase,output


def worker_command(case,phase):
    if phase not in REPORTS:raise ValueError('Explicit complete phase required')
    return [str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',MODULE,'--job-id',case['id'],'--stage',phase]


def source_binding(c,case):
    # Original complete guard/hash/array binding, with only the explicit new
    # producer's audited data-ready function in its private global namespace.
    body=FunctionType(metric.source_binding.__code__,dict(metric.source_binding.__globals__,ROOT=ROOT,
        source=SimpleNamespace(ready_data=producer.binding)))
    return body(c,case)


def scalar_worker(c,case):
    body=FunctionType(metric.worker.__code__,dict(metric.worker.__globals__,ROOT=ROOT,CONFIG=CONFIG,
        source_binding=source_binding))
    return body(dict(c,artifact_root=c['artifact_root']+'/scalar'),case)


def ready_scalar(c,case):
    attempt,output=paths(c,case,'scalar');path=output/REPORTS['scalar'];guard=attempt/'guard_receipt.json'
    if not path.exists() and not guard.exists():return None
    if not path.exists() or not guard.exists():raise ValueError('Complete scalar proof and successful actual guard required')
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(case,'scalar')
            or not proof['source_metrics_exactly_recomputed'] or not proof['all_original_arrays_checked']
            or set(proof['label_view_results'])!=set(case['required_label_policies'])
            or not proof['all_validation_thresholds_and_predictions_unchanged']):
        raise ValueError('Actual complete all4label metrics and scalar guard required')
    if source_binding(c,case)[-1]!=proof['source_binding']:raise ValueError('Complete common105 scalar source changed')
    return dict(path=str(path),sha256=sha(path),guard_sha256=sha(guard),source_binding=proof['source_binding'])


def native_body(name):
    scalar_api=SimpleNamespace(**dict(vars(metric),source_binding=source_binding))
    namespace=dict(baseline.native_worker.__globals__,ROOT=ROOT,CONFIG=CONFIG,scalar=scalar_api,
        paths=paths,ready_scalar=ready_scalar,ready_native=ready_native,worker_command=worker_command)
    return FunctionType(getattr(baseline,name).__code__,namespace)


def native_worker(c,case):return native_body('native_worker')(c,case)
def review_worker(c,case):return native_body('review_worker')(c,case)
def ready_native(c,case):return native_body('ready_native')(c,case)


def controller(c,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing complete5DCD replay owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve previous replay attempts')
    done=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_cases=5,required_label_views=20,active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**extra):
        state.update(value or {});state.update(extra,completed_cases=list(done),failed_replays_preserved=dict(failed),
            source_failures_preserved=dict(source_failed),captured_utc=datetime.now(timezone.utc).isoformat())
        atomic(base/'status.json',state)
    def stage(case,phase):
        attempt,output=paths(c,case,phase)
        if attempt.exists() or output.exists():raise FileExistsError('Preserve all scalar/native/review attempts;no retry')
        attempt.mkdir(parents=True,exist_ok=False);command=worker_command(case,phase)
        policy=dict(c,settings=c['scalar_settings']) if phase=='scalar' else c
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_complete_DCD105_'+phase,active_pid=child.pid,active_job=case['id'])
            code,usage=capped_wait(child,policy,api,update)
        report=output/REPORTS[phase];ok=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(report) if report.exists() else None))
        if not ok:raise ValueError('Complete DCD105 phase failed;preserve without retry')
    try:
        while len(done)+len(failed)+len(source_failed)<len(cases):
            for case in cases:
                identity=case['id']
                if identity in done or identity in failed or identity in source_failed:continue
                try:
                    status=metric.source_status(case)
                    if status is None:continue
                    if status['kind']=='source_failed_preserved':source_failed[identity]=status;continue
                    with api['resource_slot'](ROOT/c['scalar_resource_lock'],c['scalar_settings'],update):
                        verify(read(ROOT/CONFIG));source_binding(c,case);stage(case,'scalar');ready_scalar(c,case)
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
                        verify(read(ROOT/CONFIG));stage(case,'native');ready_native(c,case);stage(case,'review')
                    done.append(identity);update(state='between_complete_DCD105_controls',active_pid=None,active_job=None)
                except Exception as error:
                    failed[identity]=str(error);update(state='failure_preserved_without_retry',active_pid=None,active_job=None)
            update(state='waiting_actual_original_DCD105_training_and_guards',active_pid=None,active_job=None);time.sleep(10)
        update(state='all5_control_outcomes_terminal_other_original_experiments_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--job-id');p.add_argument('--stage',choices=list(REPORTS));a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=5,label_views=20)));return
    c=read(ROOT/CONFIG);cases=verify(c)
    if a.probe_only:
        states=[metric.source_status(case) for case in cases]
        print(json.dumps(dict(passed=True,cases=5,label_views=20,source_complete=sum(s is not None and s['kind']=='source_complete' for s in states),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif a.job_id:
        if a.stage is None:raise ValueError('Explicit complete replay phase required')
        case=next(x for x in cases if x['id']==a.job_id)
        dict(scalar=scalar_worker,native=native_worker,review=review_worker)[a.stage](c,case)
    else:controller(c,cases)


if __name__=='__main__':main()
