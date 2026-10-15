"""Original TAB21 metric controls for all873 fullraw plus5 common105 cases.

This evaluates unchanged scores after actual native/review closure. It does not
claim to reproduce TAB's training/data strategy or historical paper protocols.
"""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as scalar
from scripts.flow_matching import replay_full5_DCD105_min105_controls_v1 as extra

ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write
MODULE='scripts.flow_matching.run_TAB_full878_metric_suite_v1'
CONFIG='configs/reproducibility/fm_TAB_full878_original21_metric_suite.2026-10-16_v1.json'
TEST='tests/test_fm_TAB_full878_original_metric_suite_v1.py'
TAB='experiments/runs/mtsad_protocol_audit/sources/tab/original'
VENDOR='F:/aicoding/IIA_Data/environments/TAB_metrics_vendor_v1'
WHEELS='F:/aicoding/IIA_Data/environments/TAB_metrics_wheels_v1'
COMMIT='5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2'
SCORE=('auc_roc','auc_pr','R_AUC_ROC','R_AUC_PR','VUS_ROC','VUS_PR')
LABEL=('accuracy','f_score','precision','recall','adjust_accuracy','adjust_f_score','adjust_precision','adjust_recall',
    'rrecall','rprecision','precision_at_k','rf','affiliation_f','affiliation_precision','affiliation_recall')
COMPONENTS=('replay_original_full15_tranad_v1','replay_original_full165_fm_objectives_v1',
    'replay_original_full18_moment_v1','replay_original_full60_reflow_v1',
    'replay_original_full105_baselines_v1','replay_original_full510_neural_v1',extra.MODULE.rsplit('.',1)[1])
REPORT='entity_TAB_metric_suite.json'


def key(case):return hashlib.sha256(case['id'].encode('utf-8')).hexdigest()[:20]


def metric_environment():
    isolated={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions(path=[VENDOR])}
    if isolated!={'statsmodels':'0.14.6','patsy':'1.0.1'}:raise ValueError('Exact isolated TAB dependencies required')
    return dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','pandas']},**isolated)


def registration():
    original=read(ROOT/scalar.CONFIG);additional=read(ROOT/extra.CONFIG)
    expected={case['id']:case for case in original['cases']+additional['cases']}
    if len(expected)!=878:raise ValueError('All873original plus5common105 controls required')
    components={};entries=[]
    pins={scalar.CONFIG,extra.CONFIG,MODULE.replace('.','/')+'.py',TEST,scalar.POLICY,
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py'}
    for name in COMPONENTS:
        module=importlib.import_module('scripts.flow_matching.'+name);c=read(ROOT/module.CONFIG)
        components[name]=dict(module=module.MODULE,configuration=module.CONFIG,configuration_sha256=sha(ROOT/module.CONFIG),
            required_native_cases=c['required_family_native_cases'])
        pins.update([module.CONFIG,module.MODULE.replace('.','/')+'.py'])
        pins.update(p['path'] for p in c['source_receipts'])
        for case in c['cases']:
            if expected.get(case['id'])!=case:raise ValueError('Every actual native case must match its scalar source')
            entries.append(dict(key=key(case),case=case,component=name,
                track='common105_DCD5' if name==COMPONENTS[-1] else 'original_full873_v2'))
    if len(entries)!=878 or len({e['key'] for e in entries})!=878 or sum(len(e['case']['required_label_policies']) for e in entries)!=2057:
        raise ValueError('All878unique cases and2057label views required')
    sources=[(ROOT/TAB/'ts_benchmark/__init__.py'),(ROOT/TAB/'ts_benchmark/evaluation/__init__.py')]
    sources+=list((ROOT/TAB/'ts_benchmark/evaluation/metrics').rglob('*.py'))
    pins.update(p.relative_to(ROOT).as_posix() for p in sources)
    vendor_pins=[p for p in Path(VENDOR).rglob('*') if p.is_file() and p.suffix not in ['.pyc']]
    pins.update(str(p) for p in vendor_pins)
    pins.update(str(p) for p in Path(WHEELS).iterdir() if p.is_file())
    policy=read(ROOT/scalar.POLICY)
    return dict(schema_version=1,entries=entries,native_components=components,required_cases=878,required_label_views=2057,
        required_entity_label_views=sum({'smap':54,'msl':27,'smd':28}[e['case']['dataset']]*len(e['case']['required_label_policies']) for e in entries),
        score_metrics=list(SCORE),label_metrics=list(LABEL),
        official_TAB_source_root=TAB,official_TAB_commit=COMMIT,
        metric_python=sys.executable,environment=metric_environment(),isolated_metric_dependencies=VENDOR,
        strict_ratios=[.1,1,5],diagnostic_pooled_ratios=[.1,.5,1,2,3,5,10,15,20,25],
        metric_calls_per_entity_label_view=201,entity_macro_bootstrap_resamples=200,
        CPU_threads=2,settings=policy['settings'],resource_lock=policy['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(pins)],
        artifact_root='F:/aicoding/IIA_Data/experiments/TAB_full878_original_metric_suite_v1',
        state_root='experiments/runs/fm_TAB_full878_original_metric_suite_v1',
        citations=['https://github.com/decisionintelligence/TAB/tree/'+COMMIT,
            'https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf'],
        original_scope=original['original_scope'],
        boundary='Original TAB6score+15label functions,unchanged source;all878cases/2057label views. '
            '201calls per entity-label=6score+15*(3VALstrict+10original pooled TESTdiagnostics). '
            'All entities independent;never concatenate adjacent entities for range/affiliation. '
            'VUS original label-median tolerance,all integer windows and250thresholds retained;no clipping/subsampling. '
            'No best-label/threshold/orientation selection. Undefined/error cells retained,not zero-filled or silently dropped. '
            'Macro full-denominator results and conditional entity200bootstrap only when every entity finite; '
            'defined-subset means separately named with explicit coverage;not training-seed uncertainty. '
            'Require actual native model and fresh all-label review guards;unified frozen Python312 metric environment. '
            'Same16/28/16GiB read-only policy,3/6caps,auditlock;one whole entity at a time allows lock release between entities. '
            'No model training/rerun/new inference/source edits/old V2 and common105 pooling. '
            'Metric controls are not complete TAB data/training harness or historical author equivalence;all broader research remains.')


def verify(c):
    if c!=registration():raise ValueError('Complete878/2057/21original metrics,all windows/grids/guards and resource policy required')
    return c['entries']


def component(entry):
    if entry['component'] not in COMPONENTS:raise ValueError('Explicit original native component required')
    return importlib.import_module('scripts.flow_matching.'+entry['component'])


def source_status(entry):return scalar.source_status(entry['case'])


def ready_case(c,entry):
    module=component(entry);native_c=read(ROOT/module.CONFIG);case=entry['case']
    if case not in native_c['cases']:raise ValueError('Actual registered native case required')
    attempt,output=module.paths(native_c,case,'review');path=output/module.REPORTS['review'];guard=attempt/'guard_receipt.json'
    if not path.exists() or not guard.exists():return None
    proof,g=read(path),read(guard)
    native_path,native=module.ready_native(native_c,case)
    api=extra if entry['track']=='common105_DCD5' else scalar
    q,job,result,dataset,binding=api.source_binding(native_c,case)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/module.CONFIG)
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=module.worker_command(case,'review')
            or proof['source_binding']!=binding or native['source_binding']!=binding
            or proof['native_proof_sha256']!=sha(native_path)
            or not proof['all_original_and_native_arrays_independently_compared']
            or not proof['all_original_labels_metrics_events_bootstrap_and_grid_recomputed']
            or set(proof['all_label_view_results'])!=set(case['required_label_policies'])
            or proof['native_source_array_digests']!=native['source_arrays']):
        raise ValueError('Actual complete native+fresh all-label review and exact original guards required')
    if (q['evaluation']['strict_false_alarm_percent']!=c['strict_ratios']
            or q['evaluation']['tab_threshold_ratios_percent']!=c['diagnostic_pooled_ratios']
            or q['evaluation']['tab_commit']!=c['official_TAB_commit']):
        raise ValueError('All original strict and pooled diagnostic thresholds required')
    return dict(queue=q,job=job,result=result,dataset=dataset,
        lineage=dict(source_binding=binding,native_proof=str(native_path),native_sha256=sha(native_path),
            review_proof=str(path),review_sha256=sha(path),review_guard_sha256=sha(guard),
            native_scores=native['scores_path'],native_scores_sha256=native['scores_sha256']))


def load_metrics(c):
    root=ROOT/c['official_TAB_source_root'];loaded=sys.modules.get('ts_benchmark')
    if loaded is not None and Path(loaded.__file__).resolve()!= (root/'ts_benchmark/__init__.py').resolve():
        raise ValueError('A different TAB package is already imported')
    sys.path.insert(0,c['isolated_metric_dependencies']);sys.path.insert(0,str(root))
    score=importlib.import_module('ts_benchmark.evaluation.metrics.classification_metrics_score')
    label=importlib.import_module('ts_benchmark.evaluation.metrics.classification_metrics_label')
    if score.__all__!=c['score_metrics'] or label.__all__!=c['label_metrics']:
        raise ValueError('Exactly all original TAB exported metrics required')
    return dict(score={k:getattr(score,k) for k in score.__all__},label={k:getattr(label,k) for k in label.__all__})


def evaluate_cell(function,actual,predicted):
    """Preserve the original function's errors/warnings and private input types."""
    import numpy as np
    output=io.StringIO();started=time.monotonic();value=None;error=None
    with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        warnings.simplefilter('always')
        try:
            value=float(function(np.asarray(actual,dtype=np.float64).copy(),np.asarray(predicted,dtype=np.float64).copy()))
            status='finite' if np.isfinite(value) else 'nonfinite'
        except Exception:
            status='error';error=traceback.format_exc()
    return dict(status=status,value=value if status=='finite' else None,error=error,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
        original_stdout_stderr=output.getvalue(),elapsed_seconds=time.monotonic()-started)


def thresholds(c,result):
    strict=result['metrics']['strict'];grid=result['metrics']['tab_core_diagnostic']['ratio_grid']
    if set(strict)!={str(r) for r in c['strict_ratios']} or [r['ratio_percent'] for r in grid]!=c['diagnostic_pooled_ratios']:
        raise ValueError('All original calibration values required without selection')
    return {**{'validation/'+str(r):strict[str(r)]['calibration']['threshold'] for r in c['strict_ratios']},
        **{'diagnostic_pooled_TEST/'+str(r['ratio_percent']):r['threshold'] for r in grid}}


def evaluate_entity(c,actual_scores,label_views,result,metrics,progress=lambda value:None):
    import numpy as np
    if actual_scores.ndim!=1 or not len(actual_scores) or not np.isfinite(actual_scores).all():
        raise ValueError('Every original timestamp needs one unchanged finite score')
    original_bytes=actual_scores.tobytes();rows={};levels=thresholds(c,result)
    for policy,labels in label_views.items():
        if labels.shape!=actual_scores.shape or not np.isin(labels,[0,1]).all():raise ValueError('Complete original binary label overlay required')
        labels_bytes=labels.tobytes();cells={}
        for name,function in metrics['score'].items():
            identity='score/'+name;progress(dict(policy=policy,metric=identity))
            cells[identity]=evaluate_cell(function,labels,actual_scores)
        for level,threshold in levels.items():
            predicted=actual_scores>threshold
            for name,function in metrics['label'].items():
                identity=level+'/'+name;progress(dict(policy=policy,metric=identity))
                cells[identity]=evaluate_cell(function,labels,predicted)
        if len(cells)!=c['metric_calls_per_entity_label_view'] or labels.tobytes()!=labels_bytes or actual_scores.tobytes()!=original_bytes:
            raise ValueError('All201metric calls and immutable original arguments required')
        rows[policy]=cells
    return rows


def validate_cells(c,rows,policies):
    expected={'score/'+name for name in c['score_metrics']}
    levels=['validation/'+str(r) for r in c['strict_ratios']]+['diagnostic_pooled_TEST/'+str(r) for r in c['diagnostic_pooled_ratios']]
    expected.update(level+'/'+name for level in levels for name in c['label_metrics'])
    if set(rows)!=set(policies) or len(expected)!=201:raise ValueError('All exact original labels/201metric identities required')
    for cells in rows.values():
        if set(cells)!=expected:raise ValueError('Every original score/label/threshold metric must be retained')
        for cell in cells.values():
            if cell['status']=='finite':
                if not isinstance(cell['value'],(int,float)) or isinstance(cell['value'],bool) or not math.isfinite(cell['value']):
                    raise ValueError('Finite metric output required for finite status')
            elif cell['status'] in ['error','nonfinite']:
                if cell['value'] is not None or (cell['status']=='error' and not cell['error']):
                    raise ValueError('Undefined results cannot be imputed or hidden')
            else:raise ValueError('Every original metric must have an explicit outcome')


def paths(c,entry,prefix):
    if prefix!='summary' and (not prefix.startswith('e') or not prefix[1:].isdigit()):raise ValueError('Exact original entity array prefix required')
    return (ROOT/c['state_root']/'attempts'/entry['key']/prefix,Path(c['artifact_root'])/'cases'/entry['key']/prefix)


def worker_command(c,entry,prefix):
    return [c['metric_python'],'-B','-X','utf8','-u','-m',MODULE,'--case-key',entry['key'],'--entity-prefix',prefix]


def worker(c,entry,prefix):
    actual=metric_environment()
    if actual!=c['environment']:raise ValueError('Exact unified read-only TAB metric environment required')
    bound=ready_case(c,entry)
    if bound is None:raise ValueError('Actual complete source/native/all-label review required')
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical
    _,output=paths(c,entry,prefix);output.mkdir(parents=True,exist_ok=False)
    dataset=bound['dataset'];entity=next(e for e in dataset['entities'] if e['array_prefix']==prefix)
    with np.load(bound['lineage']['native_scores'],allow_pickle=False) as native,np.load(ROOT/bound['job']['output_directory']/'scores.npz',allow_pickle=False) as source:
        scores=native[prefix+'_test'];require_identical(scores,source[prefix+'_test'])
        native_labels=native[prefix+'_labels'];require_identical(native_labels,source[prefix+'_labels'])
    views={}
    for policy,record in dataset['labels_only_controls'].items():
        if sha(record['path'])!=record['sha256']:raise ValueError('Original labels changed')
        with np.load(record['path'],allow_pickle=False) as labels:views[policy]=labels[prefix+'_labels']
    if set(views)!=set(entry['case']['required_label_policies']):raise ValueError('All4/2/1label controls required')
    require_identical(views[dataset['primary_label_policy']],native_labels)
    if len(scores)!=entity['test_points']:raise ValueError('All real original entity test points required')
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write as atomic
    def progress(value):
        atomic(output/'progress.json',dict(case_id=entry['case']['id'],entity=entity['entity_id'],
            captured_utc=datetime.now(timezone.utc).isoformat(),**value))
    rows=evaluate_entity(c,scores,views,bound['result'],load_metrics(c),progress)
    validate_cells(c,rows,entry['case']['required_label_policies'])
    after=ready_case(c,entry)
    if after is None or after['lineage']!=bound['lineage']:raise ValueError('All actual native and source evidence must stay unchanged')
    states=Counter(cell['status'] for view in rows.values() for cell in view.values())
    write(output/REPORT,dict(passed=True,case=entry['case'],track=entry['track'],entity=entity['entity_id'],array_prefix=prefix,
        configuration_sha256=sha(ROOT/CONFIG),source_lineage=bound['lineage'],environment=actual,
        test_points=len(scores),source_scores_digest=array_digest(scores),label_digests={p:array_digest(v) for p,v in views.items()},
        all_original_metric_calls_attempted=True,metric_calls=sum(states.values()),metric_outcome_counts=dict(states),
        all_metric_values_finite=states.get('error',0)+states.get('nonfinite',0)==0,metric_results=rows,
        entities_concatenated=False,original_inputs_and_thresholds_unchanged=True,
        neural_training_or_inference_performed=False,full_TAB_training_or_author_harness_equivalence=False,
        range_metric_training_seed_uncertainty_certified=False,all_original_experiments_complete=False))


def checked_fragment(c,entry,prefix):
    attempt,output=paths(c,entry,prefix);path=output/REPORT;guard=attempt/'guard_receipt.json'
    proof,g=read(path),read(guard)
    validate_cells(c,proof['metric_results'],entry['case']['required_label_policies'])
    if (not proof['passed'] or proof['case']!=entry['case'] or proof['array_prefix']!=prefix
            or proof['configuration_sha256']!=sha(ROOT/CONFIG) or not proof['all_original_metric_calls_attempted']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(c,entry,prefix)):
        raise ValueError('Actual complete original TAB cell coverage and successful exact guard required')
    return path,proof


def aggregate(c,fragments,seed):
    import numpy as np
    policies=list(fragments[0]['metric_results']);n=len(fragments)
    indices=np.random.default_rng(seed+10000).integers(0,n,size=(c['entity_macro_bootstrap_resamples'],n))
    rows={}
    for policy in policies:
        names=list(fragments[0]['metric_results'][policy]);rows[policy]={}
        for name in names:
            cells=[p['metric_results'][policy][name] for p in fragments]
            valid=np.array([x['value'] for x in cells if x['status']=='finite'],dtype=np.float64)
            complete=len(valid)==n
            ci=np.percentile(valid[indices].mean(axis=1),[2.5,97.5]).tolist() if complete else None
            rows[policy][name]=dict(total_entities=n,finite_entities=len(valid),
                error_entities=sum(x['status']=='error' for x in cells),nonfinite_entities=sum(x['status']=='nonfinite' for x in cells),
                complete_entity_macro_mean=float(valid.mean()) if complete else None,
                defined_subset_macro_mean=float(valid.mean()) if len(valid) else None,
                complete_entity_macro_CI95=ci,CI_resamples=c['entity_macro_bootstrap_resamples'] if complete else 0,
                CI_unit='entity_conditional_on_fixed_model_and_threshold',training_seed_uncertainty=False)
    return rows


def summary_worker(c,entry):
    bound=ready_case(c,entry)
    if bound is None:raise ValueError('Actual complete native case required')
    fragments=[];pins=[]
    for entity in bound['dataset']['entities']:
        path,proof=checked_fragment(c,entry,entity['array_prefix'])
        if proof['entity']!=entity['entity_id'] or proof['source_lineage']!=bound['lineage']:raise ValueError('All original entities from the same model/threshold protocol required')
        fragments.append(proof);pins.append(dict(path=str(path),sha256=sha(path)))
    _,output=paths(c,entry,'summary');output.mkdir(parents=True,exist_ok=False)
    write(output/'complete_case_TAB_metric_controls.json',dict(passed=True,case=entry['case'],track=entry['track'],
        configuration_sha256=sha(ROOT/CONFIG),source_lineage=bound['lineage'],entities=len(fragments),fragments=pins,
        metric_calls=sum(x['metric_calls'] for x in fragments),macro_metrics=aggregate(c,fragments,entry['case']['seed']),
        undefined_metrics_preserved=True,cross_entity_event_join_performed=False,source_training_or_scoring_modified=False,
        original_local_training_fidelity_limits_retained=True,full_TAB_training_or_author_harness_equivalence=False,
        all_original_experiments_complete=False))


def controller(c,entries):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing full878 TAB owner preserved')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve every prior metric attempt')
    done=[];failed={};source_failed={};entity_done={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_cases=878,required_label_views=2057,active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_case_keys=list(done),failed_cases_preserved=dict(failed),
            source_failures_preserved=dict(source_failed),completed_entities=dict(entity_done),captured_utc=datetime.now(timezone.utc).isoformat())
        atomic(base/'status.json',state)
    def stage(entry,prefix):
        attempt,output=paths(c,entry,prefix)
        if attempt.exists() or output.exists():raise FileExistsError('Preserve previous metric fragment without retry')
        attempt.mkdir(parents=True,exist_ok=False);command=worker_command(c,entry,prefix)
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_original_TAB_metric_suite',active_pid=child.pid,active_case=entry['case']['id'],active_entity_prefix=prefix)
            code,usage=capped_wait(child,c,api,update)
        report=output/('complete_case_TAB_metric_controls.json' if prefix=='summary' else REPORT)
        ok=code==0 and not usage['resource_guard_aborted'] and report.exists() and read(report)['passed']
        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(report) if report.exists() else None))
        if not ok:raise ValueError('Original TAB fragment failed;preserve without retry')
    try:
        while len(done)+len(failed)+len(source_failed)<len(entries):
            for entry in entries:
                identity=entry['key']
                if identity in done or identity in failed or identity in source_failed:continue
                try:
                    old=source_status(entry)
                    if old is not None and old['kind']=='source_failed_preserved':source_failed[identity]=old;continue
                    bound=ready_case(c,entry)
                    if bound is None:continue
                    for entity in bound['dataset']['entities']:
                        with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
                            verify(read(ROOT/CONFIG));stage(entry,entity['array_prefix'])
                        entity_done[identity]=entity_done.get(identity,0)+1;update(state='between_original_entities',active_pid=None)
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):stage(entry,'summary')
                    done.append(identity);update(state='between_complete_TAB_cases',active_pid=None)
                except Exception as error:
                    failed[identity]=traceback.format_exc();update(state='original_TAB_failure_preserved',active_pid=None)
            update(state='waiting_actual_native_and_fresh_all_label_reviews',active_pid=None);time.sleep(10)
        update(state='all878_metric_outcomes_terminal_other_research_requirements_remain',active_pid=None)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--case-key');p.add_argument('--entity-prefix');a=p.parse_args()
    if a.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=878,label_views=2057)));return
    c=read(ROOT/CONFIG);entries=verify(c)
    if a.probe_only:
        ready=[e['key'] for e in entries if ready_case(c,e) is not None]
        print(json.dumps(dict(passed=True,cases=878,label_views=2057,actual_native_ready_case_keys=ready,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif a.case_key:
        entry=next(e for e in entries if e['key']==a.case_key)
        if a.entity_prefix=='summary':summary_worker(c,entry)
        elif a.entity_prefix:worker(c,entry,a.entity_prefix)
        else:raise ValueError('An explicit full original entity is required')
    else:controller(c,entries)


if __name__=='__main__':main()
