"""Original Grimshaw SPOT/POT controls for all25 unique FP64 cases.

Thirty two-track comparison slots reuse the five original SMD cases, exactly
as the frozen matched105 registry specifies; they are not five new runs.

Reuse complete fixed scores; no model fitting or generative inference. Exact
author threshold/PA functions retain their epsilon and index0 behavior. The
old SciPy-GPD approximation is preserved as historical code, never relabeled.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.flow_matching import run_TAB_full878_metric_suite_v1 as original
from scripts.flow_matching import run_common105_matched577_TAB_v1 as matched
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.evaluate_tranad_original_SPOT_full30_v1'
CONFIG='configs/reproducibility/fm_tranad_original_SPOT_full30.2026-10-16_v1.json'
TEST='tests/test_fm_tranad_original_SPOT_full30_v1.py'
MODEL='configs/models/fm_tranad_author_source_v1/TranAD.json'
REPORT='complete_original_SPOT_fixed_model_controls.json'


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def registration():
    entries=[]
    for module,required in [(original,15),(matched,10)]:
        c=read(ROOT/module.CONFIG)
        selected=[e for e in c['entries'] if e['case']['family']=='tranad64']
        expected={(d,s) for d in (['smap','msl','smd'] if module is original else ['smap','msl']) for s in range(1103,1108)}
        if len(selected)!=required or {(e['case']['dataset'],e['case']['seed']) for e in selected}!=expected:
            raise ValueError('All frozen unique FP64 cases and primary seeds required')
        entries.extend(dict(entry=e,source_configuration=module.CONFIG,source_module=module.MODULE) for e in selected)
    references=[dict(comparison_track='common105_matched577',reference_source_case=r['entry']['case']['id'],
        source_configuration=original.CONFIG,dataset='smd',seed=r['entry']['case']['seed'],new_independent_case=False)
        for r in entries if r['source_module']==original.MODULE and r['entry']['case']['dataset']=='smd']
    resource='configs/runtime/fm_spectral_full70_artifact_reviews.v1.json';policy=read(ROOT/resource)
    model=read(ROOT/MODEL)
    paths=[MODULE.replace('.','/')+'.py',TEST,MODEL,original.CONFIG,matched.CONFIG,
        original.MODULE.replace('.','/')+'.py',matched.MODULE.replace('.','/')+'.py',resource,
        'src/iia_benchmark/models/tranad_author_source_v1.py','scripts/mtsad/score_pot.py']
    paths += [(Path(model['source_root'])/p['path']).as_posix() for p in model['source_receipts']]
    return dict(schema_version=1,entries=entries,required_cases=25,required_comparison_slots=30,
        shared_original_SMD_comparison_references=references,primary_training_seeds=[1103,1104,1105,1106,1107],
        required_label_views=sum(len(x['entry']['case']['required_label_policies']) for x in entries),
        source_model_configuration=MODEL,original_risk=1e-5,deterministic_fit_seed_offset=100000,
        calibration_population='complete_saved_fit_training_scores_per_original_entity',
        other_label_views_share_same_fixed_entity_calibration=True,CPU_threads=2,python=sys.executable,
        settings=policy['settings'],resource_lock=policy['resource_lock'],
        state_root='experiments/runs/fm_tranad_original_SPOT_full30_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/tranad_original_SPOT_full30_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(paths))],
        original_scope=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273,paper_scalar_targets=702),
        citations=['https://doi.org/10.14778/3514061.3514067','Frozen imperial-qore/TranAD src/spot.py and src/pot.py'],
        boundary='Exact original SPOT Grimshaw fit,source dataset lm,risk1e-5,static dynamic=False thresholds and original PA. '
            'No SciPy-GPD approximation,quantile inversion,score truncation,TEST-label threshold optimization or model refit. '
            'One fixed seeded POT fit per entity,all original label views share its threshold;source PA epsilon/index0 semantics retained. '
            'The calibrated model remains the original local FP64 adaptation,not a corrected author architecture. '
            'All15fullraw+10newmatched cases and full entities required;five original SMD cases also serve matched comparisons. '
            'All30comparison slots retained without counting shared SMD as new cases. Source fit scores may come from the local80percent training split; '
            'no calibration on saved validation/TEST scores. Source history,precision environment and fullTAB/author equivalence unproven. '
            'Degenerate constant training score population is explicitly undefined,never replaced by a percentile. '
            'Failure/abort preserved,no automatic retry. All original45/681/273/702 and existing training tracks remain.',
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration() or len(c['entries'])!=25:raise ValueError('Complete frozen source SPOT control scope required')
    return c['entries']


def source_api(record):
    module=original if record['source_module']==original.MODULE else matched
    if module.CONFIG!=record['source_configuration']:raise ValueError('Explicit fullraw/matched lineage required')
    return module,read(ROOT/module.CONFIG)


def ready(record):
    module,c=source_api(record)
    return module.ready_case(c,record['entry'])


def metrics_namespace(dataset):
    from iia_benchmark.models.tranad_author_source_v1 import load_author_metrics
    return load_author_metrics(read(ROOT/MODEL),dataset.upper())


def outcomes(values):
    import math
    result={k:float(v) if math.isfinite(float(v)) else None for k,v in values.items()}
    return dict(values=result,nonfinite_fields=[k for k,v in result.items() if v is None])


def exact_entity(train,test,views,primary,ns,fit_seed,risk):
    import numpy as np
    if (train.ndim!=1 or test.ndim!=1 or not len(train) or not len(test)
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError('Complete finite original score vectors required')
    if primary not in views or any(y.shape!=test.shape or not np.isin(y,[0,1]).all() for y in views.values()):
        raise ValueError('All complete original aligned binary label views required')
    if np.unique(train).size<2:
        return dict(status='undefined_constant_training_scores_no_source_peaks',threshold=None,controls={},
            calibration_undefined_not_percentile_imputed=True),{}
    originals={k:v.tobytes() for k,v in views.items()};before=(train.tobytes(),test.tobytes())
    np.random.seed(fit_seed)
    primary_values,primary_prediction=ns['pot_eval'](train.copy(),test.copy(),views[primary].copy(),q=risk)
    threshold=float(primary_values['threshold'])
    if not np.isfinite(threshold):raise ValueError('Original source returned a nonfinite threshold;preserve failure')
    controls={};arrays={}
    for policy,labels in views.items():
        raw=(test>threshold)
        adjusted=ns['adjust_predicts'](test.copy(),labels.copy(),threshold=threshold)
        if policy==primary:assert adjusted.tobytes()==primary_prediction.tobytes()
        a=ns['calc_point2point'](raw,labels);b=ns['calc_point2point'](adjusted,labels)
        keys=['f1','precision','recall','TP','TN','FP','FN','ROC/AUC']
        controls[policy]=dict(raw_source_point_metrics=outcomes(dict(zip(keys,a))),source_PA_point_metrics=outcomes(dict(zip(keys,b))),
            positives=int(labels.sum()),original_source_epsilon=1e-5,original_source_PA_index0_behavior_retained=True)
        arrays[policy+'__raw']=raw;arrays[policy+'__source_PA']=adjusted
    assert before==(train.tobytes(),test.tobytes()) and all(views[k].tobytes()==v for k,v in originals.items())
    return dict(status='complete_original_source_SPOT',threshold=threshold,controls=controls,fit_seed=fit_seed,
        original_risk=risk,original_lm=ns['lm'],training_points=len(train),test_points=len(test),
        threshold_uses_TEST_labels=False,PA_uses_TEST_labels=True,static_POT_stream=True,
        source_primary_return=outcomes(primary_values)),arrays


def worker(c,record):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    bound=ready(record)
    if bound is None:raise ValueError('Actual complete source/native/all-label review required')
    case=record['entry']['case'];dataset=bound['dataset'];ns=metrics_namespace(case['dataset'])
    out=Path(c['artifact_root'])/case['id'];out.mkdir(parents=True,exist_ok=False)
    label_arrays={}
    for policy,source in dataset['labels_only_controls'].items():
        assert sha(source['path'])==source['sha256']
        with np.load(source['path'],allow_pickle=False) as z:label_arrays[policy]={k:z[k] for k in z.files}
    entities=[];predictions={}
    with np.load(bound['lineage']['native_scores'],allow_pickle=False) as scores:
        for index,entity in enumerate(dataset['entities']):
            prefix=entity['array_prefix'];train=scores[prefix+'_train'];test=scores[prefix+'_test']
            views={policy:v[prefix+'_labels'] for policy,v in label_arrays.items()}
            assert set(views)==set(case['required_label_policies']) and len(test)==entity['test_points']
            write(out/'progress.json',dict(entity=entity['entity_id'],completed_entities=len(entities),required_entities=len(dataset['entities']),phase='original_Grimshaw_SPOT_and_all_label_controls'))
            result,arrays=exact_entity(train,test,views,dataset['primary_label_policy'],ns,c['deterministic_fit_seed_offset']+case['seed']+index,c['original_risk'])
            entities.append(dict(entity=entity['entity_id'],array_prefix=prefix,train_scores=array_digest(train),test_scores=array_digest(test),
                label_digests={k:array_digest(v) for k,v in views.items()},**result))
            predictions.update({prefix+'__'+k:v for k,v in arrays.items()})
    after=ready(record);assert after is not None and after['lineage']==bound['lineage']
    np.savez_compressed(out/'all_original_entity_predictions.npz',**predictions)
    exclusive(out/REPORT,dict(passed=True,case=case,source_configuration=record['source_configuration'],source_lineage=bound['lineage'],
        configuration_sha256=sha(ROOT/CONFIG),entities=entities,complete_original_entities=len(entities),
        undefined_calibrations=sum(x['status']!='complete_original_source_SPOT' for x in entities),
        prediction_file=str(out/'all_original_entity_predictions.npz'),prediction_sha256=sha(out/'all_original_entity_predictions.npz'),
        old_local_model_architecture_and_preprocessing_unchanged=True,full_TAB_or_historical_author_equivalence=False,
        source_model_training_or_inference_performed=False,new_training_seed_slots=0,all_original_experiments_complete=False))


def controller(c):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None or (base/'status.json').exists():
        if owner:owner.close()
        raise FileExistsError('Never duplicate or overwrite this source SPOT attempt')
    done=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(value):
        state.update(value,completed_ids=done,failed_preserved=failed,required_cases=c['required_cases'],
            captured_utc=datetime.now(timezone.utc).isoformat(),all_original_experiments_complete=False);write(base/'status.json',state)
    try:
        while len(done)+len(failed)<c['required_cases']:
            for record in c['entries']:
                case=record['entry']['case'];identity=case['id']
                if identity in done or identity in failed:continue
                try:
                    if ready(record) is None:continue
                    attempt=base/'attempts'/identity
                    if attempt.exists() or (Path(c['artifact_root'])/identity).exists():raise FileExistsError('Preserve prior SPOT attempt without retry')
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update):
                        attempt.mkdir(parents=True,exist_ok=False)
                        command=[c['python'],'-B','-X','utf8','-u','-m',MODULE,'--case-id',identity]
                        with (attempt/'stdout.log').open('xb') as stdout,(attempt/'stderr.log').open('xb') as stderr:
                            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
                                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',
                                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            update(dict(state='running_complete_original_source_SPOT',active_pid=child.pid,exact_child_handle=handle,active_case=identity))
                            code,usage=capped_wait(child,c,api,lambda v:update(dict(v,state='running_complete_original_source_SPOT')))
                        path=Path(c['artifact_root'])/identity/REPORT;passed=code==0 and not usage['resource_guard_aborted'] and path.exists() and read(path)['passed']
                        exclusive(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(path) if path.exists() else None))
                        if not passed:raise ValueError('Full original SPOT control failed; preserve without retry')
                        done.append(identity)
                except Exception as error:failed[identity]=repr(error)
                update(dict(state='between_complete_original_SPOT_cases',active_pid=None))
            update(dict(state='waiting_actual_native_FP64_cases',active_pid=None));time.sleep(10)
        update(dict(state='completed' if not failed else 'completed_with_preserved_failures',active_pid=None))
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--case-id');parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if args.probe_only:print(json.dumps(dict(cases=c['required_cases'],comparison_slots=c['required_comparison_slots'],numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if args.case_id:worker(c,next(r for r in c['entries'] if r['entry']['case']['id']==args.case_id))
    else:controller(c)


if __name__=='__main__':main()
