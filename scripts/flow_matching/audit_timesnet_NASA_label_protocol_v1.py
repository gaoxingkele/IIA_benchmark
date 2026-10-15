"""Fixed-score NASA label/threshold diagnostics on all five original models.

Require completed full checkpoint replay first. Never fit,flip scores,tune a
validation threshold or select a label policy from test performance.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.flow_matching import prepare_original_full_entity_tsad_v1 as data
from scripts.flow_matching import run_timesnet_SMAP_remaining_full_audits_v2 as audit

ROOT,read,sha=data.ROOT,data.read,data.sha
MODULE='scripts.flow_matching.audit_timesnet_NASA_label_protocol_v1'
CONFIG='configs/reproducibility/fm_timesnet_all5_NASA_label_protocol.2026-10-16_v1.json'
TEST='tests/test_fm_timesnet_NASA_label_protocol_v1.py'
POLICIES=('source_first_exclusive','source_first_inclusive','source_last_exclusive','source_last_inclusive')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def registration():
    original=read(ROOT/audit.GUARD)
    pins=[MODULE.replace('.','/')+'.py',TEST,audit.CONFIG,audit.GUARD,audit.QUEUE,data.CONFIG,
        'scripts/flow_matching/prepare_original_full_entity_tsad_v1.py',
        'scripts/flow_matching/run_timesnet_SMAP_remaining_full_audits_v2.py',
        'scripts/flow_matching/audit_timesnet_constant_fit_complete_v1.py',
        'scripts/flow_matching/run_tsad_experiment.py',
        'scripts/flow_matching/verify_tranad_precision_and_smd_completion_v1.py',
        'src/iia_benchmark/evaluation/mtsad_metrics.py','src/iia_benchmark/evaluation/mtsad_strict.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/cached_resource_snapshot_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py']
    q=read(ROOT/audit.QUEUE)
    jobs=[next(j for j in q['jobs'] if j['id']==f'fm_strict_timesnet__smap__registered__seed{s}__constant_fit_v2') for s in range(1103,1108)]
    return dict(schema_version=1,source_queue=audit.QUEUE,jobs=jobs,seeds=list(range(1103,1108)),
        required_original_models=5,entities_per_model=51,test_points_per_model=408160,
        raw_data_configuration=data.CONFIG,label_policies=list(POLICIES),
        source_evaluation=q['evaluation'],python=q['python'],
        environment=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
            pandas=importlib.metadata.version('pandas'),scikit_learn=importlib.metadata.version('scikit-learn')),
        settings=original['settings'],resource_lock=original['resource_lock'],
        state_root='experiments/runs/fm_timesnet_all5_NASA_label_protocol_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/timesnet_all5_NASA_label_protocol_v1',
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in pins],
        exact_coordinate_mapping_required=True,no_tolerance_or_unregistered_offset_permitted=True,
        baseline_label_rule='existing original CSV labels;alternatives only when every coordinate mapping is proven',
        boundary='All5original TimesNet seeds on the unchanged51entity subset,after successful full native replay. '
            'No retraining or reinference. Four NASA label views only if complete raw TEST/CSV coordinate identity '
            'and unambiguous label mappings hold;otherwise diagnostic records data inequivalence and skips views. '
            'Fixed validation thresholds,noPA versus entitywisePA and original test-calibrated oracle grid separated. '
            'Conditional entity uncertainty is not five-seed uncertainty. Full109entities,873jobs and all other scope remain.')


def verify(config):
    if config!=registration():raise ValueError('All5models/all51entities/all4label controls and unchanged readonly policy required')
    return config['jobs']


def source_native(config,job):
    seed=job['seed']
    if seed==1103:
        stage_path=ROOT/'configs/reproducibility/fm_timesnet_SMAP1103_constant_fit_full_replay.native.2026-10-14_v1.json'
        stage=read(stage_path);guard_path=ROOT/stage['state_root']/'receipt.json';flag='full_saved_score_review_passed'
    else:
        parent=read(ROOT/audit.CONFIG);stage=audit.stage_config(parent,job,'native')
        stage_path=ROOT/stage['state_root']/'frozen_stage.json';guard_path=ROOT/stage['state_root']/'receipt.json';flag='passed'
    proof_path=ROOT/stage['output_directory']/'independent_full_saved_score_verification.json'
    if not guard_path.exists():return None
    g=read(guard_path)
    if not g[flag] or g['exit_code']!=0 or g['resource_guard_aborted']:
        raise ValueError('Original complete native replay failed;preserve without numeric retry')
    p=read(proof_path)
    if (not p['passed'] or p['configuration_sha256']!=sha(stage_path) or p['canonical_id']!=job['id']
            or p['full_entities']!=51 or p['full_test_points']!=408160 or not p['full_native_checkpoint_reinference_passed']
            or not p['all_replayed_timestamp_scores_bitwise_equal']):raise ValueError('Actual all51entity native proof required')
    if seed==1103:audit.original.binding(stage)
    else:audit.binding(stage)
    result=ROOT/job['output_directory']/'result.json'
    if sha(result)!=p['original_result_sha256']:raise ValueError('Original source result bytes changed')
    if sha(ROOT/stage['output_directory']/'replayed_scores.npz')!=p['replay_scores_sha256']:
        raise ValueError('Current full native replay scores differ')
    return dict(native_proof_sha256=sha(proof_path),native_guard_sha256=sha(guard_path),
        source_result_sha256=sha(result),native_score_sha256=p['replay_scores_sha256'])


def mapped_labels(values,timestamps,raw,metadata_rows):
    """Prove each CSV coordinate;never guess offsets or select using scores."""
    import numpy as np
    if (values.ndim!=2 or raw.ndim!=2 or values.shape[1]!=raw.shape[1] or len(values)!=len(timestamps)
            or not np.isfinite(timestamps).all() or not np.equal(timestamps,np.floor(timestamps)).all()
            or np.any(np.diff(timestamps)<=0)):
        raise ValueError('Complete numeric features and unique monotonic integer timestamps required')
    candidates=[];vectors=[]
    for origin in [0,1]:
        indices=timestamps.astype(np.int64)-origin
        if not len(indices) or indices.min()<0 or indices.max()>=len(raw):continue
        expected=np.asarray(raw[indices],dtype=np.float64);equal=np.array_equal(values,expected,equal_nan=True)
        finite=np.isfinite(values)&np.isfinite(expected)
        candidates.append(dict(origin=origin,compared_coordinates=int(values.size),every_coordinate_equal=bool(equal),
            maximum_finite_absolute_difference=float(np.max(np.abs(values[finite]-expected[finite]))) if finite.any() else None))
        if equal:
            vectors.append({p:data.label_array(metadata_rows,len(raw),p)[indices] for p in POLICIES})
    stable=bool(vectors) and all(all(np.array_equal(v[p],vectors[0][p]) for p in POLICIES) for v in vectors)
    return dict(full_mapping_proven=stable,candidates=candidates,
        exact_source_mappings=sum(c['every_coordinate_equal'] for c in candidates),
        timestamp_ambiguity_changes_labels=bool(vectors) and not stable,
        tolerance_used=False,offset_inferred_or_test_selected=False),vectors[0] if stable else None


def protocol_diagnostics(entities,threshold):
    import numpy as np
    from iia_benchmark.evaluation.mtsad_metrics import point_adjust,pointwise_metrics
    if not entities or not np.isfinite(threshold):raise ValueError('Full finite fixed-threshold diagnostic required')
    for truth,score in entities.values():
        if (truth.ndim!=1 or score.shape!=truth.shape or not len(truth)
                or not np.isin(truth,[0,1]).all() or not np.isfinite(score).all()):
            raise ValueError('Every original binary target and finite score required')
    truth=np.concatenate([y for y,s in entities.values()]).astype(bool)
    scores=np.concatenate([s for y,s in entities.values()])
    prediction=scores>threshold
    adjusted=np.concatenate([point_adjust(y,s>threshold) for y,s in entities.values()])
    anomaly_count=int(truth.sum());n=len(truth)
    return dict(validation_threshold=float(threshold),same_fixed_scores_used=True,test_points=n,anomaly_points=anomaly_count,
        anomaly_prevalence=anomaly_count/n,validation_only_noPA=pointwise_metrics(truth,prediction),
        same_validation_threshold_entitywise_PA_diagnostic=pointwise_metrics(truth,adjusted),
        always_anomaly_noPA_F1=2*anomaly_count/(n+anomaly_count),
        PA_oracle_fields_are_strict_leaderboard_metrics=False)


def worker(config,job,output):
    binding=source_native(config,job)
    if binding is None:raise ValueError('Actual full native model proof required before numerical imports')
    actual=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
        pandas=importlib.metadata.version('pandas'),scikit_learn=importlib.metadata.version('scikit-learn'))
    if actual!=config['environment']:raise ValueError('Registered numerical audit environment required')
    import numpy as np
    import pandas as pd
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    from scripts.flow_matching.verify_tranad_precision_and_smd_completion_v1 import check_arrays
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=read(ROOT/job['output_directory']/'result.json');manifest=read(ROOT/job['dataset_manifest_path'])
    dataset=next(d for d in manifest['datasets'] if d['id']=='smap')
    if sha(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Original prepared inputs changed')
    raw_jobs={j['entity']:j for j in read(ROOT/config['raw_data_configuration'])['jobs'] if j['dataset']=='SMAP'}
    coverage=[];views={p:{} for p in POLICIES};stored_labels={p:{} for p in POLICIES}
    score_path=ROOT/job['output_directory']/'scores.npz'
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs,np.load(score_path,allow_pickle=False) as saved:
        entities,lengths,validation=check_arrays(inputs,saved,dataset)
        entities={k:(y.copy(),s.copy()) for k,(y,s) in entities.items()}
        training=np.concatenate([saved[e['array_prefix']+'_train'] for e in dataset['entities']])
        metrics=evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],config['source_evaluation'],job['seed'])
        if metrics!=result['metrics']:raise ValueError('Every original point/core/event/bootstrap metric must agree exactly')
        for entry in dataset['entities']:
            name=entry['entity_id'];entity=name.removeprefix('SMAP_');raw_job=raw_jobs[entity]
            facts=raw_job['raw']['test']
            if sha(facts['path'])!=facts['sha256']:raise ValueError('Full original raw TEST changed')
            path=next(p for p in dataset['source_files'] if Path(p['path']).name==name+'_test.csv')
            if sha(ROOT/path['path'])!=path['sha256']:raise ValueError('Original CSV changed')
            frame=pd.read_csv(ROOT/path['path'],float_precision='round_trip')
            features=[c for c in frame.columns if c.startswith('feature_')]
            raw=np.load(facts['path'],mmap_mode='r',allow_pickle=False)
            crosswalk,labels=mapped_labels(frame[features].to_numpy(np.float64),frame['timestamp'].to_numpy(np.float64),raw,raw_job['metadata_rows'])
            if not np.array_equal(frame['is_anomaly'].to_numpy(),entities[name][0]):raise ValueError('Original CSV and model target labels differ')
            coverage.append(dict(entity=name,full_test_points=len(frame),**crosswalk,
                original_CSV_label_policy_matches=[p for p in POLICIES if labels is not None and np.array_equal(labels[p],entities[name][0])]))
            if labels is not None:
                for p in POLICIES:
                    views[p][name]=(labels[p],entities[name][1]);stored_labels[p][entry['array_prefix']+'_labels']=labels[p]
    threshold=metrics['strict']['1']['calibration']['threshold'];controls={}
    all_mapped=len(coverage)==51 and all(r['full_mapping_proven'] for r in coverage)
    if all_mapped:
        for policy in POLICIES:
            path=output/(policy+'_labels.npz')
            with path.open('xb') as stream:np.savez_compressed(stream,**stored_labels[policy])
            m=evaluate_saved(views[policy],validation,training,dataset['input_sha256'],result['checkpoint_sha256'],config['source_evaluation'],job['seed'])
            if any(m['strict'][r]['calibration']!=metrics['strict'][r]['calibration'] for r in metrics['strict']):
                raise ValueError('All original validation thresholds must remain identical across label controls')
            controls[policy]=dict(metrics=m,protocol_diagnostics=protocol_diagnostics(views[policy],threshold),
                labels_path=str(path),labels_sha256=sha(path),same_scores_sha256=result['scores_sha256'],
                label_policy_selected_by_performance=False)
    if sha(score_path)!=result['scores_sha256'] or source_native(config,job)!=binding:raise ValueError('Preserve all original model and score bytes')
    write(output/'full_label_protocol_audit.json',dict(passed=True,job=job,configuration_sha256=sha(ROOT/CONFIG),
        source_binding=binding,environment=actual,full_entities=51,full_test_points=sum(lengths.values()),
        original_metrics_exactly_recomputed=True,baseline_protocol_diagnostics=protocol_diagnostics(entities,threshold),
        original_test_calibrated_oracle_grid=metrics['tab_core_diagnostic'],raw_CSV_crosswalk=coverage,
        all_raw_label_controls_evaluable=all_mapped,label_controls=controls,
        controls_skipped_reason=None if all_mapped else 'At least one complete source TEST/CSV coordinate or label mapping unproven;no guessed offsets or tolerance',
        model_training_or_inference_executed=False,additional_training_seeds=0,
        conditional_entity_uncertainty_not_training_seed_uncertainty=True,complete_five_seed_comparison=False,
        all_original_experiments_complete=False))


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](root/'controller.lock')
    if owner is None:raise RuntimeError('Label diagnostic owner already live')
    if (root/'status.json').exists():owner.close();raise FileExistsError('Preserve every prior diagnostic attempt')
    completed=[];failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_original_models=5,
        active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed=list(completed),failed_preserved=dict(failed),captured_utc=datetime.now(timezone.utc).isoformat());atomic(root/'status.json',state)
    try:
        for job in jobs:
            try:
                while source_native(config,job) is None:
                    update(state='waiting_actual_original_model_full_native_replay',seed=job['seed'],active_pid=None);time.sleep(10)
                attempt=root/'attempts'/job['id'];output=Path(config['artifact_root'])/job['id']
                if attempt.exists() or output.exists():raise FileExistsError('Preserve every full/partial label audit;no retry')
                with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                    verify(read(ROOT/CONFIG));attempt.mkdir(parents=True,exist_ok=False)
                    command=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--worker','--seed',str(job['seed'])]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='-1'),
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                        write(attempt/'launch.json',handle);update(state='auditing_full_saved_scores_labels_and_protocol',seed=job['seed'],active_pid=child.pid)
                        code,usage=capped_wait(child,config,api,update)
                    proof=output/'full_label_protocol_audit.json'
                    passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed:raise ValueError('Actual label/protocol audit failed;preserve source and partial artifacts')
                completed.append(job['id']);update(state='between_full_model_label_audits',active_pid=None);time.sleep(2)
            except Exception as error:failed[job['id']]=str(error);update(state='failed_diagnostic_preserved',active_pid=None)
        update(state='all5audit_attempts_terminal_other_paper_experiments_remain',active_pid=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,required_models=5)));return
    config=read(ROOT/CONFIG);jobs=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,models=5,entities_per_model=51,label_policies=4,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,model_inference_performed=False)))
    elif args.worker:
        job=next(j for j in jobs if j['seed']==args.seed);worker(config,job,Path(config['artifact_root'])/job['id'])
    else:controller(config,jobs)


if __name__=='__main__':main()
