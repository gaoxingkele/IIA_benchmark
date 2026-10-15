"""Reload all105 fullraw baseline checkpoints and review every label view.

This is one native-replay component of the full873 experiment scope. Other
families, original papers, source failures and full TAB remain separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as scalar

ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write
MODULE='scripts.flow_matching.replay_original_full105_baselines_v1'
CONFIG='configs/reproducibility/fm_original_full105_baselines_checkpoint_replays.2026-10-16_v1.json'
TEST='tests/test_fm_original_full105_baseline_replays_v1.py'
POLICY='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
MATH='configs/runtime/fm_objective_full440_checkpoint_replay.v1.json'
REPORTS=dict(native='native_replay.json',review='all_label_view_independent_review.json')


def registration():
    old=read(ROOT/scalar.CONFIG);scalar.verify(old)
    cases=[case for case in old['cases'] if case['family']=='baselines'];guard=read(ROOT/POLICY)
    if (len(cases)!=105 or len({case['id'] for case in cases})!=105
            or sum(len(case['required_label_policies']) for case in cases)!=245
            or {(case['dataset'],case['seed']) for case in cases}!={(d,s) for d in ['smap','msl','smd'] for s in range(1103,1108)}):
        raise ValueError('All105original fullraw baseline cases and245label views required')
    models={}
    for case in cases:
        queue,job=scalar.resolve_case(case);path=job['model_config'];models[path]=read(ROOT/path)
    variants={(model['parameters']['method'],model['parameters']['parameters'].get('adversarial')) for model in models.values()}
    if len(models)!=21 or variants!={('anomaly_transformer',None),('dcdetector',None),('itransformer',None),('timesnet',None),('tranad',None),('usad',False),('usad',True)}:
        raise ValueError('All six methods including both separately named USAD loss variants required')
    files={scalar.CONFIG,POLICY,MATH,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/legacy_baseline_checkpoint_math_v1.py','tests/test_fm_legacy_baseline_checkpoint_math_v1.py',
        'scripts/flow_matching/reflow_checkpoint_evidence_v1.py',
        'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'src/iia_benchmark/models/fm_legacy_window_adapter.py','src/iia_benchmark/models/mtsad_detectors.py'}
    files.update(models);files.update(pin['path'] for pin in old['source_receipts'])
    return dict(schema_version=1,upstream_scalar_configuration=scalar.CONFIG,source_execution=old['source_execution'],cases=cases,
        required_family_native_cases=105,required_family_label_views=245,original_all_family_cases=873,original_all_label_views=2037,
        other_family_cases=768,other_family_components_complete=False,source_model_configurations=models,
        settings=guard['settings'],resource_lock=guard['resource_lock'],controller_python=old['controller_python'],
        source_receipts=[dict(path=path,sha256=sha(ROOT/path)) for path in sorted(files)],
        expected_source_model_entrypoint='iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter',
        model_scoring='unchanged_source_legacy_adapter_and_delegate_score_with_passive_forward_observer',
        native_comparison='exact_dtype_shape_content_bytes',math_tolerance=read(ROOT/MATH)['math_tolerance'],
        paper_and_data_citations=old['paper_and_data_citations'],original_scope=old['original_scope'],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full105_baselines_checkpoint_replays_v1',
        state_root='experiments/runs/fm_original_full105_baselines_checkpoint_replays_v1',
        boundary='All105original fullraw strong baseline cases,245label views. Keep all873/2037and full original scope. '
            'Preserve every21per-dataset model config,seed,epoch,width,patch,precision,loss sign,window and batch size. '
            'Restore all original modules/parameters/buffers,including all3USADcomponents. No fitting,retry,source edits. '
            'Native arrays require exact dtype/shape/content bytes plus serialized reload. Independent NumPyFP64 '
            'checks every forward score and each per-head KL/normalization/softmax primitive in FP64, '
            'source-FP32 temperature/layer energy,reconstruction,USADweighted losses or '
            'TranADlast-step MSE at original2e-6tolerance. Full idealFP64attention score mismatch counts/errors '
            'remain diagnostics,not certified end-to-end equivalence. Preserve source parameter requires_grad '
            'flags even in no_grad to retain exact CPU dispatch. SourceNNkernels remain;not independent neural implementation. '
            'Fresh process reviews all source/native arrays and all245label metrics/events/bootstrap200/grids. '
            'Sourcefailures including signedUSADinstability and any short-window incompatibility remain negative outcomes. '
            'Preserve sourceCPU2threads and strict16/32/16GiBheadroom,10/12GiBcaps,strictCPUlock. '
            'Legacy local transcriptions,USADnonadversarial and FP32TranAD/window-score broadcast are not '
            'historical author-harness/POT/fullTAB equivalence. Source training update counters are not persisted. '
            'Other768native cases,paired training-seed uncertainty and full original experiments remain separate.')


def verify(config):
    if config!=registration():raise ValueError('All105native cases/245views and original fidelity/resource contracts required')
    return config['cases']


def paths(config,case,phase):
    return (ROOT/config['state_root']/'attempts'/case['id']/phase,
        Path(config['artifact_root'])/case['id']/phase)


def worker_command(case,phase):
    if phase not in REPORTS:raise ValueError('Explicit complete native/review stage required')
    return [str(ROOT/case['worker_python']),'-B','-X','utf8','-u','-m',MODULE,'--job-id',case['id'],'--stage',phase]


def ready_scalar(config,case):
    original=read(ROOT/config['upstream_scalar_configuration'])
    path=Path(original['artifact_root'])/case['id']/scalar.REPORT
    guard=ROOT/original['state_root']/'attempts'/case['id']/'guard_receipt.json'
    if not path.exists() or not guard.exists():return None
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/config['upstream_scalar_configuration'])
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or not proof['source_metrics_exactly_recomputed'] or not proof['all_original_arrays_checked']
            or set(proof['label_view_results'])!=set(case['required_label_policies'])
            or not proof['all_validation_thresholds_and_predictions_unchanged']):
        raise ValueError('Actual full original labels/metrics audit and successful guard required')
    queue,job,result,data,binding=scalar.source_binding(config,case)
    if proof['source_binding']!=binding:raise ValueError('Full original scalar source binding changed')
    return dict(path=str(path),sha256=sha(path),guard_sha256=sha(guard),source_binding=binding)


def validate_model(config,case,queue,job,model):
    expected=config['source_model_configurations'].get(job['model_config']);p=model['parameters']
    if (model!=expected or case['family']!='baselines' or model['entrypoint']!=config['expected_source_model_entrypoint']
            or job['window']!=case['window'] or case['window']!=p['parameters']['window']
            or job['parameter_overrides']!=dict(seed=case['seed'],device='cpu') or queue['evaluation']['cpu_threads']!=2):
        raise ValueError('Exact original per-dataset capacity,budget,window,loss sign,batch and source2CPUthreads required')


def validate_training_result(result):
    if result['training_losses'] is not None or result['parameter_count']<=0:
        raise ValueError('Source legacy loss/update counters are unpersisted;never manufacture a training history')


def validate_observation(coverage,observation,model):
    p=model['parameters'];method=p['method'];window=p['parameters']['window']
    batch=512 if method=='usad' else p['parameters']['batch_size'];windows=sum(row['windows'] for row in coverage)
    calls=sum(math.ceil(row['windows']/batch) for row in coverage)*(3 if method=='usad' else 1)
    features={row['features'] for row in coverage}
    if len(features)!=1:raise ValueError('Original full feature geometry required')
    f=next(iter(features));rec=0 if method=='dcdetector' else windows*f*(1 if method=='tranad' else window)*(2 if method=='usad' else 1)
    layers=p['parameters'].get('e_layers',0)*(len(p['parameters']['patch_sizes']) if method=='dcdetector' else 1)
    attention=windows*p['parameters'].get('n_heads',0)*window*window*layers*2 if method in ['anomaly_transformer','dcdetector'] else 0
    KLcalls=sum(math.ceil(row['windows']/batch) for row in coverage)*2*layers if method in ['anomaly_transformer','dcdetector'] else 0
    if (observation['method']!=method or observation['source_score_batch_size']!=batch
            or observation['windows']!=windows or observation['window_points']!=windows*window
            or observation['forward_calls']!=calls or observation['score_calls']!=len(coverage)
            or observation['reconstruction_coordinates_checked']!=rec
            or observation['attention_probability_coordinates_checked']!=attention
            or observation['per_head_KL_calls_checked']!=KLcalls
            or observation['source_FP32_energy_program_checked']!=(method in ['anomaly_transformer','dcdetector'])
            or not observation['independent_FP64_score_program_checked'] or observation['independent_neural_kernel_implementation']):
        raise ValueError('Every original window,tail,batch,decoder,attention head/layer and score coordinate required')
    failures=observation['ideal_FP64_score_failure_coordinates']
    if (not isinstance(failures,int) or failures<0
            or observation['ideal_FP64_complete_score_equivalence_certified']!=(failures==0)):
        raise ValueError('All end-to-end idealFP64 mismatch diagnostics must remain explicit')


def producer_environment(result):
    import torch
    from scripts.flow_matching.run_objective_full440_checkpoint_replay_v1 import producer_environment as original
    actual=original(result);torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU baseline replay must not initialize CUDA')
    return actual


def check_restored_weights(detector,checkpoint):
    import torch
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    expected=checkpoint['modules']['_model'];actual=detector._model.state_dict()
    if set(actual)!=set(expected):raise ValueError('Every original complete module parameter and buffer required')
    for key,value in actual.items():
        if not torch.isfinite(value).all():raise ValueError('Finite complete original state required')
        tensor_bytes(value,expected[key])
    return {key:array_digest(value.detach().cpu().numpy()) for key,value in actual.items()}


def native_worker(config,case):
    import numpy as np,torch
    from scripts.flow_matching.legacy_baseline_checkpoint_math_v1 import restore_model,checked_score
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE as TOLERANCE
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    old=ready_scalar(config,case)
    if old is None:raise ValueError('Actual complete labels/scalar audit required before native baseline inference')
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result)
    model_config=read(ROOT/job['model_config']);validate_model(config,case,queue,job,model_config);validate_training_result(result)
    if config['math_tolerance']!=TOLERANCE:raise ValueError('Original full independent score numerical tolerance required')
    _,output=paths(config,case,'native');output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(ROOT/job['output_directory']/'model.pt',map_location='cpu',weights_only=True)
    if checkpoint['adjacency'] is not None:raise ValueError('Original full baseline checkpoint schema required')
    features={e['features'] for e in data['entities']}
    if len(features)!=1:raise ValueError('Full original source feature geometry required')
    restored=restore_model(checkpoint,job,model_config,features.pop())
    if sum(p.numel() for p in restored._model.parameters())!=result['parameter_count']:
        raise ValueError('Complete original baseline capacity must match')
    before=check_restored_weights(restored,checkpoint)
    def forbid_fit(*args,**kwargs):raise RuntimeError('No model fitting allowed during native baseline replay')
    restored.fit=forbid_fit;restored._delegate.fit=forbid_fit
    score,observation=checked_score(restored);payload={};coverage=[]
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        for entity in data['entities']:
            prefix=entity['array_prefix']
            for role in ['train','validation','test']:
                key=prefix+'_'+role;values=inputs[key]
                if values.ndim!=2 or values.shape[1]!=entity['features'] or not np.isfinite(values).all():
                    raise ValueError('Complete original finite baseline model input required')
                windows,starts=covering_windows(values,job['window'])
                observed=align_scores(score(windows),starts,len(values),job['window'])
                require_identical(observed,saved[key]);payload[key]=observed
                coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values),windows=len(windows),features=entity['features']))
            key=prefix+'_labels';require_identical(inputs[key],saved[key]);payload[key]=inputs[key]
            print(json.dumps(dict(id=case['id'],stage='full_entity_baseline_checkpoint_reinferred',entity=entity['entity_id'])),flush=True)
        if set(payload)!=set(saved.files):raise ValueError('Every original score/label array required')
    validate_observation(coverage,observation,model_config)
    after=check_restored_weights(restored,checkpoint)
    if after!=before:raise ValueError('Native scoring changed an original trained tensor')
    score_path=output/'scores.npz'
    with score_path.open('xb') as stream:np.savez_compressed(stream,**payload)
    with np.load(score_path,allow_pickle=False) as saved:
        if set(saved.files)!=set(payload):raise ValueError('Every native serialized array must reload')
        for key,value in payload.items():require_identical(saved[key],value)
    if scalar.source_binding(config,case)[-1]!=binding or ready_scalar(config,case)!=old:
        raise ValueError('Original data,weights,scores and full upstream evidence must stay unchanged')
    write(output/REPORTS['native'],dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        upstream_scalar_evidence=old,actual_producer_environment=environment,parameters=result['parameter_count'],
        complete_restored_state_tensors=before,all_model_tensors_identical_before_and_after=True,
        full_native_coverage=coverage,all_array_dtype_shape_content_bytes_identical=True,
        every_saved_native_array_serialized_and_reloaded=True,source_arrays={k:array_digest(v) for k,v in payload.items()},
        all_source_forward_and_FP64_formula_observation=observation,math_tolerance=config['math_tolerance'],
        independent_FP64_scoring_formula_verified=True,
        ideal_FP64_complete_score_equivalence_certified=observation['ideal_FP64_complete_score_equivalence_certified'],
        source_model_configuration=model_config,scores_path=str(score_path),scores_sha256=sha(score_path),
        additional_fit_calls=0,additional_optimizer_updates=0,CUDA_context_initialized=torch.cuda.is_initialized(),
        source_training_loss_or_update_counter_persisted=False,original_training_update_counter_independently_observed=False,
        full_TAB_or_historical_author_equivalence=False,all_original_experiments_complete=False))


def ready_native(config,case):
    attempt,output=paths(config,case,'native');path=output/REPORTS['native'];guard=attempt/'guard_receipt.json'
    if not path.exists() or not guard.exists():raise ValueError('Actual complete model reinference and successful native guard required')
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or not proof['all_array_dtype_shape_content_bytes_identical'] or not proof['all_model_tensors_identical_before_and_after']
            or not proof['every_saved_native_array_serialized_and_reloaded'] or proof['additional_fit_calls'] or proof['additional_optimizer_updates']
            or proof['CUDA_context_initialized'] or sha(proof['scores_path'])!=proof['scores_sha256']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(case,'native')):
        raise ValueError('Complete unchanged native model proof and guard required')
    validate_observation(proof['full_native_coverage'],proof['all_source_forward_and_FP64_formula_observation'],proof['source_model_configuration'])
    if (proof['math_tolerance']!=config['math_tolerance'] or not proof['independent_FP64_scoring_formula_verified']
            or proof['ideal_FP64_complete_score_equivalence_certified']!=proof['all_source_forward_and_FP64_formula_observation']['ideal_FP64_complete_score_equivalence_certified']):
        raise ValueError('Original complete independent FP64score checks required')
    return path,proof


def review_worker(config,case):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical
    old=ready_scalar(config,case);native_path,native=ready_native(config,case)
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result)
    if old is None or native['source_binding']!=binding or native['upstream_scalar_evidence']!=old:
        raise ValueError('Actual native output and all original scalar sources must align')
    with np.load(native['scores_path'],allow_pickle=False) as scores,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as original,np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs:
        if set(scores.files)!=set(original.files):raise ValueError('Every complete original/native output must be compared')
        for key in original.files:require_identical(scores[key],original[key])
        entities,validation,training,coverage,digests=scalar.collect_arrays(inputs,scores,data)
    previous=read(old['path'])
    if digests!=native['source_arrays'] or digests!=previous['source_array_digests']:
        raise ValueError('Every full native array must retain all original content digests')
    overlays={}
    for policy,label in data['labels_only_controls'].items():
        if sha(label['path'])!=label['sha256']:raise ValueError('Immutable raw test-label control changed')
        with np.load(label['path'],allow_pickle=False) as labels:overlays[policy]={k:labels[k] for k in labels.files}
    views=scalar.evaluate_views(entities,validation,training,overlays,data,result,queue['evaluation'],job['seed'])
    if views!=previous['label_view_results'] or coverage!=previous['full_positional_saved_score_coverage']:
        raise ValueError('Full native outputs must recover every original point/event/bootstrap/label/grid metric exactly')
    if scalar.source_binding(config,case)[-1]!=binding or ready_scalar(config,case)!=old:
        raise ValueError('Every original source and scalar proof must remain unchanged')
    _,output=paths(config,case,'review');output.mkdir(parents=True,exist_ok=False)
    write(output/REPORTS['review'],dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        native_proof_sha256=sha(native_path),all_original_and_native_arrays_independently_compared=True,
        all_original_labels_metrics_events_bootstrap_and_grid_recomputed=True,all_label_view_results=views,
        native_source_array_digests=digests,full_native_coverage=coverage,actual_source_environment=environment,
        conditional_entity_CI_not_training_seed_uncertainty=True,new_training_or_model_inference_in_review=False,
        independent_velocity_network_kernel_implementation=False,
        independent_FP64_scoring_formula_verified=native['independent_FP64_scoring_formula_verified'],
        ideal_FP64_complete_score_equivalence_certified=native['ideal_FP64_complete_score_equivalence_certified'],
        full_TAB_or_author_equivalence=False,all_original_experiments_complete=False))


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full105baseline native replay owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior complete/partial replay attempt')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_family_native_cases=105,required_family_label_views=245,active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_cases=list(completed),failed_replays_preserved=dict(failed),
            source_failures_preserved=dict(source_failed),captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    def stage(case,phase):
        attempt,output=paths(config,case,phase)
        if attempt.exists() or output.exists():raise FileExistsError('Preserve all prior native/review evidence without retry')
        attempt.mkdir(parents=True,exist_ok=False)
        command=worker_command(case,phase)
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_full_legacy_baseline_'+phase,active_pid=child.pid,active_job=case['id'])
            code,usage=capped_wait(child,config,api,update)
        proof=output/REPORTS[phase];ok=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
        if not ok:raise ValueError('Full FM native/review attempt failed;preserve without retry')
    try:
        while len(completed)+len(failed)+len(source_failed)<len(cases):
            for case in cases:
                identity=case['id']
                if identity in completed or identity in failed or identity in source_failed:continue
                try:
                    original=scalar.source_status(case)
                    if original is not None and original['kind']=='source_failed_preserved':source_failed[identity]=original;continue
                    if ready_scalar(config,case) is None:continue
                    with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                        verify(read(ROOT/CONFIG));stage(case,'native');stage(case,'review')
                    completed.append(identity);update(state='between_full_legacy_baseline_replays',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='replay_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_original_training_and_scalar_audits',active_pid=None,active_job=None);time.sleep(10)
        update(state='all105family_replay_outcomes_terminal_other768native_cases_and_papers_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');parser.add_argument('--stage',choices=list(REPORTS));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=105,label_views=245)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=105,label_views=245,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.job_id:
        if args.stage is None:raise ValueError('An explicit native/review stage is required')
        case=next(c for c in cases if c['id']==args.job_id)
        (native_worker if args.stage=='native' else review_worker)(config,case)
    else:controller(config,cases)


if __name__=='__main__':main()
