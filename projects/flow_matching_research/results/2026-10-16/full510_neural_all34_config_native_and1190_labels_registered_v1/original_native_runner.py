"""Reload all510 fullraw neural checkpoints and review every label view.

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
MODULE='scripts.flow_matching.replay_original_full510_neural_v1'
CONFIG='configs/reproducibility/fm_original_full510_neural_checkpoint_replays.2026-10-16_v1.json'
TEST='tests/test_fm_original_full510_neural_replays_v1.py'
POLICY='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
GPU_POLICY='configs/runtime/fm_tsad_constant_fit_gpu_native_resource.v2.json'
MATH='configs/runtime/fm_objective_full440_checkpoint_replay.v1.json'
REPORTS=dict(native='native_replay.json',review='all_label_view_independent_review.json')


def registration():
    old=read(ROOT/scalar.CONFIG);scalar.verify(old)
    cases=[case for case in old['cases'] if case['family']=='neural'];cpu,gpu=read(ROOT/POLICY),read(ROOT/GPU_POLICY)
    if (len(cases)!=510 or len({case['id'] for case in cases})!=510
            or sum(len(case['required_label_policies']) for case in cases)!=1190
            or {(case['dataset'],case['seed']) for case in cases}!={(d,s) for d in ['smap','msl','smd'] for s in range(1103,1108)}):
        raise ValueError('Every510original neural and ablation case and1190label views required')
    models={};queues={}
    for case in cases:
        queue,job=scalar.resolve_case(case);models[job['model_config']]=read(ROOT/job['model_config']);queues[case['queue_path']]=queue
    if len(models)!=34 or len(queues)!=1:raise ValueError('All34original neural configurations and source queue required')
    queue=next(iter(queues.values()))
    if queue['lane']!='gpu' or queue['evaluation']['cpu_threads']!=4:raise ValueError('Original CUDA lane and4numerical CPU threads required')
    entries={model['entrypoint'] for model in models.values()}
    if entries!={'iia_benchmark.models.fm_comparator_reconstructions:ComparatorWindowDetector',
        'iia_benchmark.models.fm_native:GRASPDetector','iia_benchmark.models.fm_native:DFMDetector'}:raise ValueError('Every comparator,GRASPandDFMtype required')
    files={scalar.CONFIG,POLICY,GPU_POLICY,MATH,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/neural_full510_checkpoint_state_v1.py','scripts/flow_matching/comparator_checkpoint_score_math_v1.py',
        'scripts/flow_matching/dfm_checkpoint_density_math_v1.py','scripts/flow_matching/grasp_native_checkpoint_math_v1.py',
        'tests/test_fm_neural_full510_checkpoint_math_v1.py','scripts/flow_matching/reflow_checkpoint_evidence_v1.py',
        'scripts/flow_matching/phase_gate.py','scripts/flow_matching/run_tsad_native_gpu_resource_queue_v1.py',
        'scripts/flow_matching/run_spectral_original_queue.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'src/iia_benchmark/models/fm_comparator_reconstructions.py','src/iia_benchmark/models/fm_native.py'}
    files.update(models);files.update(pin['path'] for pin in old['source_receipts'])
    return dict(schema_version=1,upstream_scalar_configuration=scalar.CONFIG,source_execution=old['source_execution'],cases=cases,
        required_family_native_cases=510,required_family_label_views=1190,original_all_family_cases=873,original_all_label_views=2037,
        other_family_cases=363,other_family_components_complete=False,source_model_configurations=models,
        settings=cpu['settings'],resource_lock=cpu['resource_lock'],GPU_settings=gpu['settings'],GPU_resource_lock=gpu['resource_lock'],
        gpu_prerequisite_queues=queue['gpu_prerequisite_queues'],legacy_gpu_queue_lock='experiments/runs/flow_matching_campaign/queue.lock',
        source_cpu_threads=4,expected_source_device='cuda:0',controller_python=old['controller_python'],
        source_receipts=[dict(path=path,sha256=sha(ROOT/path)) for path in sorted(files)],
        expected_source_model_entrypoints=sorted(entries),model_scoring='unchanged_source_native_CUDA_same_device_with_private_readonly_mathematical_observer',
        native_comparison='exact_dtype_shape_content_bytes',math_tolerance=read(ROOT/MATH)['math_tolerance'],
        paper_and_data_citations=old['paper_and_data_citations'],original_scope=old['original_scope'],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full510_neural_checkpoint_replays_v1',
        state_root='experiments/runs/fm_original_full510_neural_checkpoint_replays_v1',
        boundary='All510original CUDA neural cases:420comparators(28configs),60GRASP(4variants),30DFM(2traces),1190label views. '
            'Preserve all capacity,epochs,losses,graphs,MCsamples,times,ODEsteps,seeds,batches,windows and CUDA:0device. '
            'Restore every original module,buffer,and actual graph eigenbasis;preserve source requires_grad flags. '
            'No fitting,retraining,retry,source edits or CPUsubstitution for native CUDA replay. '
            'Independent FP64 primitives/source-precision score programs retain2e-6tolerances;idealFP64 '
            'comparator differences remain diagnostics. DFM verifies analytic TimeMLP divergence and every Euler/autograd '
            'trace call;not a separate idealFP64ODEtrajectory. GRASPchecks full train-only graph,all5sources*10times '
            'and independent spectral residuals. No independent general neural kernel or optimizer-trajectory claim. '
            'Every native array must match source dtype/shape/content bytes and serialized reload. '
            'Fresh CPUreview recovers all labels/metrics/events/bootstrap200/grid controls. Source failures remain negative outcomes. '
            'NativeGPUkeeps original18/32/16GiBhost,14/4GiBGPU,UUID,35second yield,phasegatesandlegacy/shared mutexes. '
            'CPUreview keeps original16/32/16GiBheadroom,10/12GiBcaps and strictCPUlock. '
            'TinyCPUunit fixtures cannot certify CUDA execution. Local adaptations are not author/image/fullTABequivalence. '
            'Keep all873/2037and original45papers/681methods/273data tasks/702targets;paired seed statistics and original experiments remain.')


def verify(config):
    if config!=registration():raise ValueError('All510CUDAcases/1190views and every original resource/fidelity gate required')
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
    expected=config['source_model_configurations'].get(job['model_config'])
    if (model!=expected or case['family']!='neural' or model['entrypoint'] not in config['expected_source_model_entrypoints']
            or job['window']!=case['window'] or case['window']!=model['parameters'].get('window',100)
            or job['parameter_overrides']!=dict(seed=case['seed'],device=config['expected_source_device'])
            or queue['evaluation']['cpu_threads']!=config['source_cpu_threads']):
        raise ValueError('Every original CUDA model/ablation,budget,device,window,seed and4CPUthreads required')


def validate_training_result(result,model):
    losses=result['training_losses'];p=model['parameters']
    comparator=model['entrypoint'].endswith(':ComparatorWindowDetector')
    if (result['parameter_count']<=0 or (comparator and losses is not None)
            or (not comparator and (not isinstance(losses,list) or len(losses)!=p['epochs'] or not all(math.isfinite(x) for x in losses)))):
        raise ValueError('Full original persisted GRASP/DFMepoch histories,or explicit unpersisted comparator counters,required')


def validate_observation(coverage,obs,model):
    p=model['parameters'];window=p.get('window',100);batch=p['batch_size'];windows=sum(row['windows'] for row in coverage)
    batches=sum(math.ceil(row['windows']/batch) for row in coverage);features={row['features'] for row in coverage}
    if len(features)!=1:raise ValueError('Every original feature coordinate required')
    f=next(iter(features))
    if obs['score_calls']!=len(coverage) or obs['windows']!=windows or obs['window_points']!=windows*window:
        raise ValueError('Every source role,window and tail must be replayed')
    entry=model['entrypoint']
    if entry.endswith(':ComparatorWindowDetector'):
        opts=p['options'];branches=len(opts.get('branches',[]));shcl=p['method']=='shcl_transformer';reconstruction=opts.get('objective')=='reconstruction'
        if (obs['method']!=p['method'] or obs['source_core_score_calls']!=batches or obs['primitive_checks']<1
                or obs['primitive_coordinates']<windows*window or not obs['independent_FP64_primitives_and_source_precision_program_checked']
                or obs['independent_neural_kernel_implementation'] or obs['masked_representation_calls']!=(2*branches*batches if shcl else 0)
                or obs['gaussian_calls']!=(branches*batches if shcl and not reconstruction else 0)
                or obs['reconstruction_decoder_calls']!=(2*branches*batches if shcl and reconstruction else 0)
                or obs['ideal_FP64_failure_coordinates']<0
                or obs['ideal_FP64_complete_score_equivalence_certified']!=(obs['ideal_FP64_failure_coordinates']==0)):
            raise ValueError('Every comparator branch,mask,decoder,primitive and explicit idealFP64diagnostic required')
    elif entry.endswith(':DFMDetector'):
        steps=p['steps'];d=window*f;exact=p['trace']=='exact'
        if (obs['method']!='dfm_'+p['trace'] or obs['CNF_batches']!=batches or obs['velocity_forwards']!=batches*steps
                or obs['autograd_calls']!=batches*steps*(d if exact else 1) or obs['trace_coordinates_checked']!=windows*steps*d
                or obs['source_Rademacher_coordinates_checked']!=(0 if exact else windows*d)
                or not obs['independent_FP64_MLP_and_divergence_checked'] or not obs['source_FP32_Euler_and_correction_program_checked']
                or obs['full_ideal_FP64_ODE_trajectory_replayed']):raise ValueError('Every DFMstate,trace coordinate,source RNG and complete Euler program required')
    else:
        samples=p['source_samples'];times=p['flow_evaluations'];evaluations=windows*samples*times
        if (obs['method']!='grasp_'+p['variant'] or obs['path_forwards']!=batches*samples*times or obs['velocity_forwards']!=batches*samples*times
                or obs['source_origin_coordinates_regenerated']!=len(coverage)*samples*f*window
                or obs['independent_path_coordinates_checked']!=2*evaluations*f*window
                or obs['independent_score_residual_coordinates_checked']!=evaluations*f*window
                or not obs['independent_FP64_path_and_score_checked'] or obs['independent_neural_kernel_implementation']):
            raise ValueError('Every GRASPsource,time,path,velocity and full spectral residual required')


def producer_environment(result,config):
    import torch,numpy as np
    actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=config['expected_source_device'])
    if actual!=result['environment']:raise ValueError('Exact original CUDA producer environment required')
    torch.set_num_threads(config['source_cpu_threads']);return actual


def check_restored_weights(detector,checkpoint):
    from scripts.flow_matching.neural_full510_checkpoint_state_v1 import check_state
    return check_state(detector,checkpoint)


def native_worker(config,case):
    import numpy as np,torch
    from scripts.flow_matching.neural_full510_checkpoint_state_v1 import restore_model,load_checkpoint
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import FORMULA_TOLERANCE as TOLERANCE
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    old=ready_scalar(config,case)
    if old is None:raise ValueError('Actual complete labels/scalar audit required before native baseline inference')
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result,config)
    model_config=read(ROOT/job['model_config']);validate_model(config,case,queue,job,model_config);validate_training_result(result,model_config)
    if config['math_tolerance']!=TOLERANCE:raise ValueError('Original full independent score numerical tolerance required')
    _,output=paths(config,case,'native');output.mkdir(parents=True,exist_ok=False)
    checkpoint=load_checkpoint(ROOT/job['output_directory']/'model.pt')
    features={e['features'] for e in data['entities']}
    if len(features)!=1:raise ValueError('Full original source feature geometry required')
    restored=restore_model(checkpoint,job,model_config,features.pop())
    if sum(p.numel() for module in vars(restored).values() if isinstance(module,torch.nn.Module) for p in module.parameters())!=result['parameter_count']:
        raise ValueError('Complete original baseline capacity must match')
    before=check_restored_weights(restored,checkpoint)
    def forbid_fit(*args,**kwargs):raise RuntimeError('No model fitting allowed during native baseline replay')
    restored.fit=forbid_fit
    entry=model_config['entrypoint']
    if entry.endswith(':ComparatorWindowDetector'):
        from scripts.flow_matching.comparator_checkpoint_score_math_v1 import checked_score
    elif entry.endswith(':DFMDetector'):
        from scripts.flow_matching.dfm_checkpoint_density_math_v1 import checked_score
    else:
        from scripts.flow_matching.grasp_native_checkpoint_math_v1 import checked_score
    score,observation=checked_score(restored);payload={};coverage=[];graph=None
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        if entry.endswith(':GRASPDetector'):
            from scripts.flow_matching.grasp_native_checkpoint_math_v1 import verify_training_graph
            training=np.concatenate([covering_windows(inputs[e['array_prefix']+'_train'],job['window'])[0] for e in data['entities']])
            graph=verify_training_graph(restored,training);del training
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
            print(json.dumps(dict(id=case['id'],stage='full_entity_native_CUDA_checkpoint_reinferred',entity=entity['entity_id'])),flush=True)
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
        GRASP_full_training_graph_evidence=graph,original_cuda_device_retained=config['expected_source_device'],
        source_model_configuration=model_config,scores_path=str(score_path),scores_sha256=sha(score_path),
        additional_fit_calls=0,additional_optimizer_updates=0,CUDA_context_initialized=torch.cuda.is_initialized(),
        source_full_epoch_loss_history_persisted=result['training_losses'] is not None,original_training_update_counter_independently_observed=False,
        full_TAB_or_historical_author_equivalence=False,all_original_experiments_complete=False))


def ready_native(config,case):
    attempt,output=paths(config,case,'native');path=output/REPORTS['native'];guard=attempt/'guard_receipt.json'
    if not path.exists() or not guard.exists():raise ValueError('Actual complete model reinference and successful native guard required')
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or not proof['all_array_dtype_shape_content_bytes_identical'] or not proof['all_model_tensors_identical_before_and_after']
            or not proof['every_saved_native_array_serialized_and_reloaded'] or proof['additional_fit_calls'] or proof['additional_optimizer_updates']
            or proof['CUDA_context_initialized']!=config['expected_source_device'].startswith('cuda') or sha(proof['scores_path'])!=proof['scores_sha256']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(case,'native')):
        raise ValueError('Complete unchanged native model proof and guard required')
    validate_observation(proof['full_native_coverage'],proof['all_source_forward_and_FP64_formula_observation'],proof['source_model_configuration'])
    if proof['math_tolerance']!=config['math_tolerance'] or not proof['independent_FP64_scoring_formula_verified']:
        raise ValueError('Every unchanged primitive tolerance and complete score program required')
    model=proof['source_model_configuration']
    if model['entrypoint'].endswith(':GRASPDetector'):
        graph=proof['GRASP_full_training_graph_evidence']
        if not graph or not graph['every_training_coordinate_used'] or not graph['actual_saved_eigenbasis_retained']:
            raise ValueError('Original actual train-only graph and saved eigenbasis evidence required')
    return path,proof


def review_worker(config,case):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical
    old=ready_scalar(config,case);native_path,native=ready_native(config,case)
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result,config)
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
        full_source_scoring_observation=native['all_source_forward_and_FP64_formula_observation'],
        full_TAB_or_author_equivalence=False,all_original_experiments_complete=False))


def phase_readiness(config):
    import psutil
    from scripts.flow_matching.phase_gate import check_queue
    pending={path:len(check_queue(ROOT/path)) for path in config['gpu_prerequisite_queues']}
    active=[p.info['pid'] for p in psutil.process_iter(['pid','cmdline']) if any('run_queue.py' in arg for arg in p.info['cmdline'] or [])]
    return dict(pending_prerequisites=pending,live_prior_queue_pids=active,ready=not any(pending.values()) and not active)


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    from scripts.flow_matching.run_tsad_native_gpu_resource_queue_v1 import native_slot
    from scripts.flow_matching.run_spectral_original_queue import guarded_api
    cpu=install_facades();gpu=guarded_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=cpu['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('All510neural replay owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve earlier native/review attempts')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_family_native_cases=510,required_family_label_views=1190,active_pid=None,all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw,completed_cases=list(completed),failed_replays_preserved=dict(failed),
            source_failures_preserved=dict(source_failed),captured_utc=datetime.now(timezone.utc).isoformat());atomic(base/'status.json',state)
    def stage(case,phase):
        attempt,output=paths(config,case,phase)
        if attempt.exists() or output.exists():raise FileExistsError('Preserve every full source/native/review attempt without retry')
        attempt.mkdir(parents=True,exist_ok=False);command=worker_command(case,phase)
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
            CUDA_VISIBLE_DEVICES='0' if phase=='native' else '-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=environment,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_full_neural_'+phase,active_pid=child.pid,active_job=case['id'])
            if phase=='native':code,usage=gpu['guarded_gpu_wait'](child,config['GPU_settings'],gpu,update)
            else:code,usage=capped_wait(child,config,cpu,update)
        proof=output/REPORTS[phase];ok=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
        if not ok:raise ValueError('Complete neural native/review failed;preserve without retry')
    def native(case):
        while True:
            readiness=phase_readiness(config)
            if not readiness['ready']:
                update(state='waiting_original_imputation_phase_gates',**readiness);time.sleep(10);continue
            legacy=gpu['exclusive_lock'](ROOT/config['legacy_gpu_queue_lock'])
            if legacy is None:
                update(state='waiting_original_legacy_GPU_queue_mutex',**readiness);time.sleep(10);continue
            try:
                registration=dict(settings=config['GPU_settings'],resource_lock=config['GPU_resource_lock'])
                with native_slot(gpu,registration,update):
                    verify(read(ROOT/CONFIG));fresh=phase_readiness(config)
                    if not fresh['ready']:continue
                    stage(case,'native')
                time.sleep(config['GPU_settings']['controller_yield_seconds']);return
            finally:legacy.close()
    try:
        while len(completed)+len(failed)+len(source_failed)<len(cases):
            for case in cases:
                identity=case['id']
                if identity in completed or identity in failed or identity in source_failed:continue
                try:
                    original=scalar.source_status(case)
                    if original is not None and original['kind']=='source_failed_preserved':source_failed[identity]=original;continue
                    if ready_scalar(config,case) is None:continue
                    native(case)
                    with cpu['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                        verify(read(ROOT/CONFIG));stage(case,'review')
                    completed.append(identity);update(state='between_full_neural_replays',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='replay_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_original_CUDA_training_and_scalar_audits',active_pid=None,active_job=None);time.sleep(10)
        update(state='all510family_outcomes_terminal_all_other_original_experiments_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');parser.add_argument('--stage',choices=list(REPORTS));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=510,label_views=1190)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=510,label_views=1190,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.job_id:
        if args.stage is None:raise ValueError('An explicit native/review stage is required')
        case=next(c for c in cases if c['id']==args.job_id)
        (native_worker if args.stage=='native' else review_worker)(config,case)
    else:controller(config,cases)


if __name__=='__main__':main()
