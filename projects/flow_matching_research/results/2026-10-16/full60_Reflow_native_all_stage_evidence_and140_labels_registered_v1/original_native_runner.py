"""Reload all60 fullraw Reflow checkpoints and review every label view.

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
MODULE='scripts.flow_matching.replay_original_full60_reflow_v1'
CONFIG='configs/reproducibility/fm_original_full60_reflow_checkpoint_replays.2026-10-16_v1.json'
TEST='tests/test_fm_original_full60_reflow_replays_v1.py'
POLICY='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
MATH='configs/runtime/fm_objective_full440_checkpoint_replay.v1.json'
REPORTS=dict(native='native_replay.json',review='all_label_view_independent_review.json')


def registration():
    old=read(ROOT/scalar.CONFIG);scalar.verify(old)
    cases=[c for c in old['cases'] if c['family']=='reflow']
    guard,formula=read(ROOT/POLICY),read(ROOT/MATH)
    expected={(d,s) for d in ['smap','msl','smd'] for s in range(1103,1108)}
    if (len(cases)!=60 or len({c['id'] for c in cases})!=60
            or {(c['dataset'],c['seed']) for c in cases}!=expected
            or sum(len(c['required_label_policies']) for c in cases)!=140):
        raise ValueError('All60original Reflow and epoch-control cases and140label views required')
    files={scalar.CONFIG,POLICY,MATH,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
        'scripts/flow_matching/objective_replay_math_v1.py',
        'src/iia_benchmark/models/fm_objective_detectors.py','src/iia_benchmark/models/fm_reflow.py',
        'scripts/flow_matching/reflow_checkpoint_evidence_v1.py',
        'scripts/flow_matching/replay_original_full15_tranad_v1.py',
        'tests/test_fm_reflow_checkpoint_evidence_v1.py'}
    files.update(p['path'] for p in old['source_receipts'])
    return dict(schema_version=1,upstream_scalar_configuration=scalar.CONFIG,source_execution=old['source_execution'],cases=cases,
        required_family_native_cases=60,required_family_label_views=140,original_all_family_cases=873,original_all_label_views=2037,
        other_family_cases=813,other_family_components_complete=False,
        separate_TranAD_native_component='configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json',
        settings=guard['settings'],resource_lock=guard['resource_lock'],controller_python=old['controller_python'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        expected_source_model_entrypoints=['iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector',
            'iia_benchmark.models.fm_reflow:ReflowCFMDetector'],
        model_scoring='unchanged_source_score_with_private_checked_probability_path_and_velocity_forward_hook',
        native_comparison='exact_dtype_shape_content_bytes',math_tolerance=formula['math_tolerance'],
        probability_path_pass_criterion=formula['probability_path_pass_criterion'],
        ideal_float64_path_audit=formula['ideal_float64_path_audit'],
        paper_and_data_citations=old['paper_and_data_citations'],original_scope=old['original_scope'],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full60_reflow_checkpoint_replays_v1',
        state_root='experiments/runs/fm_original_full60_reflow_checkpoint_replays_v1',
        boundary='All60original fullraw Reflow2/3stage and independent40/60epoch controls,140label views. '
            'Epoch-matched architecture controls are not compute-matched:Reflow additionally generates teacher ODE pairs. '
            'Restore every final weight,all prior stage teacher weights,all ordered endpoint pairs and canonical stage records. '
            'Regenerate every source seeded origin and reintegrate every pair with exact source50stepHeun. '
            'An independent FP32 state recurrence checks every ODE forward input,update and discrete action. '
            'Unchanged Torch teacher kernels are reused;optimizer/warm-start trajectories are not independently replayed. '
            'No fitting/retraining/retry or source edits. Source2CPUthreads,20initial+20each stage,hidden128,batch64, '
            'all4MC*5scoretimes,100window,tails and original RNG remain. Complete native arrays must match bytewise. '
            'Use original2e-6residual tolerances and retain idealFP64path failures as diagnostics. '
            'Second fresh review reproduces all140label metrics/events/bootstrap200/grid controls. '
            'Keep full873cases/2037views,45papers/681methods/273data tasks/702targets. '
            'Other813native cases,fullTAB and original image/one-step distillation experiments remain separate. '
            'Keep sourceCPU16/32/16GiBheadroom,10/12GiBcaps and strictCPUlock.')


def verify(config):
    if config!=registration():raise ValueError('All60native cases/140views and unchanged complete numerical/resource policies required')
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
    p=model['parameters'];staged=model['entrypoint']=='iia_benchmark.models.fm_reflow:ReflowCFMDetector'
    stages=p.get('reflow_stages',1);total=p['epochs']+(stages-1)*p.get('reflow_epochs',0)
    if (case['family']!='reflow' or model['entrypoint'] not in config['expected_source_model_entrypoints']
            or p['epochs']!=(20 if staged else total) or total not in [40,60]
            or job['total_training_epochs']!=total or job['additional_pair_generation_compute']!=staged
            or p['window']!=100 or p['hidden_size']!=128 or p['batch_size']!=64
            or p['monte_carlo_samples']!=4 or p['score_times']!=[.1,.3,.5,.7,.9] or p['learning_rate']!=.001
            or case['window']!=100 or p['objective']!='rectified' or p['sigma']!=0 or p['score_weight']!=1.
            or job['parameter_overrides']!=dict(seed=case['seed'],device='cpu') or queue['evaluation']['cpu_threads']!=2
            or (staged and (stages not in [2,3] or p['reflow_epochs']!=20 or p['reflow_ode_steps']!=50
                            or p['reflow_solver']!='heun' or p.get('reflow_time_epsilon',1e-3)!=1e-3))):
        raise ValueError('All original full40/60epoch Reflow/controls,50Heun,capacity,MC and2CPUthreads required')


def validate_training_result(result,job):
    losses=result['training_losses']
    if (not isinstance(losses,list) or len(losses)!=job['total_training_epochs']
            or not all(math.isfinite(x) and x>=0 for x in losses) or result['parameter_count']<=0):
        raise ValueError('Actual complete finite40/60epoch source history and positive capacity required')


def validate_observation(coverage,observation):
    windows=sum(r['windows'] for r in coverage);calls=sum(math.ceil(r['windows']/64)*20 for r in coverage)
    if (observation['windows']!=windows or observation['window_points']!=windows*100
            or observation['path_calls']!=calls or observation['forward_calls']!=calls
            or observation['score_calls']!=len(coverage) or not observation['source_FP32_path_program_checked']):
        raise ValueError('Every full window,batch,MCsample,time and checked probability-path forward required')
    counts=observation['float64_path_failure_coordinate_counts']
    if (set(counts)!=set(['state','velocity','std']) or any(not isinstance(v,int) or v<0 for v in counts.values())
            or observation['ideal_float64_path_allclose_passed']!=all(v==0 for v in counts.values())):
        raise ValueError('All ideal FP64 mismatch counts must remain visible and consistent')


def producer_environment(result):
    import torch
    from scripts.flow_matching.run_objective_full440_checkpoint_replay_v1 import producer_environment as original
    actual=original(result);torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU objective replay must not initialize CUDA')
    return actual


def validate_coupling_summary(proof):
    total=proof['source_total_training_epochs'];count=proof['training_windows'];dimensions=proof['source_window_dimensions']
    evidence=proof['all_stage_coupling_evidence'];stages=proof['source_reflow_stages']
    if (total not in [40,60] or count<1 or dimensions<100 or dimensions%100
            or stages not in [1,2,3] or not evidence['all_source_evidence_required']
            or evidence['independent_epoch_control']!=(stages==1)
            or len(evidence['stages'])!=stages-1 or (stages>1 and total!=20*stages)):
        raise ValueError('Complete source stage and window geometry required')
    for stage,record in enumerate(evidence['stages'],2):
        if (record['stage']!=stage or record['pairs']!=count or record['dimensions']!=dimensions
                or record['pair_seed']!=proof['case']['seed']+1000*stage
                or record['solver']!='heun' or record['steps']!=50
                or record['teacher_forward_calls']!=math.ceil(count/64)*100
                or record['independent_state_program_checked_coordinates']!=count*dimensions*100
                or not record['all_origin_coordinates_regenerated_exactly']
                or not record['all_teacher_endpoint_coordinates_reinferred_exactly']
                or not record['all_source_teacher_geometry_exact']
                or record['neural_teacher_kernel_independently_rewritten']
                or len(record['pair_sha256'])!=64):
            raise ValueError('All source pairs,50Heun steps and unchanged teachers must be checked')


def check_restored_weights(detector,checkpoint):
    import torch
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    expected=checkpoint['modules'];modules={name:value for name,value in vars(detector).items() if isinstance(value,torch.nn.Module)}
    if set(modules)!=set(expected):raise ValueError('Every complete original module and stage evidence required')
    digests={}
    for module,observed in modules.items():
        if set(observed.state_dict())!=set(expected[module]):raise ValueError('Every original tensor key required')
        for key,value in observed.state_dict().items():
            if not torch.isfinite(value).all():raise ValueError('Finite original buffers required')
            tensor_bytes(value,expected[module][key]);digests[module+'.'+key]=array_digest(value.detach().cpu().numpy())
    return digests


def native_worker(config,case):
    import numpy as np,torch
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import restore_model,verify_all_couplings
    from scripts.flow_matching.objective_replay_math_v1 import checked_score,TOLERANCE
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    old=ready_scalar(config,case)
    if old is None:raise ValueError('Actual complete labels/scalar audit required before native FM inference')
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result)
    model_config=read(ROOT/job['model_config']);validate_model(config,case,queue,job,model_config);validate_training_result(result,job)
    if config['math_tolerance']!=TOLERANCE:raise ValueError('Original full probability-path numerical tolerance required')
    _,output=paths(config,case,'native');output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(ROOT/job['output_directory']/'model.pt',map_location='cpu',weights_only=True)
    if checkpoint['adjacency'] is not None:raise ValueError('Original full objective checkpoint schema required')
    features={e['features'] for e in data['entities']}
    if len(features)!=1:raise ValueError('Full original source feature geometry required')
    restored=restore_model(checkpoint,job,model_config,features.pop())
    if sum(p.numel() for p in restored._model.parameters())!=result['parameter_count']:
        raise ValueError('Complete trained velocity and score-head capacity must match')
    before=check_restored_weights(restored,checkpoint)
    def forbid_fit(*args,**kwargs):raise RuntimeError('No model fitting allowed during native FM replay')
    restored.fit=forbid_fit
    score,observation=checked_score(restored);payload={};coverage=[]
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        training_windows=sum(len(covering_windows(inputs[e['array_prefix']+'_train'],100)[0]) for e in data['entities'])
        coupling=verify_all_couplings(restored,job,result,training_windows)
        for entity in data['entities']:
            prefix=entity['array_prefix']
            for role in ['train','validation','test']:
                key=prefix+'_'+role;values=inputs[key]
                if values.ndim!=2 or values.shape[1]!=entity['features'] or not np.isfinite(values).all():
                    raise ValueError('Complete original finite FM model input required')
                windows,starts=covering_windows(values,100)
                observed=align_scores(score(windows),starts,len(values),100)
                require_identical(observed,saved[key]);payload[key]=observed
                coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values),windows=len(windows)))
            key=prefix+'_labels';require_identical(inputs[key],saved[key]);payload[key]=inputs[key]
            print(json.dumps(dict(id=case['id'],stage='full_entity_FM_checkpoint_reinferred',entity=entity['entity_id'])),flush=True)
        if set(payload)!=set(saved.files):raise ValueError('Every original score/label array required')
    validate_observation(coverage,observation)
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
        full_MC_probability_path_and_score_formula_observation=observation,math_tolerance=config['math_tolerance'],
        ideal_float64_path_equivalence_certified=observation['ideal_float64_path_allclose_passed'],
        source_FP32_operation_program_verified=True,scores_path=str(score_path),scores_sha256=sha(score_path),
        additional_fit_calls=0,additional_optimizer_updates=0,CUDA_context_initialized=torch.cuda.is_initialized(),
        original_full_epoch_loss_history_preserved=True,training_windows=training_windows,all_stage_coupling_evidence=coupling,
        source_reflow_stages=getattr(restored,'reflow_stages',1),source_total_training_epochs=job['total_training_epochs'],
        source_window_dimensions=math.prod(restored._shape),original_training_update_counter_independently_observed=False,
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
    validate_observation(proof['full_native_coverage'],proof['full_MC_probability_path_and_score_formula_observation'])
    validate_coupling_summary(proof)
    if (proof['math_tolerance']!=config['math_tolerance'] or not proof['source_FP32_operation_program_verified']
            or proof['ideal_float64_path_equivalence_certified']!=proof['full_MC_probability_path_and_score_formula_observation']['ideal_float64_path_allclose_passed']):
        raise ValueError('Original FP32 checks and explicit ideal-FP64 diagnostics required')
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
        source_FP32_operation_program_verified=native['source_FP32_operation_program_verified'],
        ideal_float64_path_equivalence_certified=native['ideal_float64_path_equivalence_certified'],
        full_TAB_or_author_equivalence=False,all_original_experiments_complete=False))


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full60Reflow native replay owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior complete/partial replay attempt')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_family_native_cases=60,required_family_label_views=140,active_pid=None,all_original_experiments_complete=False)
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
            write(attempt/'launch.json',handle);update(state='running_full_Reflow_'+phase,active_pid=child.pid,active_job=case['id'])
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
                    completed.append(identity);update(state='between_full_Reflow_replays',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='replay_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_original_training_and_scalar_audits',active_pid=None,active_job=None);time.sleep(10)
        update(state='all60family_replay_outcomes_terminal_other813native_cases_and_papers_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');parser.add_argument('--stage',choices=list(REPORTS));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=60,label_views=140)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=60,label_views=140,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.job_id:
        if args.stage is None:raise ValueError('An explicit native/review stage is required')
        case=next(c for c in cases if c['id']==args.job_id)
        (native_worker if args.stage=='native' else review_worker)(config,case)
    else:controller(config,cases)


if __name__=='__main__':main()
