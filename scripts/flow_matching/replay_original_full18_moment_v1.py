"""Reload all18 fullraw official MOMENT cases and review all42 label views.

This is one native-replay component of the full873 experiment scope. Other
families, original papers, source failures and full TAB remain separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import hashlib
from types import FunctionType
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as scalar

ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write
MODULE='scripts.flow_matching.replay_original_full18_moment_v1'
CONFIG='configs/reproducibility/fm_original_full18_moment_checkpoint_replays.2026-10-16_v1.json'
TEST='tests/test_fm_original_full18_moment_replays_v1.py'
POLICY='configs/runtime/fm_moment_full78_checkpoint_replay.v1.json'
REPORTS=dict(native='native_case_verification.json',review='all_label_view_independent_review.json')


def registration():
    old=read(ROOT/scalar.CONFIG);scalar.verify(old);guard=read(ROOT/POLICY)
    cases=[c for c in old['cases'] if c['family']=='moment'];combinations=set()
    if len(cases)!=18 or len({c['id'] for c in cases})!=18 or sum(len(c['required_label_policies']) for c in cases)!=42:
        raise ValueError('All18official checkpoint/context/dataset cases and42label views required')
    for case in cases:
        queue,job=scalar.resolve_case(case);model=read(ROOT/job['model_config']);p=model['parameters']
        combinations.add((p['checkpoint_sha256'],case['window'],case['dataset']))
    if len(combinations)!=18 or len({x[0] for x in combinations})!=3 or {x[1] for x in combinations}!={100,512}:
        raise ValueError('All3official sizes,both contexts and all3full datasets without duplicate slots required')
    files={scalar.CONFIG,POLICY,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/run_moment_full_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_moment_TAB100_metadata_recovery_v2.py',
        'scripts/flow_matching/run_moment_pretrained_job.py','src/iia_benchmark/models/fm_moment_pretrained.py'}
    files.update(p['path'] for p in old['source_receipts'])
    return dict(schema_version=1,upstream_scalar_configuration=scalar.CONFIG,source_execution=old['source_execution'],cases=cases,
        required_family_native_cases=18,required_family_label_views=42,original_all_family_cases=873,original_all_label_views=2037,
        settings=guard['settings'],resource_lock=guard['resource_lock'],controller_python=old['controller_python'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        mse_tolerance=guard['mse_tolerance'],native_comparison='exact_dtype_shape_content_bytes',
        expected_source_model_entrypoint='iia_benchmark.models.fm_moment_pretrained:MomentPretrainedWindow',
        official_checkpoint_sequence_length=512,external_contexts=[100,512],source_epochs=0,source_batch_size=1,source_seed=42,
        paper_and_data_citations=old['paper_and_data_citations'],original_scope=old['original_scope'],
        official_weights_citation='https://huggingface.co/AutonLab',paper_citation='https://arxiv.org/abs/2402.03885',
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full18_moment_checkpoint_replays_v1',
        state_root='experiments/runs/fm_original_full18_moment_checkpoint_replays_v1',
        boundary='All18newraw official zero-shot cases,not independent training-seed repetitions. '
            '3sizes*2contexts*3datasets;42raw label views. Require actual complete109data,successful source '
            'guard and all scalar label metrics before inference. Hash exact official weights/config and source assets. '
            'Keep checkpoint seq_len512 for both external contexts100/512;use corrected registered-length predicate, '
            'not the known old512!=100false rejection. This is a new-data component,not retry of old inference. '
            'Original full78native inference function code object and observed_score/groups remain unchanged; '
            'only validated source-binding provider and passive parameter observer are privately supplied. '
            'All frozen named parameters hashed before/after;no optimizer updates or data parameter estimation. '
            'Model fit() loads frozen weights only. Exact source batch1,padding,all-observed masks,2CPUthreads, '
            'allnative windows/tails and full source producer interpreter retained. Full array dtype/shape/bytes exact. '
            'Original2e-6full-window float64MSE check retained. Fresh review reproduces all source/label/event/ '
            'bootstrap/grid metrics. Original16/32/16GiB CPU headroom,10/12GiB caps and strict lock remain. '
            'Other855newraw native cases and all original papers,full Tiny/ICM pretraining,author fine-tuning, '
            'training-seed uncertainty,fullTAB and historical-paper equivalence remain separate.')


def verify(config):
    if config!=registration():raise ValueError('All18official models/42views and original numerical/resource policies required')
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
    p=model['parameters'];window=case['window']
    if (case['family']!='moment' or model['entrypoint']!=config['expected_source_model_entrypoint']
            or case['seed']!=42 or window not in [100,512] or job['input_kind']!='standard'
            or p['window']!=window or p['epochs']!=0 or p['batch_size']!=1 or p['device']!='cpu'
            or p['padding_policy']!=('minimal_patch' if window==512 else 'TAB_always_extra_patch')
            or job['parameter_overrides']!=dict(device='cpu',seed=42) or queue['evaluation']['cpu_threads']!=2):
        raise ValueError('Exact official zero-shot size/context/padding/batch/device/seed required')


def registered_length(model):
    from scripts.flow_matching.run_moment_TAB100_metadata_recovery_v2 import registered_checkpoint_length
    return registered_checkpoint_length(dict(config=model))


def validate_release_audit(result,model):
    p=model['parameters'];audit=result['checkpoint_load_audit']
    if (result['purpose']!='full_frozen_pretrained_zero_shot_tsad' or result['training_seconds']!=0
            or result['training_losses']!=[] or result['parameter_count']<=0 or result['parameter_count']!=audit['parameter_count']
            or audit['checkpoint_sha256']!=p['checkpoint_sha256'] or audit['config_sha256']!=p['config_sha256']
            or not audit['all_keys_strictly_loaded'] or audit['configured_sequence_length']!=registered_length(model)
            or audit['task']!='reconstruction' or audit['random_reinitialized_head'] or audit['hub_downloads']
            or audit['fine_tuning_epochs']!=0 or audit['fit_points_used_for_parameter_estimation']!=0):
        raise ValueError('Complete strict official512checkpoint metadata and zero parameter-estimation contract required')


def frozen_named_parameters(model):
    import torch
    records={}
    for name,p in model.model.named_parameters():
        if p.device.type!='cpu' or p.requires_grad:raise ValueError('Every official parameter must be frozen on CPU')
        flat=p.detach().contiguous().reshape(-1).view(torch.uint8);h=hashlib.sha256()
        for start in range(0,len(flat),2**20):h.update(flat[start:start+2**20].numpy().tobytes())
        records[name]=dict(dtype=str(p.dtype),shape=list(p.shape),elements=p.numel(),sha256=h.hexdigest())
    if not records:raise ValueError('Complete nonempty official named-parameter set required')
    return records


def legacy_binding(binding,case):
    return dict(binding,source_guard_history=dict(original_execution_guard_history_proven=True,
        records=[dict(path=case['source_guard'],sha256=binding['source_guard_sha256'])],
        missing_history_is_retroactively_certified=False))


def unchanged_native_body(source_provider,observer):
    from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as original
    namespace=dict(original.native_worker.__globals__,ROOT=ROOT,check_source=source_provider,observed_score=observer)
    return FunctionType(original.native_worker.__code__,namespace,original.native_worker.__name__,original.native_worker.__defaults__)


def producer_environment(result):
    import torch
    from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import environment_matches
    actual=environment_matches(result);torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU official-checkpoint replay must not initialize CUDA')
    return actual




def native_worker(config,case):
    import numpy as np,torch
    from scripts.flow_matching import run_moment_full_checkpoint_replay_v1 as original
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    old=ready_scalar(config,case)
    if old is None:raise ValueError('Actual complete source label/scalar audit required before MOMENT inference')
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result)
    model_config=read(ROOT/job['model_config']);validate_model(config,case,queue,job,model_config);validate_release_audit(result,model_config)
    reference=torch.load(ROOT/job['output_directory']/'model.pt',map_location='cpu',weights_only=True)
    if (reference['configuration']!=model_config or reference['overrides']!=job['parameter_overrides']
            or reference['checkpoint_load_audit']!=result['checkpoint_load_audit']
            or reference['checkpoint_reference']!=model_config['parameters']['checkpoint_directory']):
        raise ValueError('Exact unchanged frozen official checkpoint reference required')
    legacy_case=dict(job=job,config=model_config,queue=queue,path=case['queue_path'],family=case['family'])
    native_binding=legacy_binding(binding,case);observed={}
    def provide(runtime,actual_case):
        if actual_case!=legacy_case or runtime['_config_path']!=CONFIG:raise ValueError('Actual full source job identity differs')
        return result,data,native_binding
    def observe(model,window,tolerance):
        if window!=job['window'] or tolerance!=config['mse_tolerance']:raise ValueError('Original full MSE/context contract required')
        observed['model']=model;observed['before']=frozen_named_parameters(model)
        if sum(x['elements'] for x in observed['before'].values())!=result['parameter_count']:
            raise ValueError('Every complete frozen official parameter required')
        return original.observed_score(model,window,tolerance)
    runtime=dict(artifact_root=config['artifact_root'],mse_tolerance=config['mse_tolerance'],_config_path=CONFIG)
    body=unchanged_native_body(provide,observe);body(runtime,legacy_case)
    after=frozen_named_parameters(observed['model'])
    if observed['before']!=after:raise ValueError('Native inference changed an official frozen parameter')
    _,output=paths(config,case,'native');legacy_path=output/'native_replay.json';native=read(legacy_path)
    if (not native['passed'] or native['job']!=job or native['source_binding']!=native_binding
            or native['runtime_sha256']!=sha(ROOT/CONFIG) or not native['all_original_groups_reinferred']
            or not native['all_array_dtype_shape_content_identical'] or native['training_updates_performed']!=0
            or native['native_coverage']!=result['native_coverage'] or native['checkpoint_load_audit']!=result['checkpoint_load_audit']):
        raise ValueError('Unchanged full official model and complete source coverage required')
    obs=native['inline_full_window_MSE_check'];windows=sum(x['windows'] for x in native['native_coverage'])
    if obs['windows']!=windows or obs['forward_calls']!=windows or obs['window_points']!=windows*job['window']:
        raise ValueError('Every original batch1window and context coordinate must be observed')
    if sha(native['scores_path'])!=native['scores_sha256']:raise ValueError('Native serialized output changed')
    with np.load(native['scores_path'],allow_pickle=False) as scores,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        if set(scores.files)!=set(saved.files) or set(scores.files)!=set(native['arrays']):raise ValueError('All native/source arrays required')
        for key in scores.files:
            require_identical(scores[key],saved[key])
            if array_digest(scores[key])!=native['arrays'][key]:raise ValueError('Every serialized native array must reload identically')
    if scalar.source_binding(config,case)[-1]!=binding or ready_scalar(config,case)!=old:
        raise ValueError('Every original source/data/weight/label/scalar evidence must remain unchanged')
    write(output/REPORTS['native'],dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        upstream_scalar_evidence=old,actual_producer_environment=environment,parameters=result['parameter_count'],
        frozen_named_parameter_digests=observed['before'],all_frozen_named_parameters_identical_before_and_after=True,
        complete_official_weights_strictly_loaded=True,source_arrays=native['arrays'],
        full_native_coverage=[dict(entity=r['group'],role=r['role'],points=r['native_points']) for r in native['native_coverage']],
        original_native_report=str(legacy_path),original_native_report_sha256=sha(legacy_path),
        original_native_worker_code_unchanged=True,all_array_dtype_shape_content_bytes_identical=True,
        every_saved_native_array_serialized_and_reloaded=True,scores_path=native['scores_path'],scores_sha256=native['scores_sha256'],
        inline_full_window_MSE_check=obs,mse_tolerance=config['mse_tolerance'],
        official_checkpoint_loader_fit_calls=1,parameter_estimation_fit_calls=0,additional_optimizer_updates=0,
        CUDA_context_initialized=torch.cuda.is_initialized(),official_checkpoint_length=512,external_context=job['window'],
        new_training_seed=False,full_TAB_or_historical_author_equivalence=False,all_original_experiments_complete=False))


def ready_native(config,case):
    attempt,output=paths(config,case,'native');path=output/REPORTS['native'];guard=attempt/'guard_receipt.json'
    if not path.exists() or not guard.exists():raise ValueError('Actual full MOMENT inference and successful native guard required')
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['case']!=case or proof['configuration_sha256']!=sha(ROOT/CONFIG)
            or not proof['all_array_dtype_shape_content_bytes_identical'] or not proof['all_frozen_named_parameters_identical_before_and_after']
            or not proof['complete_official_weights_strictly_loaded'] or not proof['every_saved_native_array_serialized_and_reloaded']
            or not proof['original_native_worker_code_unchanged'] or proof['official_checkpoint_loader_fit_calls']!=1
            or proof['parameter_estimation_fit_calls'] or proof['additional_optimizer_updates'] or proof['CUDA_context_initialized']
            or proof['official_checkpoint_length']!=512 or proof['external_context']!=case['window']
            or proof['mse_tolerance']!=config['mse_tolerance'] or sha(proof['scores_path'])!=proof['scores_sha256']
            or sha(proof['original_native_report'])!=proof['original_native_report_sha256']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)
            or g['exact_child_handle']['command']!=worker_command(case,'native')):
        raise ValueError('Full unchanged official MOMENT native proof and exact resource guard required')
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
    evaluation=dict(queue['evaluation'],bootstrap_block_points=max(queue['evaluation']['bootstrap_block_points'],job['window']))
    views=scalar.evaluate_views(entities,validation,training,overlays,data,result,evaluation,job['seed'])
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
        independent_transformer_kernel_implementation=False,official_checkpoint_seed_is_not_a_new_training_seed=True,
        full_TAB_or_author_equivalence=False,all_original_experiments_complete=False))


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full18MOMENT native replay owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior complete/partial replay attempt')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_family_native_cases=18,required_family_label_views=42,active_pid=None,all_original_experiments_complete=False)
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
                    CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
            write(attempt/'launch.json',handle);update(state='running_full_MOMENT_'+phase,active_pid=child.pid,active_job=case['id'])
            code,usage=capped_wait(child,config,api,update)
        proof=output/REPORTS[phase];ok=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
        write(attempt/'guard_receipt.json',dict(passed=ok,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
        if not ok:raise ValueError('Full native/review attempt failed;preserve without retry')
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
                    completed.append(identity);update(state='between_full_MOMENT_replays',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='replay_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_original_training_and_scalar_audits',active_pid=None,active_job=None);time.sleep(10)
        update(state='all18family_replay_outcomes_terminal_other855native_cases_and_papers_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');parser.add_argument('--stage',choices=list(REPORTS));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=18,label_views=42)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=18,label_views=42,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.job_id:
        if args.stage is None:raise ValueError('An explicit native/review stage is required')
        case=next(c for c in cases if c['id']==args.job_id)
        (native_worker if args.stage=='native' else review_worker)(config,case)
    else:controller(config,cases)


if __name__=='__main__':main()
