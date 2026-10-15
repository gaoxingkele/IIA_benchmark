"""Complete native/independent review of440 local FM objective case records.

Original failures are negative outcomes,preserved separately from numerical
repairs. This verifies local TSAD adapters,not original image/single-cell papers.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import ROOT,read,sha,write,environment_matches

QUEUES={
 'original':('configs/experiments/fm_tsad_cpu_queue.v1.json',280),
 'repair':('configs/experiments/fm_tsad_entropic_repair_queue.v1.json',105),
 'constant_fit':('configs/experiments/fm_tsad_constant_fit_cpu_queue.v2.json',40),
 'constant_fit_repair':('configs/experiments/fm_tsad_constant_fit_entropic_numeric_repair_queue.v2.json',15)}
RESOURCE='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
MODULE='scripts.flow_matching.run_objective_full440_checkpoint_replay_v1'
OLD_SB='configs/reproducibility/fm_local_SB_SWaT_full_checkpoint_audit.2026-10-14_v1.json'
LEGACY_TERMINAL='experiments/runs/fm_tsad_execution_v1/cpu_status.json'


def verify(runtime):
    original=read(ROOT/RESOURCE)
    if (runtime['queues']!={k:dict(path=p,count=n) for k,(p,n) in QUEUES.items()} or runtime['required_cases']!=440
        or runtime['settings']!=original['settings'] or runtime['resource_lock']!=original['resource_lock']
        or runtime['math_tolerance']!=dict(rtol=2e-6,atol=2e-6)
        or runtime['probability_path_pass_criterion']!='independent_numpy_source_FP32_operation_program'
        or runtime['ideal_float64_path_audit']!='all_coordinate_failure_counts_and_errors_retained_diagnostic'
        or runtime['array_comparison']!='exact_dtype_shape_content_bytes'
        or runtime['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('All440 original/repair/control cases and frozen full numerical safeguards required')
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full objective replay dependency changed: '+pin['path'])
    cases={}
    for family,(path,count) in QUEUES.items():
        q=read(ROOT/path)
        if len(q['jobs'])!=count or q['evaluation']['cpu_threads']!=4:raise ValueError('Full original case count and4CPUthreads required')
        for job in q['jobs']:
            c=read(ROOT/job['model_config']);p=c['parameters']
            if (job['id'] in cases or job['seed'] not in [1103,1104,1105,1106,1107]
                or job['parameter_overrides']!=dict(seed=job['seed'],device='cpu')
                or p['epochs']!=20 or p['window']!=100 or job['window']!=100 or p['hidden_size']!=128
                or p['batch_size']!=64 or p['monte_carlo_samples']!=4 or p['score_times']!=[.1,.3,.5,.7,.9]
                or p['learning_rate']!=.001 or c['entrypoint'] not in [
                    'iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector',
                    'iia_benchmark.models.fm_objective_stable:StableCFMObjectiveDetector']):
                raise ValueError('Complete unique full-budget local objective case required')
            cases[job['id']]=dict(job=job,config=c,queue=q,path=path,family=family)
    if runtime['required_ids']!=list(cases):raise ValueError('Every original and separate repair/control identity required')
    terminal=read(ROOT/LEGACY_TERMINAL)
    failed={row['id']:row for row in terminal['records'] if row['status']=='failed' and row.get('returncode')!=0}
    if terminal['status']!='completed_with_unresolved_jobs' or len(failed)!=105 or runtime['preserved_original_failures']!=failed:
        raise ValueError('All105 actual terminal original failures must remain unmodified,not retried or called successful')
    for identity in failed:
        if identity not in cases or cases[identity]['family']!='original':raise ValueError('Wrong original failure identity')
    closed=runtime['existing_native_certificate']
    if closed['configuration']!=OLD_SB or closed['id']!='fm_objective_sb__swat__seed1103__stable_numerics_v2':
        raise ValueError('Only the exact alreadyclosed full SB native certificate may be reused')
    return cases


def source_guard_records(runtime,case):
    records=[]
    for directory in runtime['source_guard_directories'][case['family']]:
        path=ROOT/directory/(case['job']['id']+'.json')
        if path.exists():
            g=read(path)
            if (g.get('job_id')!=case['job']['id'] or g.get('queue_sha256')!=sha(ROOT/case['path'])):
                raise ValueError('Source guard belongs to another original case')
            records.append((path,g))
    return records


def check_source(runtime,case):
    job=case['job'];base=ROOT/job['output_directory'];r=read(base/'result.json');p=case['config']['parameters']
    if (r['status']!='completed' or r['job']!=job or r['id']!=job['id'] or r['purpose']!='full_local_tsad_benchmark'
        or r['experiment_sha256']!=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
        or len(r['training_losses'])!=20 or not all(math.isfinite(v) and v>=0 for v in r['training_losses'])
        or r['parameter_count']<=0 or sha(base/'model.pt')!=r['checkpoint_sha256'] or sha(base/'scores.npz')!=r['scores_sha256']):
        raise ValueError('Actual complete20-epoch unchanged source weights,scores and result required')
    for pin in job['frozen_files']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Original frozen objective asset changed: '+pin['path'])
    ds=next(d for d in read(ROOT/job['dataset_manifest_path'])['datasets'] if d['id']==job['dataset'])
    if not ds['grouped_splits_verified'] or sha(ROOT/ds['input_npz'])!=ds['input_sha256']:
        raise ValueError('Complete original grouped input required')
    guards=source_guard_records(runtime,case)
    for path,g in guards:
        if g.get('exit_code')!=0 or g.get('resource_guard_aborted') is not False or not g.get('full_registered_budget_retained'):
            raise ValueError('Existing source execution resource guard did not pass')
    return r,ds,dict(result_sha256=sha(base/'result.json'),checkpoint_sha256=r['checkpoint_sha256'],scores_sha256=r['scores_sha256'],
        input_sha256=ds['input_sha256'],queue_sha256=sha(ROOT/case['path']),model_config_sha256=sha(ROOT/job['model_config']),
        historical_execution_guard_proven=bool(guards),source_guard_records=[dict(path=str(path),sha256=sha(path)) for path,_ in guards])


def producer_environment(result):
    import platform,numpy as np,torch
    actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device='cpu')
    if actual!=result['environment']:raise ValueError('Exact original objective producer environment required')
    return actual


def reused_native(runtime,case,result):
    known=runtime['existing_native_certificate']
    if case['job']['id']!=known['id']:raise ValueError('No existing native certificate for this case')
    c=read(ROOT/known['configuration']);proof=ROOT/c['output_directory']/'independent_full_model_score_verification.json'
    guard=ROOT/c['state_root']/'receipt.json';p,g=read(proof),read(guard)
    if (not p['passed'] or p['canonical_id']!=case['job']['id'] or g['exit_code']!=0 or g['resource_guard_aborted']
        or not g['full_cold_model_replay_passed'] or p['configuration_sha256']!=sha(ROOT/known['configuration'])
        or p['source_result_sha256']!=sha(ROOT/case['job']['output_directory']/'result.json')
        or p['source_checkpoint_sha256']!=result['checkpoint_sha256'] or p['source_scores_sha256']!=result['scores_sha256']
        or not p['original_train_validation_test_native_scores_reinferred_bitwise']
        or not p['complete_checkpoint_tensors_restored_bitwise'] or p['additional_fit_calls']!=0):
        raise ValueError('Existing native certificate is not a valid full unchanged original case proof')
    return dict(path=str(proof),sha256=sha(proof),guard_sha256=sha(guard),full_native_reinference_reused=True)


def restore_model(case,reference,features):
    import torch
    from torch import nn
    from scripts.flow_matching.model_registry import build_configured_method
    if (reference['configuration']!=case['config'] or reference['overrides']!=case['job']['parameter_overrides']
        or reference['window']!=100 or set(reference['modules'])!={'_model'}):raise ValueError('Full exact objective checkpoint structure required')
    model=build_configured_method(case['config'],case['job']['parameter_overrides'])
    model._shape=(100,features);model._device=torch.device('cpu');model._score_head=model.objective=='sf2m' and model.score_weight>0
    dimensions=100*features;hidden=model.hidden_size
    model._model=nn.Sequential(nn.Linear(dimensions+1,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),
        nn.Linear(hidden,dimensions*(2 if model._score_head else 1)))
    state=reference['modules']['_model'];model._model.load_state_dict(state,strict=True);model._model.eval()
    if any(not torch.equal(v,state[k]) or not torch.isfinite(v).all() for k,v in model._model.state_dict().items()):
        raise ValueError('Every complete trained velocity/score-head tensor must restore bitwise')
    model._model.requires_grad_(False)
    return model


def native_worker(runtime,case):
    import numpy as np,torch
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical
    from scripts.flow_matching.objective_replay_math_v1 import checked_score
    result,dataset,binding=check_source(runtime,case);environment=producer_environment(result);torch.set_num_threads(4)
    job=case['job'];base=ROOT/job['output_directory'];output=Path(runtime['artifact_root'])/job['id']/'native'
    output.mkdir(parents=True,exist_ok=False)
    reference=torch.load(base/'model.pt',map_location='cpu',weights_only=True)
    payload={};coverage=[];metadata={};start=time.perf_counter()
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs,np.load(base/'scores.npz',allow_pickle=False) as saved:
        features=inputs[dataset['entities'][0]['array_prefix']+'_train'].shape[1]
        model=restore_model(case,reference,features)
        if sum(p.numel() for p in model._model.parameters())!=result['parameter_count']:raise ValueError('Full parameter count differs')
        score,observation=checked_score(model)
        for entity in dataset['entities']:
            prefix=entity['array_prefix']
            for role in ['train','validation','test']:
                values=inputs[prefix+'_'+role];windows,starts=covering_windows(values,100)
                raw=score(windows);aligned=align_scores(raw,starts,len(values),100)
                require_identical(aligned,saved[prefix+'_'+role]);payload[prefix+'_'+role]=aligned
                coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values),windows=len(windows)))
            require_identical(inputs[prefix+'_labels'],saved[prefix+'_labels']);payload[prefix+'_labels']=inputs[prefix+'_labels']
            for suffix in ['_labels','_timestamps','_source_rows','_fault_labels']:
                if prefix+suffix in inputs.files:metadata[prefix+suffix]=array_digest(inputs[prefix+suffix])
            print(json.dumps(dict(id=job['id'],stage='full_native_objective_entity_replayed',entity=entity['entity_id'])),flush=True)
        if set(saved.files)!=set(payload):raise ValueError('Every complete original score/label array required')
    expected_windows=sum(r['windows'] for r in coverage)
    expected_calls=sum(math.ceil(r['windows']/64)*20 for r in coverage)
    if (observation['windows']!=expected_windows or observation['path_calls']!=expected_calls
        or observation['forward_calls']!=expected_calls or observation['score_calls']!=len(coverage)):
        raise ValueError('Every window,batch,MC sample and scoring time must be replayed')
    path=output/'scores.npz';np.savez_compressed(path,**payload)
    write(output/'native_replay.json',dict(passed=True,job=job,source_binding=binding,runtime_sha256=sha(ROOT/runtime['_config_path']),
        complete_checkpoint_restored_bitwise=True,parameter_count=result['parameter_count'],producer_environment=environment,
        full_split_coverage=coverage,all_original_arrays_identical=True,all_native_groups_reinferred=True,
        full_MC_probability_path_and_score_formula_observation=observation,math_tolerance=runtime['math_tolerance'],
        ideal_float64_path_equivalence_certified=observation['ideal_float64_path_allclose_passed'],
        scores_path=str(path),scores_sha256=sha(path),arrays={k:array_digest(v) for k,v in payload.items()},
        input_metadata_digests=metadata,wall_seconds=time.perf_counter()-start,training_updates_performed=0,fit_calls=0,
        original_author_TSAD_equivalence=False,strict_TAB_harness_equivalence=False,all_original_experiments_complete=False))


def review_worker(runtime,case):
    import numpy as np
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    from scripts.flow_matching.run_moment_full_checkpoint_replay_v1 import independent_controls
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical
    result,dataset,binding=check_source(runtime,case);producer_environment(result)
    job=case['job'];base=ROOT/job['output_directory'];output=Path(runtime['artifact_root'])/job['id']
    reuse=job['id']==runtime['existing_native_certificate']['id']
    if reuse:
        proof_binding=reused_native(runtime,case,result);score_path=base/'scores.npz';native=None
    else:
        native_path=output/'native/native_replay.json';native=read(native_path)
        if (not native['passed'] or native['job']!=job or native['source_binding']!=binding
            or native['runtime_sha256']!=sha(ROOT/runtime['_config_path']) or not native['all_original_arrays_identical']
            or not native['all_native_groups_reinferred'] or native['math_tolerance']!=runtime['math_tolerance']
            or native['scores_sha256']!=sha(native['scores_path']) or native['fit_calls']!=0 or native['training_updates_performed']!=0):
            raise ValueError('Unchanged complete native model replay proof required')
        proof_binding=dict(path=str(native_path),sha256=sha(native_path));score_path=Path(native['scores_path'])
    entities={};train=[];validation=[];coverage=[]
    with np.load(score_path,allow_pickle=False) as scores,np.load(base/'scores.npz',allow_pickle=False) as saved,np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs:
        if set(scores.files)!=set(saved.files):raise ValueError('All original arrays required in independent review')
        for key in scores.files:
            require_identical(scores[key],saved[key])
            if native and array_digest(scores[key])!=native['arrays'][key]:raise ValueError('Native score coordinate receipt differs')
        for entity in dataset['entities']:
            prefix=entity['array_prefix']
            for role in ['train','validation','test']:
                values=scores[prefix+'_'+role]
                if len(values)!=len(inputs[prefix+'_'+role]) or not np.isfinite(values).all():raise ValueError('Full native source time coverage required')
                coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values)))
                if role=='train':train.append(values)
                elif role=='validation':validation.append(values)
                else:
                    require_identical(inputs[prefix+'_labels'],scores[prefix+'_labels'])
                    entities[entity['entity_id']]=(scores[prefix+'_labels'],values)
            if native:
                for suffix in ['_labels','_timestamps','_source_rows','_fault_labels']:
                    if prefix+suffix in inputs.files and array_digest(inputs[prefix+suffix])!=native['input_metadata_digests'][prefix+suffix]:
                        raise ValueError('Input timestamp/row/fault/label provenance differs')
        train=np.concatenate(train);validation=np.concatenate(validation)
        metrics=evaluate_saved(entities,validation,train,dataset['input_sha256'],result['checkpoint_sha256'],case['queue']['evaluation'],job['seed'])
        if metrics!=result['metrics']:raise ValueError('Every original metric,event and200resample grouped interval must replay exactly')
        controls=independent_controls(entities,validation,train,metrics,case['queue']['evaluation'])
    output.joinpath('independent').mkdir(parents=True,exist_ok=False)
    write(output/'independent/review.json',dict(passed=True,job=job,source_binding=binding,native_proof_binding=proof_binding,
        all_original_arrays_and_labels_reviewed=True,full_native_split_coverage=coverage,existing_native_certificate_reused=reuse,
        new_native_inference_performed_by_this_review=False,inline_path_formula_audited_by_new_native_stage=not reuse,
        ideal_float64_path_equivalence_certified=native['ideal_float64_path_equivalence_certified'] if native else False,
        metrics_events_and_bootstrap_replayed_exactly=True,independent_all_core_protocol_controls=controls,
        grouped_bootstrap_uses_original_library=True,VUS_affiliation_integrals_independently_recomputed=False,
        raw_preprocessing_values_independently_reconstructed=False,training_updates_performed=0,
        original_author_TSAD_equivalence=False,strict_TAB_harness_equivalence=False,all_original_experiments_complete=False))


def source_ready(runtime,case):
    if (ROOT/case['job']['output_directory']/'result.json').exists():return True
    for path,g in source_guard_records(runtime,case):
        if g.get('exit_code') is not None:
            raise ValueError('Terminal source receipt has no completed result: '+str(path))
    return False


def controller(runtime,cases):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    import psutil
    api=install_facades();directory=ROOT/runtime['state_root'];directory.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](directory/'controller.lock')
    if lane is None:raise RuntimeError('Full440 objective verification already live')
    verified=[];source_failed=dict(runtime['preserved_original_failures']);failed={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=440,original_scope=runtime['original_scope'],all_original_experiments_complete=False)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw);state.update(verified_ids=list(verified),verified_count=len(verified),
            source_failure_outcomes=dict(source_failed),source_failure_count=len(source_failed),failed_verifications=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat());write(directory/'status.json',state)
    def stage(case,phase):
        identity=case['job']['id'];attempt=directory/'attempts'/identity/phase;receipt=attempt/'guard_receipt.json'
        proof=Path(runtime['artifact_root'])/identity/('native/native_replay.json' if phase=='native' else 'independent/review.json')
        if receipt.exists():
            old=read(receipt)
            if old['passed'] and old['exit_code']==0 and not old['resource_guard_aborted'] and old['proof_sha256']==sha(proof) and read(proof)['passed']:
                _,_,binding=check_source(runtime,case)
                if read(proof)['source_binding']!=binding:raise ValueError('Previously verified source changed')
                return True
            raise ValueError('Prior full verification failed and remains preserved')
        if attempt.exists() or proof.exists() or (phase=='native' and proof.parent.exists()):raise ValueError('Prior incomplete verification preserved;no automatic retry')
        attempt.mkdir(parents=True,exist_ok=False)
        command=[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',MODULE,'--config',runtime['_config_path'],'--stage',phase,'--job-id',identity]
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command);write(attempt/'launch.json',handle)
            update(state='running_full_objective_'+phase,active_pid=child.pid,active_create_time=handle['create_time'],job=identity)
            code,usage=capped_wait(child,runtime,api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(receipt,dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None,exact_child_handle=handle))
        return passed
    try:
        while len(verified)+len(source_failed)+len(failed)<len(cases):
            for identity,case in cases.items():
                if identity in verified or identity in source_failed or identity in failed:continue
                try:
                    if not source_ready(runtime,case):continue
                except Exception as error:source_failed[identity]=str(error);continue
                try:
                    def waiting(value):update(dict(value,active_pid=None,job=identity))
                    with api['resource_slot'](ROOT/runtime['resource_lock'],runtime['settings'],waiting,poll_seconds=10):
                        if identity!=runtime['existing_native_certificate']['id'] and not stage(case,'native'):raise ValueError('Full native guard/worker failed')
                        if not stage(case,'independent'):raise ValueError('Full independent guard/worker failed')
                    verified.append(identity)
                except Exception as error:failed[identity]=str(error)
                update(state='scanning_actual_objective_results',active_pid=None,job=None)
            update(state='waiting_actual_full_objective_results',active_pid=None,job=None);time.sleep(10)
        update(state='all440_cases_terminal_replay_or_preserved_failure_other_experiments_remain',active_pid=None,job=None)
    finally:lane.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--probe-only',action='store_true')
    p.add_argument('--stage',choices=['native','independent']);p.add_argument('--job-id');args=p.parse_args()
    runtime=read(ROOT/args.config);cases=verify(runtime);runtime['_config_path']=args.config
    if args.probe_only:print(json.dumps(dict(passed=True,cases=440,preserved_terminal_original_failures=105,
        existing_source_results=sum((ROOT/c['job']['output_directory']/'result.json').exists() for c in cases.values()),
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,new_native_models_verified=0,all_original_experiments_complete=False)))
    elif args.stage:(native_worker if args.stage=='native' else review_worker)(runtime,cases[args.job_id])
    else:controller(runtime,cases)


if __name__=='__main__':main()
