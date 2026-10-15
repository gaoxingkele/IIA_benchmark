"""Full native replay of all72 MOMENT jobs plus6 constant-fit controls.

Use the exact producer interpreter,checkpoint,context,batch1,padding,group
boundaries and all observed tokens. Source jobs and historical guard evidence
are never changed. Range integrals/historical paper fidelity remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.run_anomaly_transformer_author_full_v1 import ROOT, read, write

QUEUES = {
    'original': ('configs/experiments/fm_moment_pretrained_queue.v1.json', 72),
    'constant_fit': ('configs/experiments/fm_tsad_constant_fit_moment_pretrained_queue.v2.json', 6)}
RESOURCE = 'configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
MODULE = 'scripts.flow_matching.run_moment_full_checkpoint_replay_v1'


def sha(path):
    """The exact producer is Python3.10; file_digest was added in Python3.11."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify(runtime):
    original = read(ROOT/RESOURCE)
    if (runtime['queues'] != {k:dict(path=p, count=n) for k,(p,n) in QUEUES.items()}
            or runtime['required_jobs'] != 78 or runtime['settings'] != original['settings']
            or runtime['resource_lock'] != original['resource_lock']
            or runtime['mse_tolerance'] != dict(rtol=2e-6,atol=2e-6)
            or runtime['array_comparison'] != 'exact_dtype_shape_content_bytes'
            or runtime['original_scope'] != dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('All78 original/control jobs and unchanged full native resource/scoring contract required')
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen native MOMENT replay dependency changed: '+pin['path'])
    cases = {}
    for family,(path,count) in QUEUES.items():
        queue = read(ROOT/path)
        if len(queue['jobs']) != count or queue['evaluation']['cpu_threads'] != 2:
            raise ValueError('All original data/context/model slots and producer threads required')
        combinations = set()
        for job in queue['jobs']:
            config = read(ROOT/job['model_config']); p = config['parameters']
            if (job['seed'] != 42 or p['epochs'] != 0 or p['batch_size'] != 1 or p['window'] != job['window']
                    or job['parameter_overrides'] != dict(device='cpu',seed=42)
                    or config['entrypoint'] != 'iia_benchmark.models.fm_moment_pretrained:MomentPretrainedWindow'
                    or job['window'] not in [100,512] or p['device'] != 'cpu'
                    or p['padding_policy'] != ('minimal_patch' if job['window']==512 else 'TAB_always_extra_patch')
                    or job['id'] in cases):
                raise ValueError('Actual unique pretrained zero-shot checkpoint/context identity required')
            combinations.add((p['checkpoint_sha256'],job['window'],job['dataset']))
            cases[job['id']] = dict(job=job,config=config,queue=queue,path=path,family=family)
        if len(combinations) != count:
            raise ValueError('No duplicate checkpoint/context/dataset slots')
    if runtime['required_ids'] != list(cases) or len(cases) != 78:
        raise ValueError('Preserve all72 original jobs and all6 preprocessing controls separately')
    return cases


def groups(job, dataset):
    if job['input_kind'] == 'industrial':
        return [(role,g,g['array_prefix']+('_test' if role=='test' else '_values'))
            for role,records in [('train',dataset['training_groups']),('validation',dataset['validation_groups']),('test',dataset['entities'])]
            for g in records]
    return [(role,g,g['array_prefix']+'_'+role) for g in dataset['entities'] for role in ['train','validation','test']]


def guard_history(runtime, case):
    records = []
    for directory in runtime['source_guard_directories'][case['family']]:
        path = ROOT/directory/(case['job']['id']+'.json')
        if not path.exists():
            continue
        guard = read(path)
        if (guard.get('job_id') != case['job']['id'] or guard.get('exit_code') != 0
                or guard.get('resource_guard_aborted') is not False or not guard.get('full_registered_budget_retained')
                or guard.get('queue_sha256') != sha(ROOT/case['path'])):
            raise ValueError('Existing source execution guard failed/differs; preserve original attempt')
        records.append(dict(path=str(path),sha256=sha(path)))
    return dict(original_execution_guard_history_proven=bool(records),records=records,
        missing_history_is_retroactively_certified=False)


def check_source(runtime, case):
    job = case['job']; base = ROOT/job['output_directory']; result = read(base/'result.json')
    for pin in job['frozen_files']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Original MOMENT source/data/weights changed: '+pin['path'])
    audit = result['checkpoint_load_audit']; parameters = case['config']['parameters']
    if (result['status'] != 'completed' or result['id'] != job['id'] or result['job'] != job
            or result['experiment_sha256'] != hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
            or result['purpose'] != 'full_frozen_pretrained_zero_shot_tsad'
            or result['training_seconds'] != 0 or result['training_losses'] != []
            or result['parameter_count'] != audit['parameter_count'] or result['parameter_count'] <= 0
            or audit['checkpoint_sha256'] != parameters['checkpoint_sha256']
            or audit['config_sha256'] != parameters['config_sha256'] or not audit['all_keys_strictly_loaded']
            or audit['configured_sequence_length'] != job['window'] or audit['task'] != 'reconstruction'
            or audit['random_reinitialized_head'] or audit['hub_downloads']
            or audit['fine_tuning_epochs'] != 0 or audit['fit_points_used_for_parameter_estimation'] != 0
            or sha(base/'model.pt') != result['checkpoint_sha256'] or sha(base/'scores.npz') != result['scores_sha256']):
        raise ValueError('Full unchanged official-weight zero-shot result required')
    manifest = read(ROOT/job['dataset_manifest_path'])
    dataset = next(d for d in manifest['datasets'] if d['id'] == job['dataset'])
    if sha(ROOT/dataset['input_npz']) != dataset['input_sha256'] or not dataset['grouped_splits_verified']:
        raise ValueError('Full immutable grouped dataset required')
    history = guard_history(runtime,case)
    return result,dataset,dict(result_sha256=sha(base/'result.json'),scores_sha256=sha(base/'scores.npz'),
        checkpoint_reference_sha256=sha(base/'model.pt'),input_sha256=dataset['input_sha256'],
        queue_sha256=sha(ROOT/case['path']),model_config_sha256=sha(ROOT/job['model_config']),source_guard_history=history)


def environment_matches(result):
    import platform
    import numpy as np
    import torch
    actual = dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu')
    if actual != result['environment']:
        raise ValueError('Actual producer interpreter/NumPy/Torch must remain exact')
    return actual


def observed_score(model, window, tolerance):
    """Wrap only this fresh model instance; preserve the exact original forward."""
    import numpy as np
    import torch
    forward, score = model.model.forward, model.score
    record = dict(forward_calls=0,windows=0,window_points=0,maximum_mse_absolute_error=0.)
    def checked(windows):
        independently = []
        def capture(*args,**kwargs):
            x = kwargs['x_enc']; observed, mask = kwargs['input_mask'],kwargs['mask']
            if (x.device.type!='cpu' or x.dtype!=torch.float32 or x.shape[-1]<window
                    or not torch.equal(observed,torch.ones_like(observed)) or not torch.equal(mask,torch.ones_like(mask))):
                raise ValueError('Exact full CPU inputs and explicit all-observed masks required')
            target = x[...,:window].detach().numpy().astype(np.float64)
            out = forward(*args,**kwargs)
            prediction = out.reconstruction[...,:window].detach().numpy().astype(np.float64)
            if prediction.shape != target.shape:
                raise ValueError('Full reconstructed feature/time shape required')
            independently.append(np.square(target-prediction).mean(axis=1))
            record['forward_calls']+=1;record['windows']+=len(x);record['window_points']+=len(x)*window
            return out
        model.model.forward = capture
        try:
            actual = score(windows)
        finally:
            model.model.forward = forward
        expected = np.concatenate(independently,axis=0)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected,**tolerance):
            raise ValueError('Every-window independent float64 reconstruction-MSE check failed')
        record['maximum_mse_absolute_error']=max(record['maximum_mse_absolute_error'],float(np.max(np.abs(actual-expected))))
        return actual
    return checked,record


def native_worker(runtime,case):
    import numpy as np
    import torch
    from scripts.flow_matching.model_registry import build_configured_method
    from scripts.flow_matching.run_moment_pretrained_job import segment_windows,segment_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical
    result,dataset,binding = check_source(runtime,case); environment = environment_matches(result)
    torch.set_num_threads(2)
    job=case['job']; base=ROOT/job['output_directory']; output=Path(runtime['artifact_root'])/job['id']/'native'
    output.mkdir(parents=True,exist_ok=False)
    reference=torch.load(base/'model.pt',map_location='cpu',weights_only=True)
    if (reference['configuration']!=case['config'] or reference['overrides']!=job['parameter_overrides']
            or reference['checkpoint_load_audit']!=result['checkpoint_load_audit']):
        raise ValueError('Original full frozen checkpoint reference must match the source job')
    model=build_configured_method(case['config'],job['parameter_overrides'])
    layout=groups(job,dataset); first=next(key for role,_,key in layout if role=='train')
    payload={}; coverage=[]; start=time.perf_counter()
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs:
        fitting,_,_=segment_windows(inputs[first],job['window']);model.fit(fitting);del fitting
        if (model.checkpoint_load_audit_!=result['checkpoint_load_audit'] or model.model.training
                or any(p.requires_grad or p.device.type!='cpu' for p in model.model.parameters())
                or sum(p.numel() for p in model.model.parameters())!=result['parameter_count']):
            raise ValueError('Complete strict frozen pretrained weights must be loaded without training')
        checked,observation=observed_score(model,job['window'],runtime['mse_tolerance']);model.score=checked
        for role,group,key in layout:
            score,audit=segment_scores(model,inputs[key],job['window'])
            prefix=group['array_prefix'];payload[prefix+'_'+role]=score
            for suffix in ['_labels','_timestamps','_source_rows','_fault_labels']:
                if prefix+suffix in inputs.files:payload[prefix+suffix]=inputs[prefix+suffix]
            coverage.append(dict(audit,role=role,group=group.get('entity_id',group.get('group_id')),source_array=key))
            print(json.dumps(dict(id=job['id'],stage='full_native_replayed_group',role=role,points=len(score))),flush=True)
    if coverage!=result['native_coverage'] or observation['windows']!=sum(r['windows'] for r in coverage):
        raise ValueError('Every original native group/timestamp/window/padding policy must replay')
    with np.load(base/'scores.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(payload):raise ValueError('Every source score/label/timestamp/group array required')
        for key in saved.files:require_identical(saved[key],payload[key])
    target=output/'scores.npz';np.savez_compressed(target,**payload)
    write(output/'native_replay.json',dict(passed=True,job=job,source_binding=binding,
        runtime_sha256=sha(ROOT/runtime['_config_path']),actual_producer_environment=environment,
        checkpoint_load_audit=model.checkpoint_load_audit_,native_coverage=coverage,all_original_groups_reinferred=True,
        all_array_dtype_shape_content_identical=True,inline_full_window_MSE_check=observation,
        scores_path=str(target),scores_sha256=sha(target),arrays={k:array_digest(v) for k,v in payload.items()},
        wall_seconds=time.perf_counter()-start,training_updates_performed=0,new_training_seed=False,
        original_source_guard_history=binding['source_guard_history'],historical_paper_equivalence_certified=False,
        full_TAB_harness_certified=False,all_original_experiments_complete=False))


def independent_controls(entities,validation,training,metrics,evaluation):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import binary_counts,ranking_metrics,segment_adjust,require_same_metrics
    labels=np.concatenate([v[0] for v in entities.values()]);scores=np.concatenate([v[1] for v in entities.values()])
    def ranked(truth,values,threshold):
        actual=binary_counts(truth,values>threshold);auc,ap=ranking_metrics(truth,values)
        if np.asarray(truth,dtype=bool).all():ap=1.
        actual.update(auroc=auc,average_precision=ap)
        return actual
    controls={}
    for ratio in evaluation['strict_false_alarm_percent']:
        source=metrics['strict'][str(ratio)];threshold=float(np.percentile(np.asarray(validation,dtype=np.float64),100-ratio,method='linear'))
        if source['calibration']['threshold']!=threshold or source['calibration']['calibration_split']!='validation':
            raise ValueError('Independent original full-validation threshold differs')
        micro=ranked(labels,scores,threshold);require_same_metrics(source['micro'],micro)
        per={key:ranked(y,s,threshold) for key,(y,s) in entities.items()}
        require_same_metrics(source['per_entity'],per)
        macro=float(np.mean([r['f1'] for r in per.values()]))
        if source['macro_f1']!=macro:raise ValueError('Original equal-entity macro F1 differs')
        adjusted=np.concatenate([segment_adjust(y,s>threshold) for y,s in entities.values()])
        controls[str(ratio)]=dict(threshold=threshold,micro=micro,per_entity=per,macro_f1=macro,
            same_validation_threshold_PA_diagnostic=binary_counts(labels,adjusted))
    if len(metrics['tab_core_diagnostic']['ratio_grid'])!=len(evaluation['tab_threshold_ratios_percent']):
        raise ValueError('Every original diagnostic calibration grid point required')
    for ratio,source in zip(evaluation['tab_threshold_ratios_percent'],metrics['tab_core_diagnostic']['ratio_grid']):
        threshold=float(np.percentile(np.concatenate([training,scores]),100-ratio))
        plain=binary_counts(labels,scores>threshold)
        adjusted=np.concatenate([segment_adjust(y,s>threshold) for y,s in entities.values()])
        if (source['ratio_percent']!=ratio or source['threshold']!=threshold
                or source['f_score']!=plain['f1'] or source['adjust_f_score']!=binary_counts(labels,adjusted)['f1']):
            raise ValueError('Independent original complete test-pooled diagnostic grid differs')
    return controls


def review_worker(runtime,case):
    import numpy as np
    from scripts.flow_matching.run_tsad_experiment import evaluate_saved
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest,require_identical
    result,dataset,binding=check_source(runtime,case);environment_matches(result)
    job=case['job'];base=ROOT/job['output_directory'];output=Path(runtime['artifact_root'])/job['id']
    path=output/'native/native_replay.json';native=read(path)
    if (not native['passed'] or native['job']!=job or native['source_binding']!=binding
            or native['runtime_sha256']!=sha(ROOT/runtime['_config_path'])
            or not native['all_original_groups_reinferred'] or not native['all_array_dtype_shape_content_identical']
            or native['native_coverage']!=result['native_coverage'] or native['training_updates_performed']!=0
            or native['scores_sha256']!=sha(native['scores_path'])):
        raise ValueError('Unchanged complete full checkpoint replay proof required')
    entities={};training=[];validation=[]
    with np.load(native['scores_path'],allow_pickle=False) as scores,np.load(base/'scores.npz',allow_pickle=False) as original,np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs:
        if set(scores.files)!=set(original.files) or set(scores.files)!=set(native['arrays']):
            raise ValueError('Every original array must be independently reviewed')
        for key in scores.files:
            require_identical(scores[key],original[key])
            if array_digest(scores[key])!=native['arrays'][key]:raise ValueError('Native array content receipt differs')
        for role,group,key in groups(job,dataset):
            prefix=group['array_prefix'];values=scores[prefix+'_'+role]
            if len(values)!=len(inputs[key]) or not np.isfinite(values).all():raise ValueError('Full original source timestamp coverage required')
            for suffix in ['_labels','_timestamps','_source_rows','_fault_labels']:
                if prefix+suffix in inputs.files:require_identical(inputs[prefix+suffix],scores[prefix+suffix])
            if role=='train':training.append(values)
            elif role=='validation':validation.append(values)
            else:entities[group['entity_id']]=(scores[prefix+'_labels'],values)
        evaluation=dict(case['queue']['evaluation'],bootstrap_block_points=max(case['queue']['evaluation']['bootstrap_block_points'],job['window']))
        validation=np.concatenate(validation);training=np.concatenate(training)
        recalculated=evaluate_saved(entities,validation,training,dataset['input_sha256'],result['checkpoint_sha256'],evaluation,job['seed'])
        if recalculated!=result['metrics']:raise ValueError('Full native score metrics/events/grouped bootstraps must recompute exactly')
        controls=independent_controls(entities,validation,training,recalculated,evaluation)
    output.joinpath('independent').mkdir(exist_ok=False)
    write(output/'independent/review.json',dict(passed=True,job=job,source_binding=binding,native_replay_sha256=sha(path),
        all_original_arrays_independently_checked=True,all_original_core_metrics_events_and_200_resample_intervals_recomputed=True,
        independent_confusion_ranking_validation_threshold_PA_and_ten_ratio_controls=controls,
        bootstrap_recalculation_uses_original_library_implementation=True,
        numerical_VUS_and_affiliation_integrals_independently_recomputed=False,
        full_preprocessing_values_independently_reconstructed=False,fresh_model_inference_in_this_second_stage=False,
        original_source_guard_history=binding['source_guard_history'],new_training_seed=False,
        historical_paper_equivalence_certified=False,full_TAB_harness_certified=False,all_original_experiments_complete=False))


def controller(runtime,cases):
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    import psutil
    api=install_facades();directory=ROOT/runtime['state_root'];directory.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](directory/'controller.lock')
    if lane is None:raise RuntimeError('All78 full MOMENT checkpoint review already live')
    verified=[];failed={}
    common=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_jobs=78,original_jobs=72,
        constant_fit_controls=6,original_scope=runtime['original_scope'],all_original_experiments_complete=False)
    state=dict(common)
    def update(value=None,**kw):
        state.update(value or {});state.update(kw)
        state.update(verified_ids=list(verified),verified_count=len(verified),failed_preserved=dict(failed),
            captured_utc=datetime.now(timezone.utc).isoformat())
        write(directory/'status.json',state)
    def stage(case,phase):
        identity=case['job']['id'];attempt=directory/'attempts'/identity/phase
        proof=Path(runtime['artifact_root'])/identity/('native/native_replay.json' if phase=='native' else 'independent/review.json')
        receipt=attempt/'guard_receipt.json'
        if receipt.exists():
            previous=read(receipt)
            if (previous['passed'] and previous['exit_code']==0 and not previous['resource_guard_aborted']
                    and sha(proof)==previous['proof_sha256'] and read(proof)['passed']):
                _,_,binding=check_source(runtime,case)
                if read(proof)['source_binding']!=binding:raise ValueError('Existing replay source binding changed')
                return True
            raise ValueError('Failed prior full MOMENT stage preserved; no automatic retry')
        if attempt.exists() or proof.exists() or (phase=='native' and proof.parent.exists()):
            raise ValueError('Incomplete prior full MOMENT stage preserved for diagnosis')
        attempt.mkdir(parents=True,exist_ok=False)
        command=[str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',MODULE,'--config',runtime['_config_path'],'--stage',phase,'--job-id',identity]
        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command);write(attempt/'launch.json',handle)
            update(state='running_full_MOMENT_'+phase,active_pid=child.pid,active_create_time=handle['create_time'],job=identity)
            code,usage=capped_wait(child,runtime,api,update)
        passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof).get('passed') is True
        write(receipt,dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.exists() else None,exact_child_handle=handle))
        return passed
    try:
        while len(verified)+len(failed)<len(cases):
            for identity,case in cases.items():
                if identity in verified or identity in failed:continue
                if not (ROOT/case['job']['output_directory']/'result.json').exists():continue
                try:
                    def waiting(value):update(dict(value,active_pid=None,job=identity))
                    with api['resource_slot'](ROOT/runtime['resource_lock'],runtime['settings'],waiting,poll_seconds=10):
                        if not stage(case,'native') or not stage(case,'independent'):
                            raise ValueError('Full guarded MOMENT stage failed; preserve without retry')
                    verified.append(identity)
                except Exception as error:failed[identity]=str(error)
                update(state='scanning_actual_full_MOMENT_results',active_pid=None,job=None)
            update(state='waiting_actual_full_MOMENT_results',active_pid=None,job=None);time.sleep(10)
        update(state='all78_full_MOMENT_attempts_terminal_other_original_experiments_remain',active_pid=None,job=None)
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--stage',choices=['native','independent']);parser.add_argument('--job-id')
    args=parser.parse_args();runtime=read(ROOT/args.config);cases=verify(runtime);runtime['_config_path']=args.config
    if args.probe_only:
        print(json.dumps(dict(passed=True,original_jobs=72,constant_fit_controls=6,unique_jobs=len(cases),
            existing_completed_source_results=sum((ROOT/c['job']['output_directory']/'result.json').exists() for c in cases.values()),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            full_checkpoint_replays_complete=0,all_original_experiments_complete=False)))
    elif args.stage:(native_worker if args.stage=='native' else review_worker)(runtime,cases[args.job_id])
    else:controller(runtime,cases)


if __name__=='__main__':main()
