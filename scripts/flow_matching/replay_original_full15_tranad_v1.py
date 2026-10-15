"""Reload all15 fullraw FP64 TranAD checkpoints and review every label view.

This is one native-replay component of the full873 experiment scope. Other
families, original papers, source failures and full TAB remain separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as scalar

ROOT,read,sha,write=scalar.ROOT,scalar.read,scalar.sha,scalar.write
MODULE='scripts.flow_matching.replay_original_full15_tranad_v1'
CONFIG='configs/reproducibility/fm_original_full15_tranad_checkpoint_replays.2026-10-16_v1.json'
TEST='tests/test_fm_original_full15_tranad_replays_v1.py'
POLICY='configs/reproducibility/fm_tranad_float64_SMAP1103_constant_fit_full_replay.native.2026-10-15_v1.json'
REPORTS=dict(native='native_replay.json',review='all_label_view_independent_review.json')


def registration():
    old=read(ROOT/scalar.CONFIG);scalar.verify(old)
    cases=[c for c in old['cases'] if c['family']=='tranad64'];guard=read(ROOT/POLICY)
    if len(cases)!=15 or {(c['dataset'],c['seed']) for c in cases}!={(d,s) for d in ['smap','msl','smd'] for s in range(1103,1108)}:
        raise ValueError('All15original fullraw TranAD cases required')
    files={scalar.CONFIG,POLICY,MODULE.replace('.','/')+'.py',TEST,
        'scripts/flow_matching/audit_tranad_float64_constant_fit_complete_v1.py',
        'scripts/flow_matching/run_tsad_experiment.py','scripts/flow_matching/anomaly_transformer_independent_math_v1.py',
        'src/iia_benchmark/models/fm_tranad_reference_float64_v1.py','src/iia_benchmark/models/mtsad_detectors.py'}
    files.update(p['path'] for p in old['source_receipts'])
    return dict(schema_version=1,upstream_scalar_configuration=scalar.CONFIG,source_execution=old['source_execution'],cases=cases,
        required_family_native_cases=15,required_family_label_views=35,original_all_family_cases=873,original_all_label_views=2037,
        settings=guard['settings'],resource_lock=guard['resource_lock'],controller_python=old['controller_python'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        expected_source_model_entrypoint='iia_benchmark.models.fm_tranad_reference_float64_v1:ReferenceFloat64TranADAdapter',
        model_scoring='unchanged_source_ReferenceFloat64TranADDetector.score',native_comparison='exact_dtype_shape_content_bytes',
        paper_and_data_citations=old['paper_and_data_citations'],original_scope=old['original_scope'],
        artifact_root='F:/aicoding/IIA_Data/experiments/original_full15_tranad_checkpoint_replays_v1',
        state_root='experiments/runs/fm_original_full15_tranad_checkpoint_replays_v1',
        boundary='All15fullraw TranAD cases;3datasets*5seeds and35label views. No fresh fitting or training. '
            'Require actual full109data,successful exact source guard and full scalar label audit. '
            'Restore every finite tensor with exact original dtype/shape and complete strict keys. '
            'Every original TRAIN/validation/TEST point is reinferred using unchanged float64 source code, '
            'all end-aligned tail windows and original2CPUthreads;every score/label byte must match. '
            'Second fresh process compares all original/native arrays and raw primary labels, '
            'recomputes every existing source metric and all35label-view event/bootstrap/grid controls. '
            'No neural attention math rewrite,training-update trace,five-seed paired statistics or fullTAB/author '
            'padding/preprocess/POT certification is inferred. Other858native cases and all original papers remain required. '
            'Existing readonly-native resource policy/lock and all original model/data/seed budgets remain.')


def verify(config):
    if config!=registration():raise ValueError('All15native cases/35views and unchanged complete source/resource policies required')
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
    p=model['parameters']
    if (case['family']!='tranad64' or model['entrypoint']!=config['expected_source_model_entrypoint']
            or model['precision']!='float64_fit_and_score' or p['method']!='tranad' or p['epochs']!=5
            or p['parameters']!=dict(window=10,d_ff=16,dropout=.1,learning_rate=.01,weight_decay=1e-5,
                scheduler_step=5,scheduler_gamma=.9,batch_size=128)
            or case['window']!=10 or job['parameter_overrides']!=dict(seed=case['seed'],device='cpu')
            or p['device']!='cpu' or queue['evaluation']['cpu_threads']!=2):
        raise ValueError('Exact full source FP64 architecture,window,capacity,budget and seed required')


def producer_environment(result):
    import numpy as np,torch
    actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device='cpu')
    if actual!=result['environment']:raise ValueError('Exact original FP64 source numerical environment required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise ValueError('Readonly CPU model replay must not initialize CUDA')
    return actual


def check_restored_weights(detector,checkpoint):
    import torch
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    expected=checkpoint['modules']['_model'];actual=detector._model.state_dict()
    if set(actual)!=set(expected):raise ValueError('Every complete original parameter and buffer key required')
    for key,observed in actual.items():
        wanted=expected[key]
        if (observed.dtype!=wanted.dtype or observed.shape!=wanted.shape or not torch.isfinite(observed).all()
                or not torch.equal(observed,wanted)):
            raise ValueError('Every restored original tensor must retain exact dtype,shape and content')
    return {k:array_digest(v.detach().cpu().numpy()) for k,v in actual.items()}


def native_worker(config,case):
    import numpy as np,torch
    from scripts.flow_matching.audit_tranad_float64_constant_fit_complete_v1 import restore_model
    from scripts.flow_matching.run_tsad_experiment import covering_windows,align_scores
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import require_identical,array_digest
    old=ready_scalar(config,case)
    if old is None:raise ValueError('Actual full successful scalar audit required before native inference')
    queue,job,result,data,binding=scalar.source_binding(config,case);environment=producer_environment(result)
    model_config=read(ROOT/job['model_config']);validate_model(config,case,queue,job,model_config)
    _,output=paths(config,case,'native');output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(ROOT/job['output_directory']/'model.pt',map_location='cpu',weights_only=True)
    features={e['features'] for e in data['entities']}
    if len(features)!=1:raise ValueError('Full original feature geometry must agree across all entities')
    detector=restore_model(checkpoint,job,model_config,features.pop(),result['parameter_count'])
    detector._model.requires_grad_(False);before=check_restored_weights(detector,checkpoint)
    def forbid_fit(*args,**kwargs):raise RuntimeError('No model fitting is allowed during checkpoint replay')
    detector.fit=forbid_fit
    payload={};coverage=[]
    with np.load(ROOT/data['input_npz'],allow_pickle=False) as inputs,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        for entity in data['entities']:
            prefix=entity['array_prefix']
            for role in ['train','validation','test']:
                key=prefix+'_'+role;values=inputs[key]
                if values.ndim!=2 or values.shape[1]!=entity['features'] or not np.isfinite(values).all():
                    raise ValueError('Complete original finite model input required')
                windows,starts=covering_windows(values,job['window'])
                observed=align_scores(detector.score(windows),starts,len(values),job['window'])
                require_identical(observed,saved[key]);payload[key]=observed
                coverage.append(dict(entity=entity['entity_id'],role=role,points=len(values),windows=len(windows)))
            key=prefix+'_labels';require_identical(inputs[key],saved[key]);payload[key]=inputs[key]
            print(json.dumps(dict(id=case['id'],stage='full_entity_checkpoint_reinferred',entity=entity['entity_id'])),flush=True)
        if set(payload)!=set(saved.files):raise ValueError('Every complete original score/label array required')
    after=check_restored_weights(detector,checkpoint)
    if after!=before:raise ValueError('Native replay changed an original model tensor')
    score_path=output/'scores.npz'
    with score_path.open('xb') as stream:np.savez_compressed(stream,**payload)
    with np.load(score_path,allow_pickle=False) as saved:
        if set(saved.files)!=set(payload):raise ValueError('Every serialized native output must reload')
        for key,value in payload.items():require_identical(saved[key],value)
    if scalar.source_binding(config,case)[-1]!=binding or ready_scalar(config,case)!=old:
        raise ValueError('Original inputs,weights,scores,labels and upstream evidence must remain unchanged')
    write(output/REPORTS['native'],dict(passed=True,case=case,configuration_sha256=sha(ROOT/CONFIG),source_binding=binding,
        upstream_scalar_evidence=old,actual_producer_environment=environment,parameters=result['parameter_count'],
        complete_restored_state_tensors=before,all_model_tensors_identical_before_and_after=True,
        full_native_coverage=coverage,all_array_dtype_shape_content_bytes_identical=True,
        every_saved_native_array_serialized_and_reloaded=True,source_arrays={k:array_digest(v) for k,v in payload.items()},
        scores_path=str(score_path),scores_sha256=sha(score_path),additional_fit_calls=0,additional_optimizer_updates=0,
        CUDA_context_initialized=torch.cuda.is_initialized(),original_training_update_counter_independently_observed=False,
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
        independent_attention_kernel_implementation=False,full_TAB_or_author_equivalence=False,all_original_experiments_complete=False))


def controller(config,cases):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write as atomic
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full15TranAD native replay owner already live')
    if (base/'status.json').exists():owner.close();raise FileExistsError('Preserve prior complete/partial replay attempt')
    completed=[];failed={};source_failed={};state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        required_family_native_cases=15,required_family_label_views=35,active_pid=None,all_original_experiments_complete=False)
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
            write(attempt/'launch.json',handle);update(state='running_full_TranAD_'+phase,active_pid=child.pid,active_job=case['id'])
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
                    completed.append(identity);update(state='between_full_TranAD_replays',active_pid=None,active_job=None)
                except Exception as error:failed[identity]=str(error);update(state='replay_failure_preserved',active_pid=None,active_job=None)
            update(state='waiting_actual_original_training_and_scalar_audits',active_pid=None,active_job=None);time.sleep(10)
        update(state='all15family_replay_outcomes_terminal_other858native_cases_and_papers_remain',active_pid=None,active_job=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--job-id');parser.add_argument('--stage',choices=list(REPORTS));args=parser.parse_args()
    if args.register:write(ROOT/CONFIG,registration());print(json.dumps(dict(configuration=CONFIG,cases=15,label_views=35)));return
    config=read(ROOT/CONFIG);cases=verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,cases=15,label_views=35,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.job_id:
        if args.stage is None:raise ValueError('An explicit native/review stage is required')
        case=next(c for c in cases if c['id']==args.job_id)
        (native_worker if args.stage=='native' else review_worker)(config,case)
    else:controller(config,cases)


if __name__=='__main__':main()
