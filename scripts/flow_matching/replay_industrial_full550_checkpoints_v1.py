"""Replay all original industrial CPU checkpoints without refitting.

Keep the whole550-case scope. Missing or failed sources get no completion
credit. Native inference and a separate complete-array/metric review each use
the existing32GiB serial CPU policy, original2threads and exact score bytes.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching import audit_industrial_full550_independent_metrics_v1 as point
from scripts.flow_matching import audit_industrial_three_recovery_independent_metrics_v1 as recovery
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write

MODULE='scripts.flow_matching.replay_industrial_full550_checkpoints_v1'
TEST='tests/test_fm_industrial_full550_checkpoint_replays_v1.py'
CONFIG='configs/reproducibility/fm_industrial_full550_checkpoint_replays.v1.json'
POLICY='configs/runtime/fm_bounded_remaining_cpu_queues.v3.json'
PRIOR='projects/flow_matching_research/results/2026-10-16/industrial_full5_466_canonical_saved_score_metric_audits_v1'
ENTRIES={
    'iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector':'objective',
    'iia_benchmark.models.fm_objective_stable:StableCFMObjectiveDetector':'objective',
    'iia_benchmark.models.fm_reflow:ReflowCFMDetector':'reflow',
    'iia_benchmark.models.fm_legacy_window_adapter:LegacyWindowAdapter':'legacy'}


def registration():
    pc=read(ROOT/point.CONFIG);q=read(ROOT/pc['queue']);policy=read(ROOT/POLICY)
    files={MODULE.replace('.','/')+'.py',TEST,POLICY,point.CONFIG,recovery.CONFIG,point.QUEUE,
        point.MODULE.replace('.','/')+'.py',recovery.MODULE.replace('.','/')+'.py',
        PRIOR+'/all466_proof_source_and_guard_bindings.json',PRIOR+'/packet_integrity.json',
        'scripts/flow_matching/run_industrial_tsad_experiment.py','scripts/flow_matching/model_registry.py',
        'scripts/flow_matching/run_objective_full440_checkpoint_replay_v1.py',
        'scripts/flow_matching/reflow_checkpoint_evidence_v1.py','scripts/flow_matching/objective_replay_math_v1.py',
        'scripts/flow_matching/legacy_baseline_checkpoint_math_v1.py',
        'scripts/flow_matching/replay_original_full105_baselines_v1.py',
        'scripts/flow_matching/audit_industrial_SB_extended_full_replay_v1.py',
        'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py'}
    files.update(p['path'] for job in q['jobs'] for p in job['frozen_files'])
    return dict(queue=point.QUEUE,required_original_CPU_cases=550,original_GPU_cases=850,
        original_global_scope=pc['original_global_scope'],all_required_ids=[j['id'] for j in q['jobs']],
        settings=policy['settings'],resource_lock=policy['resource_lock'],source_cpu_threads=q['evaluation']['cpu_threads'],
        state_root='experiments/runs/fm_industrial_full550_checkpoint_replays_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/industrial_full550_checkpoint_replays_v1',
        source_point_audit=point.CONFIG,three_source_recoveries=recovery.CONFIG,
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(files)],
        score_comparison='exact_dtype_shape_full_payload_bytes',independent_metric_tolerance=pc['numerical_metric_tolerance'],
        independent_probability_path='source_FP32_operation_program_with_ideal_FP64_residual_diagnostics',
        independent_legacy_scoring='source_FP32_energy_program_and_FP64_output_residual_checks',
        independent_neural_kernels=False,independent_historical_training_updates=False,
        source_historical_resource_gate_certification=False,independent_bootstrap=False,independent_VUS=False,
        independent_affiliation=False,new_training_seed_slots=0,new_optimizer_updates=0,
        new_model_training_results=0,full_TAB_harness_equivalence=False,paper_equivalence=False,
        all_original_experiments_complete=False)


def verify(c):
    if c!=registration():raise ValueError('All550 original identities, original2threads and unchanged32GiB serial policy required')
    q,manifest=point.verify(read(ROOT/point.CONFIG));_,_,_,recoveries=recovery.verify(read(ROOT/recovery.CONFIG))
    jobs={j['id']:j for j in q['jobs']};executions=dict(jobs)
    for case in recoveries:executions[case['original_id']]=recovery.validate_identity(case,jobs)
    if len(jobs)!=550 or len({j['model_config'] for j in jobs.values()})!=22 or q['evaluation']['cpu_threads']!=2:
        raise ValueError('Every original industrial CPU model/data/seed record required')
    for j in jobs.values():
        model=read(ROOT/j['model_config'])
        if model['entrypoint'] not in ENTRIES:raise ValueError('Explicit original architecture required')
        parameters=model['parameters']
        window=parameters['parameters']['window'] if ENTRIES[model['entrypoint']]=='legacy' else parameters['window']
        if (j['window']!=window or window<=0 or j['seed'] not in range(1103,1108)
                or j['parameter_overrides']!=dict(seed=j['seed'],device='cpu')):
            raise ValueError('Explicit complete source architecture and CPU scoring protocol required')
    return q,manifest,jobs,executions


def source_binding(c,jobs,executions,cid):
    """Require a real source and its complete independently checked scores."""
    job=executions[cid];path=ROOT/job['output_directory']/'result.json'
    if not path.is_file():return None
    source=read(path)
    pc=read(ROOT/point.CONFIG)
    if job['id']!=cid:
        prior=read(ROOT/PRIOR/'all466_proof_source_and_guard_bindings.json')['records'][cid]
        report_path=Path(prior['independent_report_container']);guard_path=Path(prior['guard_path'])
        if not report_path.is_file() or not guard_path.is_file():return None
        report=read(report_path)['records'][cid]['independent_report']
    else:
        report_path=Path(pc['artifact_root'])/'conditions'/(cid+'.json')
        guard_path=ROOT/pc['state_root']/'attempts'/cid/'receipt.json'
        if not report_path.is_file() or not guard_path.is_file():return None
        report=read(report_path)
    guard=read(guard_path)
    proof_digest=guard.get('proof_sha256') if job['id']!=cid else guard.get('report_sha256')
    if (not report['passed'] or report['canonical_id']!=job['id'] or not guard['passed']
            or guard['exit_code']!=0 or guard['resource_guard_aborted'] or proof_digest!=sha(report_path)
            or source['status']!='completed' or source['job']!=job or source['id']!=job['id']
            or source['dataset']!=jobs[cid]['dataset'] or source['seed']!=jobs[cid]['seed']
            or report['source_result_sha256']!=sha(path)
            or source['checkpoint_sha256']!=report['checkpoint_sha256']
            or source['scores_sha256']!=report['scores_sha256']):
        raise ValueError('Exact complete canonical/recovered source and independent score proof required')
    for pin in job['frozen_files']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Original frozen source changed: '+pin['path'])
    for name,key in [('model.pt','checkpoint_sha256'),('scores.npz','scores_sha256')]:
        if sha(path.parent/name)!=report[key]:raise ValueError('Full source artifact changed')
    binding=dict(canonical_id=cid,actual_execution_id=job['id'],source_result_path=str(path),
        source_result_sha256=sha(path),checkpoint_sha256=report['checkpoint_sha256'],scores_sha256=report['scores_sha256'],
        point_report_path=str(report_path),point_report_sha256=sha(report_path),point_guard_path=str(guard_path),
        point_guard_sha256=sha(guard_path),source_historical_resource_gate_certified=False)
    return job,source,report,binding


def native_align(scores,starts,length,window):
    import numpy as np
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import native_align as matrix_align
    values=np.asarray(scores,dtype=float)
    if values.shape==(len(starts),):values=np.broadcast_to(values[:,None],(len(starts),window))
    return matrix_align(values,starts,length,window)


def restore(checkpoint,job,configuration,features):
    if configuration['entrypoint'] not in ENTRIES:raise ValueError('Explicit original architecture required')
    family=ENTRIES[configuration['entrypoint']]
    if family=='legacy':
        from scripts.flow_matching.legacy_baseline_checkpoint_math_v1 import restore_model,checked_score
        model=restore_model(checkpoint,job,configuration,features)
    else:
        from scripts.flow_matching.reflow_checkpoint_evidence_v1 import restore_model
        from scripts.flow_matching.objective_replay_math_v1 import checked_score
        model=restore_model(checkpoint,job,configuration,features)
    def forbidden(*args,**kwargs):raise RuntimeError('No fit or optimizer update during native replay')
    model.fit=forbidden
    if family=='legacy':model._delegate.fit=forbidden
    score,observation=checked_score(model)
    return model,score,observation


def check_weights(model,checkpoint):
    import torch
    from scripts.flow_matching.reflow_checkpoint_evidence_v1 import tensor_bytes
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    actual={name:module.state_dict() for name,module in vars(model).items() if isinstance(module,torch.nn.Module)}
    if set(actual)!=set(checkpoint['modules']):raise ValueError('Every persisted module and Reflow evidence buffer required')
    bindings={}
    for name,state in actual.items():
        expected=checkpoint['modules'][name]
        if set(state)!=set(expected):raise ValueError('Every original tensor key required')
        bindings[name]={}
        for key,value in state.items():
            tensor_bytes(value,expected[key])
            if not torch.isfinite(value).all():raise ValueError('Every persisted tensor must remain finite')
            bindings[name][key]=array_digest(value.detach().cpu().numpy())
    return bindings


def validate_observation(coverage,observation,configuration):
    if ENTRIES[configuration['entrypoint']]=='legacy':
        from scripts.flow_matching.replay_original_full105_baselines_v1 import validate_observation as original
        original(coverage,observation,configuration);return
    p=configuration['parameters'];windows=sum(r['windows'] for r in coverage)
    calls=sum(math.ceil(r['windows']/p['batch_size']) for r in coverage)*p['monte_carlo_samples']*len(p['score_times'])
    if (observation['score_calls']!=len(coverage) or observation['windows']!=windows
            or observation['window_points']!=windows*p['window'] or observation['path_calls']!=calls
            or observation['forward_calls']!=calls or not observation['source_FP32_path_program_checked']):
        raise ValueError('Every original window/batch/MC sample/time and path forward required')


def paths(c,cid,phase):
    if phase not in ['native','review']:raise ValueError('Explicit native or independent review phase required')
    return ROOT/c['state_root']/'attempts'/cid/phase,Path(c['artifact_root'])/cid/phase


def replay_groups(dataset,window,score,data,saved):
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import roles,native_windows,require_array_equal
    payload,arrays,coverage={},{},[]
    for role,g in roles(dataset):
        prefix=g['array_prefix'];input_key=prefix+('_test' if role=='test' else '_values')
        values=data[input_key]
        if values.shape!=(g['points'],g['features']):raise ValueError('Every native source coordinate required')
        if role=='test' and prefix+'_labels' not in data.files:raise ValueError('Every native test label required')
        windows,starts=native_windows(values,window);key=prefix+'_'+role
        observed=native_align(score(windows),starts,len(values),window)
        arrays[key]=require_array_equal(observed,saved[key],key);payload[key]=observed
        for suffix in point.SUFFIXES:
            key=prefix+suffix
            if key in data.files:
                arrays[key]=require_array_equal(data[key],saved[key],key);payload[key]=data[key]
        coverage.append(dict(group=g['group_id'],role=role,points=g['points'],features=g['features'],windows=len(windows)))
    if set(payload)!=set(saved.files):raise ValueError('Every original score and metadata array must be present')
    return payload,arrays,coverage


def worker(c,cid,phase):
    q,manifest,jobs,executions=verify(c);ready=source_binding(c,jobs,executions,cid)
    if ready is None:raise ValueError('Wait for complete real source and independent point-score audit')
    job,source,point_proof,binding=ready
    dataset=next(d for d in manifest['datasets'] if d['id']==job['dataset'])
    if sha(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Full original prepared data required')
    for pin in dataset['source_files']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Original raw data changed')
    _,out=paths(c,cid,phase);out.mkdir(parents=True,exist_ok=False)
    if phase=='review':
        review(c,cid,job,source,dataset,binding,out);return
    import numpy as np
    import torch
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import roles,native_windows,require_array_equal
    from scripts.flow_matching.run_objective_full440_checkpoint_replay_v1 import producer_environment
    environment=producer_environment(source);torch.set_num_threads(c['source_cpu_threads'])
    if torch.cuda.is_initialized():raise ValueError('Native industrial CPU replay must not initialize CUDA')
    checkpoint=torch.load(ROOT/job['output_directory']/'model.pt',map_location='cpu',weights_only=True)
    configuration=read(ROOT/job['model_config']);groups=roles(dataset);features={g['features'] for _,g in groups}
    if len(features)!=1:raise ValueError('Full original feature geometry required')
    model,score,observation=restore(checkpoint,job,configuration,features.pop())
    count=sum(p.numel() for m in vars(model).values() if isinstance(m,torch.nn.Module) for p in m.parameters())
    if count!=source['parameter_count']:raise ValueError('Complete original parameter count required')
    before=check_weights(model,checkpoint);payload={};arrays={};coverage=[];reflow=None
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as data,np.load(ROOT/job['output_directory']/'scores.npz',allow_pickle=False) as saved:
        if ENTRIES[configuration['entrypoint']]=='reflow':
            from scripts.flow_matching.reflow_checkpoint_evidence_v1 import verify_all_couplings
            descriptor=dict(job,total_training_epochs=source['epoch_audit']['declared_epochs'],additional_pair_generation_compute=True)
            reflow=verify_all_couplings(model,descriptor,source,source['unique_fit_windows'])
        payload,arrays,coverage=replay_groups(dataset,job['window'],score,data,saved)
        for row in coverage:print(json.dumps(dict(canonical_id=cid,stage='complete_native_group_reinferred',**row)),flush=True)
    validate_observation(coverage,observation,configuration)
    if check_weights(model,checkpoint)!=before:raise ValueError('Native scoring changed a source parameter or evidence buffer')
    score_path=out/'scores.npz'
    with score_path.open('xb') as f:np.savez_compressed(f,**payload)
    with np.load(score_path,allow_pickle=False) as reloaded:
        for key,value in payload.items():require_array_equal(reloaded[key],value,key)
    if source_binding(c,jobs,executions,cid)[-1]!=binding:raise ValueError('Original source inputs changed during replay')
    write(out/'report.json',dict(passed=True,canonical_id=cid,source_binding=binding,configuration_sha256=sha(ROOT/CONFIG),
        producer_environment=environment,source_cpu_threads=c['source_cpu_threads'],parameter_count=count,
        every_checkpoint_tensor_restored_bitwise=True,all_native_arrays_equal_bitwise=True,all_original_roles_reinferred=True,
        coverage=coverage,arrays=arrays,complete_checkpoint_tensor_bindings=before,score_formula_observation=observation,
        reflow_all_teacher_pairs_and_full_ODE_evidence=reflow,independent_neural_kernels=False,
        training_optimizer_updates=0,fit_calls=0,new_training_seed_slots=0,scores_path=str(score_path),scores_sha256=sha(score_path),
        original_global_scope=c['original_global_scope'],all_original_experiments_complete=False))


def review(c,cid,job,source,dataset,binding,out):
    """Fresh process: every native score/metadata byte plus independent metrics."""
    import hashlib
    import zipfile
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import roles,strict_metrics
    _,native=paths(c,cid,'native');report=read(native/'report.json');proof=native/'scores.npz'
    guard=read(paths(c,cid,'native')[0]/'guard_receipt.json')
    if (not report['passed'] or report['source_binding']!=binding or not guard['passed']
            or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(native/'report.json')
            or sha(proof)!=report['scores_sha256'] or report['configuration_sha256']!=sha(ROOT/CONFIG)):
        raise ValueError('Exact guarded complete native replay is required')
    payload={};arrays={}
    with zipfile.ZipFile(proof) as actual,zipfile.ZipFile(ROOT/job['output_directory']/'scores.npz') as expected:
        if len(actual.namelist())!=len(set(actual.namelist())) or set(actual.namelist())!=set(expected.namelist()):
            raise ValueError('All unique source/native array names required')
        for role,g in roles(dataset):
            key=g['array_prefix']+'_'+role
            for name in [key,*[g['array_prefix']+suffix for suffix in point.SUFFIXES]]:
                if name+'.npy' not in actual.namelist():
                    if name==key:raise ValueError('Every role score array required')
                    continue
                value=point.archive_array(actual,name)
                if value!=point.archive_array(expected,name):raise ValueError('Every native score/metadata byte must match')
                header,data=value;arrays[name]=dict(dtype=header['descr'],shape=list(header['shape']),sha256=hashlib.sha256(data).hexdigest())
                if name==key:payload[name]=point.vector(header,data,g['points'],'<f8')
                elif name.endswith('_labels'):payload[name]=point.vector(header,data,g['points'],'|u1')
        if set(arrays)!={x[:-4] for x in actual.namelist()}:raise ValueError('No original or native array omitted')
    metrics=strict_metrics(dataset,payload,source['metrics']['strict'],c['independent_metric_tolerance'],source['checkpoint_sha256'])
    write(out/'report.json',dict(passed=True,canonical_id=cid,source_binding=binding,native_report_sha256=sha(native/'report.json'),
        native_scores_sha256=sha(proof),all_original_native_score_metadata_bytes_equal=True,full_array_bindings=arrays,
        independent_all_three_strict_metrics=metrics,independent_bootstrap=False,independent_VUS=False,independent_affiliation=False,
        fit_calls=0,training_optimizer_updates=0,new_training_seed_slots=0,
        original_global_scope=c['original_global_scope'],all_original_experiments_complete=False))


def command(cid,phase):
    paths(dict(state_root='',artifact_root=''),cid,phase)
    return [sys.executable,'-B','-X','utf8','-u','-m',MODULE,'--case',cid,'--phase',phase]


def controller(c,jobs,executions):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Industrial full550 native controller already owned')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        verified_ids=[],native_only_ids=[],failed_preserved={},required_original_CPU_cases=550,
        all_original_experiments_complete=False)
    if (base/'status.json').exists():
        old=read(base/'status.json')
        if old.get('active_pid') and psutil.pid_exists(old['active_pid']):
            owner.close();raise RuntimeError('Preserve potentially live inherited numerical worker')
        state.update({k:old[k] for k in ['verified_ids','native_only_ids','failed_preserved']})
    def update(v):
        state.update(v,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while True:
            for cid in jobs:
                if cid in state['verified_ids'] or cid in state['failed_preserved']:continue
                if source_binding(c,jobs,executions,cid) is None:continue
                for phase in ['native','review']:
                    if cid in state['failed_preserved']:break
                    if phase=='native' and cid in state['native_only_ids']:continue
                    attempt,out=paths(c,cid,phase)
                    if attempt.exists() or out.exists():
                        state['failed_preserved'][cid]=dict(phase=phase,reason='Existing unqualified attempt preserved;no retry')
                        update(dict(state='existing_attempt_preserved'));break
                    update(dict(active_case=cid,phase=phase,source_ready=True))
                    with api['resource_slot'](ROOT/c['resource_lock'],c['settings'],update,poll_seconds=10):
                        attempt.mkdir(parents=True,exist_ok=False);cmd=command(cid,phase)
                        with (attempt/'stdout.log').open('xb') as stdout,(attempt/'stderr.log').open('xb') as stderr:
                            child=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            update(dict(state='full_industrial_'+phase,active_case=cid,phase=phase,active_pid=child.pid,
                                active_create_time=psutil.Process(child.pid).create_time(),actual_command=cmd))
                            code,usage=capped_wait(child,c,api,update)
                        proof=out/'report.json';passed=code==0 and not usage['resource_guard_aborted'] and proof.is_file() and read(proof)['passed']
                        write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,proof_sha256=sha(proof) if proof.is_file() else None))
                        if passed:state['verified_ids' if phase=='review' else 'native_only_ids'].append(cid)
                        else:state['failed_preserved'][cid]=dict(phase=phase,exit_code=code,resource_guard_aborted=usage['resource_guard_aborted'])
                        update(dict(active_pid=None))
            update(dict(state='waiting_actual_full_industrial_sources_or_native_review'))
            if len(set(state['verified_ids'])|set(state['failed_preserved']))==550:return
            time.sleep(10)
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--probe-only',action='store_true')
    p.add_argument('--case');p.add_argument('--phase',choices=['native','review']);a=p.parse_args()
    if a.register:
        with (ROOT/CONFIG).open('x',encoding='utf-8') as f:json.dump(registration(),f,indent=2)
        return
    c=read(ROOT/CONFIG);q,m,j,e=verify(c)
    if a.probe_only:
        print(json.dumps(dict(passed=True,required_cases=550,original_model_configs=22,source_cpu_threads=2,
            source_completed_snapshot=466,settings=c['settings'],resource_lock=c['resource_lock'],
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)));return
    if a.case:
        if a.case not in j or a.phase is None:raise ValueError('An exact original case and explicit phase are required')
        worker(c,a.case,a.phase);return
    if a.phase:raise ValueError('Phase requires an exact case')
    controller(c,j,e)


if __name__=='__main__':main()
