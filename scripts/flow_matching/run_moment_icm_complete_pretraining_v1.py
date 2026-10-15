"""Guarded full PILE one-epoch Tiny independent/ICM training; never shrink scope.

Cold metadata waits for actual complete input and mask proofs. Fresh numerical
workers use the tested pinned source body and save full final training state.
Independent native model replay and downstream performance remain separate.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from types import SimpleNamespace
from scripts.flow_matching import audit_moment_icm_full_training_masks_v1 as masks
from scripts.flow_matching import moment_icm_masked_training_v1 as train
from scripts.flow_matching import moment_icm_training_artifacts_v1 as artifacts
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=masks.ROOT,masks.read,masks.sha,masks.write
MODULE='scripts.flow_matching.run_moment_icm_complete_pretraining_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_full_PILE_two_model_pretraining.2026-10-15_v1.json'
POLICY='configs/runtime/fm_saits_saved_epoch271_full210_recovery.v2.json'


def ready():
    binding=masks.ready_binding()
    c=read(ROOT/masks.CONFIG);path=Path(c['artifact_root'])/masks.REPORT;guard=ROOT/c['state_root']/'guard_receipt.json'
    if binding is None or not path.exists() or not guard.exists():return None
    proof,g=read(path),read(guard)
    if (not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/masks.CONFIG) or proof['sources']!=2086
            or proof['epoch_items']!=80850189 or proof['incompatible_training_items']!=0
            or not proof['every_original_random_mask_draw_has_nonempty_RevIN_inputs']
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(path)):
        raise ValueError('Actual full successful mask audit and original resource receipt required')
    return binding


def registration():
    body=read(ROOT/masks.TRAIN_BODY);masks.verify(read(ROOT/masks.CONFIG))
    policy=read(ROOT/POLICY)
    # Data preparation's 3GiB process cap is not the neural GPU job policy.
    # Preserve the complete existing neural policy rather than imposing an
    # unmeasured smaller cap that already excludes live original GPU training.
    settings=dict(policy['settings'])
    files=[masks.TRAIN_BODY,masks.CONFIG,POLICY,MODULE.replace('.','/')+'.py',
        'scripts/flow_matching/moment_icm_training_artifacts_v1.py','tests/test_fm_moment_icm_training_artifacts_v1.py',
        'tests/test_fm_moment_icm_complete_pretraining_v1.py','scripts/flow_matching/run_grasp_protocol_queue.py',
        'scripts/flow_matching/run_anomaly_transformer_cold_resource_entry_v1.py',
        'scripts/flow_matching/run_cached_resource_monitor_controller_v1.py',
        'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py','scripts/flow_matching/heavy_resources.py',
        'scripts/flow_matching/heavy_scheduler_v2.py',body['original_source_root']+'/configs/pretraining/pretrain.yaml']
    files += [p['path'] for p in body['source_receipts']]+[p['path'] for p in read(ROOT/masks.CONFIG)['source_receipts']]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=body['paper_citation'],data_citation=body['data_citation'],
        training_body_configuration=masks.TRAIN_BODY,mask_audit_configuration=masks.CONFIG,original_resource_policy=POLICY,
        python=body['python'],settings=settings,resource_lock=policy['resource_lock'],
        jobs=[dict(id='moment_tiny_'+variant+'_full_PILE_seed13',mixer=variant,seed=13,epochs=1,sources=2086,
            expected_items=80850189,expected_batches=39479,batch_size=2048,scientific=True) for variant in ['independent','icm']],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_full_PILE_two_model_pretraining_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_full_PILE_two_model_pretraining_v1',
        boundary=('All2086sources/all80,850,189items/39,479full+partialbatches/seed13/oneepoch per '
            'TinyindependentandICM model. SourcepretrainingYAML recipe except paperTiny256/4x256/4head/1024FF '
            'architecture anddeclared target-validity/epoch/checkpointcontext adaptations. Same SAITS shared '
            'GPUlock/12GiBphysical/28GiBcommit/16GiBemergencyhost/16GiBGPUadmit/4GiBGPUemergency and '
            'neural resource policy unchanged,never bypassexisting queues. Separate maskpreflight '
            'retains its original3GiBdataworker caps. Wait actual fullinput+maskproofs '
            'before any CUDA/NN import in worker. Full final state andperiodic sourcecallbacks saved exactly; '
            'sourceexternal evaluations explicitlypending,notcompleted. No silent retry/resume,budget/precision/'
            'batchreduction,rawmutation,independentmodelreplay,downstream performance orhistoricalequivalence.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full pretraining dependency changed: '+pin['path'])
    body=read(ROOT/masks.TRAIN_BODY);policy=read(ROOT/POLICY)
    expected=registration_settings(policy)
    jobs=[dict(id='moment_tiny_'+v+'_full_PILE_seed13',mixer=v,seed=13,epochs=1,sources=2086,
        expected_items=80850189,expected_batches=39479,batch_size=2048,scientific=True) for v in ['independent','icm']]
    if (config['jobs']!=jobs or config['settings']!=expected or config['resource_lock']!=policy['resource_lock']
            or config['python']!=body['python'] or config['training_body_configuration']!=masks.TRAIN_BODY
            or config['mask_audit_configuration']!=masks.CONFIG or config['original_resource_policy']!=POLICY):
        raise ValueError('Both full models, complete oneepoch budgets and original guarded GPU lane required')
    masks.verify(read(ROOT/masks.CONFIG))
    return body


def registration_settings(policy):
    return dict(policy['settings'])


def numerical_environment():
    names={'torch':'torch','transformers':'transformers','numpy':'numpy','einops':'einops','pandas':'pandas',
        'scipy':'scipy','scikit_learn':'scikit-learn','PyYAML':'PyYAML'}
    return dict(python=platform.python_version(),**{k:importlib.metadata.version(v) for k,v in names.items()})


def worker(config,job_id):
    body=verify(config);binding=ready()
    if binding is None:raise ValueError('All actual complete data/mask prerequisites required before numerical imports')
    job=next(j for j in config['jobs'] if j['id']==job_id)
    if numerical_environment()!=body['verified_native_environment']:raise ValueError('Exact verified native training environment required')
    # Rehash full immutable values/masks and generated raw-target profiles before
    # either model: a prior data audit is not a license to consume modified input.
    checked=set()
    for row in binding['source_records']:
        for path,digest in row['input_files_sha256'].items():
            if path not in checked:
                if sha(path)!=digest:raise ValueError('Full frozen training input changed: '+path)
                checked.add(path)
    for profile in binding['original_anomaly_target_bindings'].values():
        if profile['kind']=='verified_boolean_array' and sha(profile['path'])!=profile['sha256']:
            raise ValueError('Actual original raw-observation target validity changed')
    import numpy as np
    import torch
    import yaml
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('Original CUDA bfloat16 training required;no CPU/precision fallback')
    torch.cuda.set_device(0);torch.set_num_threads(config['settings']['cpu_threads'])
    source=ROOT/body['original_source_root']
    seednode=next(n for n in ast.parse((source/'moment/utils/utils.py').read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name=='control_randomness')
    namespace=dict(random=random,os=os,np=np,torch=torch)
    exec(compile(ast.Module(body=[seednode],type_ignores=[]),'<pinned-source-randomness>','exec'),namespace)
    namespace['control_randomness'](job['seed'])
    core_config=read(ROOT/body['core_model_configuration']);parameters=dict(core_config['parameters'],mixer=job['mixer'])
    core=train.core_class(ROOT/'src/iia_benchmark/models')(**parameters)
    Masking,_=train.load_source_symbols(source,body['original_numerical_source_sha256'])
    model=train.masked_model(core,Masking,gradient_checkpointing=True,mask_ratio=.3)
    epoch_config=read(ROOT/masks.inputs.epoch.CONFIG)
    plan=masks.inputs.epoch.sampler.DiskEpochPlan(Path(epoch_config['artifact_root'])/'paired_epoch')
    if (len(plan)!=39479 or plan.plan['total_items']!=80850189 or plan.plan['sources']!=binding['sources']
            or plan.plan['batch_size']!=2048):raise ValueError('Actual complete frozen paired oneepoch plan required')
    reader=masks.data.PileCacheReader(binding['sources'],epoch_seed=13,
        original_anomaly_target_bindings=binding['original_anomaly_target_bindings'])
    loader=train.CompleteEpochLoader(plan,reader)
    output=Path(config['artifact_root'])/job['id'];store=artifacts.TrainingArtifacts(output,job,sha(ROOT/CONFIG))
    args=yaml.safe_load((source/'configs/pretraining/pretrain.yaml').read_text(encoding='utf-8'))
    if (args['train_batch_size']!=2048 or args['mask_ratio']!=.3 or args['optimizer_name']!='AdamW'
            or args['torch_dtype']!='bfloat16' or not args['use_amp'] or not args['set_input_mask']
            or args['log_interval']<=39479 or args['checkpoint_interval']!=5000):raise ValueError('Exact pinned source training/validation callback recipe required')
    args.update(max_epoch=1,checkpoint_path=str(store.checkpoint_root))
    task=train.SourceTrainingTask(model,loader,SimpleNamespace(**args),source,body['original_numerical_source_sha256'],store,
        device='cuda:0',save_callback=store.save_periodic,external_callback=store.defer_external)
    try:
        task.train();proof=store.finalize(task,loader)
    finally:reader.close()
    write(output/'pretraining_result.json',dict(passed=True,job=job,configuration_sha256=sha(ROOT/CONFIG),
        complete_training_state_proof=str(output/'complete_training_state.json'),complete_training_state_sha256=sha(output/'complete_training_state.json'),
        full_optimizer_updates=proof['actual_audited_optimizer_updates'],full_epoch_items=proof['scope']['items'],
        parameter_count=sum(p.numel() for p in model.parameters()),environment=numerical_environment(),
        historical_initialization_or_paper_equivalence_certified=False,independent_model_replay_complete=False,
        downstream_evaluations_complete=False,all_original_experiments_complete=False))


def guarded_gpu_functions(api):
    """Load the unchanged original neural host/GPU guard without model imports."""
    import psutil
    path=ROOT/'scripts/flow_matching/run_grasp_protocol_queue.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['gpu_snapshot','guarded_gpu_wait']]
    guard=next(n for n in nodes if n.name=='guarded_gpu_wait')
    conditions=[n for n in ast.walk(guard) if isinstance(n,ast.If)
        and "memory['commit_available_bytes'] < settings['emergency_commit_headroom_bytes']" in ast.unparse(n.test)]
    if len(conditions)!=1:raise ValueError('Original full host/GPU emergency condition required')
    module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    namespace=dict(subprocess=subprocess,psutil=psutil,time=time)
    exec(compile(module,'<unchanged-original-neural-host-and-gpu-guard>','exec'),namespace)
    return namespace['gpu_snapshot'],namespace['guarded_gpu_wait']


def controller(config):
    import msvcrt,psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=(base/'controller.lock').open('a+b')
    if lane.tell()==0:lane.write(b'0');lane.flush()
    lane.seek(0)
    try:msvcrt.locking(lane.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lane.close();raise RuntimeError('Existing full pretraining queue owner preserved')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,completed_jobs=[])
    publish=status_writer(base/'status.json',write)
    def update(value=None,**extra):state.update(value or {});state.update(extra);publish(base/'status.json',state)
    try:
        while ready() is None:
            update(state='waiting_actual_full2086_inputs_and_full_mask_compatibility')
            for module in (masks,masks.inputs):
                path=ROOT/read(ROOT/module.CONFIG)['state_root']/'status.json'
                if path.exists() and read(path).get('state')=='failed_preserved_without_retry':raise ValueError('Actual full data/mask prerequisite failed')
            time.sleep(10)
        api=install_facades();gpu_snapshot,gpu_wait=guarded_gpu_functions(api)
        for job in config['jobs']:
            attempt=base/job['id'];output=Path(config['artifact_root'])/job['id']
            if attempt.exists() or output.exists():raise FileExistsError('Preserve every previous full pretraining attempt;no implicit retry')
            while True:
                gpu=gpu_snapshot(config['settings'])
                if gpu['gpu_free_bytes']<config['settings']['minimum_gpu_free_bytes']:
                    update(gpu,state='waiting_original_GPU_headroom',active_job=job['id']);time.sleep(10);continue
                with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],lambda v:update(v,active_job=job['id'])):
                    if gpu_snapshot(config['settings'])['gpu_free_bytes']<config['settings']['minimum_gpu_free_bytes']:continue
                    verify(config)
                    if ready() is None:raise ValueError('Full scientific prerequisites disappeared before dispatch')
                    attempt.mkdir(parents=True,exist_ok=False)
                    command=[str(ROOT/config['python']),'-B','-X','utf8','-u','-m',MODULE,'--worker','--job-id',job['id']]
                    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'),
                        PYTHONHASHSEED=str(job['seed']),CUDA_VISIBLE_DEVICES=str(config['settings']['gpu_index']),
                        OMP_NUM_THREADS=str(config['settings']['cpu_threads']),MKL_NUM_THREADS=str(config['settings']['cpu_threads']))
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=environment,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                        write(attempt/'launch.json',handle);update(state='running_full_original_epoch',active_pid=child.pid,active_job=job['id'])
                        code,usage=gpu_wait(child,config['settings'],api,lambda v:update(v,active_job=job['id']))
                    proof=output/'pretraining_result.json'
                    passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                        proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed:
                        update(state='failed_preserved_without_retry',active_pid=None);raise RuntimeError('Full scientific attempt failed;preserved without retry')
                    state['completed_jobs'].append(job['id']);update(state='between_full_models',active_pid=None)
                    break
            time.sleep(config['settings']['controller_yield_seconds'])
        update(state='both_full_training_states_saved_downstream_and_independent_replay_pending',active_pid=None)
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--job-id');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen complete two-model pretraining queue')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,models=2,sources_per_model=2086,items_per_epoch=80850189,batches_per_epoch=39479)))
    elif args.worker:worker(config,args.job_id)
    else:controller(config)


if __name__=='__main__':main()
