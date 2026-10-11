"""Execute all ten original full LS4 evaluation replays after actual native completion."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from scripts.flow_matching.ls4_full_saved_evaluation_replay_v1 import ROOT,read,sha,budget,constructor_ast,exclusive,replay_job,descriptor_equal,tensor_descriptors
from scripts.flow_matching.ls4_physio_full_evaluation_observation_v1 import verify_extended_result
from scripts.flow_matching.run_ls4_physio_full_observation_dispatch_v1 import verify as verify_observation,environment
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import write
from scripts.flow_matching.full_gpu_memory_slot_v2 import full_gpu_memory_slot
from scripts.flow_matching.run_spectral_original_queue import guarded_api


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen whole LS4 full-replay source changed')
    observation=read(ROOT/config['observation_configuration']);queue=verify_observation(observation)
    if (config['original_queue']!=observation['original_queue'] or config['original_jobs']!=queue['jobs']
            or len(config['original_jobs'])!=10 or len({j['id'] for j in config['original_jobs']})!=10
            or config['settings']!=queue['settings']
            or config['resource_lock']!=queue['resource_lock'] or config['worker_python']!=queue['python']
            or config['training_state_root']!=observation['state_root']
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or config['required_full_native_evaluations']!=500 or config['required_original_model_calls']!=37500
            or config['required_original_training_updates']!=750000 or config['additional_training_seed_slots']!=0):
        raise ValueError('All ten full original jobs,500evals,37500calls and unchanged original guards required')
    calls=updates=0
    for job in queue['jobs']:
        model=read(ROOT/job['model_config']);b=budget(model)
        if model['original_optim'].get('use_ema',False):raise ValueError('Exact original non-EMA PhysioNet profiles required')
        calls+=50*2*b['test_batches'];updates+=b['optimizer_updates']
    if (calls,updates)!=(37500,750000):raise ValueError('Whole original method and training budgets required')
    return queue


def terminal_original(queue,job,config):
    receipt_path=ROOT/config['training_state_root']/'receipts'/(job['id']+'.json')
    if not receipt_path.exists():return None
    receipt=read(receipt_path)
    if receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or not receipt['full_result_validated']:
        raise ValueError('Original complete training must actually terminate without guard abort')
    result=verify_extended_result(queue,job,config['observation_configuration'])
    return dict(job=job,original_result_sha256=sha(ROOT/job['output_directory']/'result.json'),
        training_receipt_sha256=sha(receipt_path),complete_original_training_receipt=receipt,
        original_full_evaluations=len(result['evaluations']))


def verify_complete_replay(report,job,config,binding):
    expected=50*2*budget(read(ROOT/job['model_config']))['test_batches']
    if (report['job']!=job or report['original_result_sha256']!=binding['original_result_sha256']
            or report['training_receipt_sha256']!=binding['training_receipt_sha256']
            or report['full_native_evaluations_executed']!=50 or len(report['records'])!=50
            or report['all_original_full_model_calls_executed']!=expected
            or not report['all_full_original_model_inputs_and_outputs_compared']
            or report['additional_optimizer_updates']!=0 or report['additional_training_seed_slots']!=0
            or report['author_equivalence_certified'] or report['strict_TAB_result']):
        raise ValueError('Actual complete original model/evaluation replays required; no shortcut or extra seeds')
    if report['configuration_sha256']!=sha(ROOT/report['configuration']):raise ValueError('Replay configuration changed')
    original=read(ROOT/job['output_directory']/'result.json')
    if sha(ROOT/job['output_directory']/'result.json')!=binding['original_result_sha256']:
        raise ValueError('Original full saved evaluation metadata changed')
    epoch_values=[r['native_epoch'] for r in report['records']]
    if epoch_values!=list(range(10,500,10))+[-1]:raise ValueError('Every original complete periodic and final evaluation required')
    for index,row in enumerate(report['records']):
        if (row['evaluation_index']!=index or row['original_full_test_patients']!=1600
                or row['full_original_method_calls']!=expected//50
                or len(row['calls'])!=expected//50 or row['additional_optimizer_updates']!=0
                or not row['all_original_model_optimizer_scheduler_rng_and_backend_flags_restored']
                or any(not c['all_complete_original_tensor_inputs_equal'] or c['original_model_call_count']!=1
                    or c['additional_optimizer_updates']!=0 for c in row['calls'])):
            raise ValueError('All complete original contexts and every full method call required')
        context=original['evaluations'][index]['full_native_evaluation_observation']
        if row['original_snapshot_sha256']!=context['sha256'] or row['original_full_test_batches']!=context['full_test_batches']:
            raise ValueError('Every replay must bind its entire original context')
        for actual,saved in zip(row['calls'],context['full_method_calls']):
            if (actual['method']!=saved['method'] or set(actual['full_tensor_comparisons'])!=set(saved['array_shapes'])
                    or actual['original_complete_output_structure']!=saved['output_structure']
                    or descriptor_equal(actual['replayed_complete_output_structure'],saved['output_structure'])
                        !=actual['output_structure_and_scalars_bitwise_equal']):
                raise ValueError('Every original full method/tensor and scalar comparison required')
            descriptors=tensor_descriptors(saved['input_structure']);descriptors.update(tensor_descriptors(saved['output_structure']))
            if set(descriptors)!=set(saved['array_shapes']):raise ValueError('All original typed tensor descriptors required')
            for key,tensor in actual['full_tensor_comparisons'].items():
                if (tensor['shape']!=saved['array_shapes'][key] or tensor['dtype']!=descriptors[key]['dtype']
                        or key.startswith('input') and not tensor['bitwise_equal']):
                    raise ValueError('Every original full tensor shape and input identity required')
            call_match=actual['output_structure_and_scalars_bitwise_equal'] and all(
                t['bitwise_equal'] for k,t in actual['full_tensor_comparisons'].items() if k.startswith('output'))
            if call_match!=actual['all_complete_outputs_bitwise_equal']:
                raise ValueError('Full numerical output differences cannot be accepted as equality')
        if (all(c['all_complete_outputs_bitwise_equal'] for c in row['calls'])!=row['all_original_method_outputs_bitwise_equal']
                or (row['replayed_metrics']==row['original_metrics'])!=row['all_original_metrics_exactly_equal']
                or any(original['evaluations'][index][k]!=v for k,v in row['original_metrics'].items())):
            raise ValueError('Every complete method comparison and original native metric must bind its source')
    strict=all(r['all_original_method_outputs_bitwise_equal'] and r['all_original_metrics_exactly_equal'] for r in report['records'])
    if strict!=report['strict_full_saved_outputs_and_metrics_match']:
        raise ValueError('Numerical discrepancies cannot be disguised as strict full reproduction')
    return strict


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job-id');parser.add_argument('--binding')
    args=parser.parse_args();config=read(ROOT/args.config);queue=verify(config)
    if args.probe_only:
        environment(queue)
        compile(constructor_ast((ROOT/queue['source_root']/'train_interp_extrap.py').read_text(encoding='utf-8')),
            '<complete-original-LS4-constructor-prefix>','exec')
        print(json.dumps(dict(passed=True,original_full_jobs=10,required_complete_evaluations=500,
            required_complete_original_model_calls=37500,exact_author_environment_lock_verified=True,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,numerical_model_inference_executed=False)));return
    if args.worker:
        if not args.job_id or not args.binding:raise ValueError('Exact original job and terminal receipt binding required')
        binding=read(ROOT/args.binding);job=next(j for j in queue['jobs'] if j['id']==args.job_id)
        if binding['job']!=job:raise ValueError('Original full-job binding differs')
        replay_job(config,args.config,job,binding);return
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=False);api=guarded_api()
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full original evaluation replay queue already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration_sha256=sha(ROOT/args.config),
        state='waiting_original_complete_training_receipts',active_pid=None,executed_original_ids=[],
        strict_verified_original_ids=[],failed_execution_preserved={},numerical_mismatch_original_ids=[],
        required_full_original_models=10,required_full_native_evaluations=500,required_full_original_method_calls=37500,
        original_scope=config['original_scope'],original_observation_retry_errors={},all_original_experiments_complete=False)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    update({})
    try:
        while len(state['executed_original_ids'])+len(state['failed_execution_preserved'])<10:
            ready=[]
            for job in queue['jobs']:
                if job['id'] in state['executed_original_ids'] or job['id'] in state['failed_execution_preserved']:continue
                try:
                    binding=terminal_original(queue,job,config)
                    if binding is not None:ready.append((job,binding))
                except (OSError,json.JSONDecodeError) as error:
                    state['original_observation_retry_errors'][job['id']]=dict(error_type=type(error).__name__,error=str(error))
                    update({})
                except Exception as error:
                    state['failed_execution_preserved'][job['id']]=dict(error_type=type(error).__name__,error=str(error));update({})
            if not ready:
                source=config['original_training_owner'];live=False
                try:
                    p=psutil.Process(source['pid']);live=p.create_time()==source['create_time'] and p.cmdline()==source['command']
                except psutil.NoSuchProcess:pass
                update(dict(state='waiting_actual_original_complete_training_receipts',actual_original_training_owner_live=live,
                    pending_original_models=10-len(state['executed_original_ids'])-len(state['failed_execution_preserved'])))
                time.sleep(config['poll_seconds']);continue
            for job,binding in ready:
                directory=base/'jobs'/job['id'];directory.mkdir(parents=True,exist_ok=False)
                binding_path=directory/'original_complete_training_binding.json';exclusive(binding_path,binding)
                with full_gpu_memory_slot(api,config,update):
                    command=[str(ROOT/config['worker_python']),'-X','utf8','-u','-m',
                        'scripts.flow_matching.run_ls4_all10_full_evaluation_replay_v1','--config',args.config,'--worker',
                        '--job-id',job['id'],'--binding',str(binding_path)]
                    with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                                OMP_NUM_THREADS=str(queue['cpu_threads']),MKL_NUM_THREADS=str(queue['cpu_threads']),
                                CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled',MPLBACKEND='Agg'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    p=psutil.Process(child.pid);update(dict(state='replaying_complete_original_fifty_evaluations',active_pid=p.pid,
                        active_create_time=p.create_time(),actual_command=p.cmdline(),active_job=job['id']))
                    code,usage=api['guarded_gpu_wait'](child,config['settings'],api,update)
                strict=False
                try:
                    if code!=0 or usage['resource_guard_aborted']:raise ValueError('Full native replay execution failed or resource guard aborted')
                    report=read(ROOT/config['proof_root']/job['id']/'complete_original_full_evaluation_replay.json')
                    strict=verify_complete_replay(report,job,config,binding)
                    state['executed_original_ids'].append(job['id'])
                    (state['strict_verified_original_ids'] if strict else state['numerical_mismatch_original_ids']).append(job['id'])
                except Exception as error:
                    state['failed_execution_preserved'][job['id']]=dict(error_type=type(error).__name__,error=str(error))
                exclusive(directory/'receipt.json',dict(exit_code=code,complete_original_job_replay_executed=job['id'] in state['executed_original_ids'],
                    strict_full_saved_outputs_and_metrics_match=strict,additional_training_seed_slots=0,**usage))
                update(dict(active_pid=None,active_job=None));time.sleep(config['settings']['controller_yield_seconds'])
        update(dict(state='all_ten_original_replay_attempts_terminal',all_ten_full_replays_executed=len(state['executed_original_ids'])==10,
            all_ten_strict_output_and_metric_reproductions_passed=len(state['strict_verified_original_ids'])==10))
    finally:owner.close()


if __name__=='__main__':main()
