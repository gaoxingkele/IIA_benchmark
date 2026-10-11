"""Preserve a failed full Energy attempt and dispatch its original cold recipe."""
import argparse
import ast
import copy
from datetime import datetime,timezone
import inspect
import json
import os
import sys
import time
import traceback

import psutil

from scripts.flow_matching import run_spectral_resource_order_v4 as ordering
from scripts.flow_matching.prepare_spectral_original_data import ROOT,read,sha,write

IDENTITY='spectral_original__spectral_flow__energy__released_author_data__seed42'


def exact_retry(original,queue,config):
    if len(original['jobs'])!=70 or len(queue['jobs'])!=70 or len({j['id'] for j in queue['jobs']})!=70:
        raise ValueError('Every original70job/14cohort obligation required')
    expected=copy.deepcopy(original);expected['state_root']=config['state_root']
    job=next(j for j in expected['jobs'] if j['id']==IDENTITY)
    old=copy.deepcopy(job)
    job['output_directory']=config['selected_output_directory']
    job['artifact_directory']=config['selected_artifact_directory']
    if (expected!=queue or job['output_directory']==old['output_directory']
            or job['artifact_directory']==old['artifact_directory']
            or old['training_updates']!=25000 or old['data_shape']!=[19712,24,28]
            or old['generation_batch_size']!=2001 or old['generation_field_chunk_size'] is not None
            or queue['spectral_sampling_minimum_commit_headroom_bytes']!=48*1024**3):
        raise ValueError('Only fresh output/artifact/controller paths may change; never data, budgets, sampler or guards')
    return old,job


def selected_ast(source):
    tree=ordering.ordered_main_ast(source)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
        and n.target.id=='job' and ast.unparse(n.iter)=='scheduler_jobs(queue)']
    if len(loops)!=1:raise ValueError('One original full sorted dispatch loop required')
    loops[0].iter=ast.parse("[j for j in scheduler_jobs(queue) if j['id']==selected_canonical_id]",mode='eval').body
    return ast.fix_missing_locations(tree)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full retry source changed: '+pin['path'])
    prior=read(ROOT/config['original_resource_order_configuration'])
    original=ordering.verify_registration(prior)
    queue=read(ROOT/config['queue']);old,job=exact_retry(original,queue,config)
    inspection=read(ROOT/config['terminal_checkpoint_inspection'])
    if (inspection['original_job']!=old or inspection['full_checkpoint_training_step']!=25000
            or not inspection['complete_optimizer_update_budget_verified']
            or inspection['actual_terminal_native_handle']['live']
            or inspection['actual_terminal_resource_receipt']['resource_guard_aborted'] is not True
            or inspection['RNG_state_saved'] or inspection['scheduler_state_saved']
            or inspection['exact_post_training_sampling_RNG_resume_certified']):
        raise ValueError('Terminal original full training and unproven RNG resume must be preserved')
    if (ROOT/old['output_directory']/'result.json').exists():raise ValueError('Inspect unexpected original completion before retry')
    preflight=read(ROOT/config['preflight']);previous=read(ROOT/prior['preflight'])
    equivalent=copy.deepcopy(preflight);equivalent['queue_sha256']=previous['queue_sha256']
    if equivalent!=previous or preflight['queue_sha256']!=sha(ROOT/config['queue']):
        raise ValueError('Only full preflight registry-path hash may differ, never its measured observations')
    return queue,old,job


def prerequisite(proof,receipt):
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('full_native_Trainer_sampling_graph_audit_passed') is not True
            or proof.get('passed') is not True
            or proof.get('full_native_Trainer_sample_executed_twice') is not True
            or proof.get('original_generated_shape')!=[4002,24,6]
            or proof.get('reviewed_generated_shape')!=[4002,24,6]
            or proof.get('complete_field_calls_per_pass')!=1000
            or proof.get('every_full_field_input_derivative_time_and_auxiliary_bitwise_equal') is not True
            or proof.get('all4002_generated_windows_bitwise_equal') is not True
            or proof.get('final_Torch_CPU_RNG_state_equal') is not True):
        raise ValueError('Actual full shared Stocks CPU graph verification must pass before the new attempt')
    # This shared-component gate is not an Energy capacity or CUDA equivalence certificate.


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args();config=read(ROOT/args.config)
    queue,old,job=verify(config)
    if args.probe_only:
        print(json.dumps(dict(required_original_jobs=70,selected_canonical_id=IDENTITY,
            cold_original_training_updates=25000,original_generation_batch=2001,
            original_full_generation_shape=[20010,24,28],sampling_RNG_resume_claimed=False,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,
            all_original_experiments_complete=False)))
        return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=resource_api()['exclusive_lock'](base/'preparation.lock')
    if lane is None:raise RuntimeError('This exact full cold retry already has a live owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_original_jobs=70,
        selected_canonical_id=IDENTITY,active_pid=None,configuration_sha256=sha(ROOT/args.config),
        additional_training_seed_slots=0,all_original_experiments_complete=False)
    try:
        while not (ROOT/config['shared_sampling_receipt']).exists():
            handle=psutil.Process(config['expected_shared_sampling_controller']['pid'])
            expected=config['expected_shared_sampling_controller']
            if handle.create_time()!=expected['create_time'] or handle.cmdline()!=expected['command']:
                raise ValueError('Shared full sampling diagnostic identity changed; inspect before any training')
            state.update(state='waiting_actual_full_shared_sampling_graph_verification',
                captured_utc=datetime.now(timezone.utc).isoformat(),actual_shared_controller_pid=handle.pid,
                actual_shared_controller_create_time=handle.create_time(),actual_shared_controller_command=handle.cmdline())
            write(base/'status.json',state);time.sleep(config['poll_seconds'])
        prerequisite(read(ROOT/config['shared_sampling_proof']),read(ROOT/config['shared_sampling_receipt']))
        for process in psutil.process_iter(['cmdline']):
            command=process.info['cmdline'] or []
            if '--job-id' in command and command[command.index('--job-id')+1]==IDENTITY:
                raise RuntimeError('An actual original canonical native job already runs; do not duplicate')
        verify(config)
        write(base/'shared_component_gate_passed.json',dict(passed=True,
            shared_Stocks_CPU_sampling_graph_verified=True,Energy_capacity_certified=False,
            CUDA_integrated_sampling_equivalence_certified=False,original_source_worker_and_sampler_unchanged=True,
            cold_restart_required_training_updates=25000,actual_training_started=False,
            prior_checkpoint_and_failed_attempt_preserved=True,original_post_training_RNG_resumed=False,
            additional_training_seed_slots=0,all_original_experiments_complete=False))
        previous=ordering.previous
        namespace=dict(previous.original.__dict__,admission_settings=previous.admission_settings,
            scheduler_jobs=ordering.scheduler_jobs,registered_settings=copy.deepcopy(queue['settings']),selected_canonical_id=IDENTITY)
        exec(compile(selected_ast(inspect.getsource(previous.original.main)),
            '<unchanged native full queue selecting only preserved Energy42 retry>', 'exec'),namespace)
        sys.argv=['original-full-generation-controller','--queue',config['queue'],'--preflight',config['preflight']]
        namespace['main']()
    except Exception as error:
        state.update(state='failed_preserved_full_retry_preparation_or_dispatch',
            captured_utc=datetime.now(timezone.utc).isoformat(),error=str(error),
            all_original_experiments_complete=False)
        write(base/'preparation_failure.json',dict(**state,traceback=traceback.format_exc()))
        # Preserve native queue status and receipts if dispatch already began.
        if not (base/'controller.lock').exists():write(base/'status.json',state)
        raise
    finally:lane.close()


if __name__=='__main__':main()
