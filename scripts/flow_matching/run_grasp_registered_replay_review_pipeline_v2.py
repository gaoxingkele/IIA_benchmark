"""Run an unchanged full native GRASP replay, then independently review every recipe."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.verify_grasp_completed_full_replay_v1 import check_terminal, verify as independent_verify


def binding(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete original GRASP continuation source changed')
    producer=read(ROOT/config['producer_configuration'])
    from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import verify_registration
    queue=verify_registration(producer)
    if (any(config[k]!=producer[k] for k in ['settings','resource_lock','python','cpu_threads','score_tolerance','metric_tolerance'])
            or producer['full_job_count']!=1 or len(producer['full_results'])!=1
            or producer['full_results'][0]['canonical_id']!=config['canonical_id']
            or config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
            or len(queue['jobs'])!=9120 or queue['cohort_count']!=480):
        raise ValueError('Keep the full9120 entity jobs/480 cohorts and every native replay guard/tolerance')
    return producer


def terminal(config,producer):
    report_path=ROOT/producer['output_directory']/'checkpoint_replay_audit.json'
    receipt_path=ROOT/producer['state_root']/'receipt.json'
    report,receipt=read(report_path),read(receipt_path)
    check_terminal(report,receipt,producer)
    if (report['configuration_sha256']!=sha(ROOT/config['producer_configuration'])
            or report['score_tolerance']!=producer['score_tolerance']
            or report['metric_tolerance']!=producer['metric_tolerance']):
        raise ValueError('Actual successful full native replay must bind its frozen configuration')
    return report,receipt


def review_registration(config,producer,report):
    canonical=config['canonical_id'];output=ROOT/producer['output_directory']
    proof=(output/(canonical+'.json')).relative_to(ROOT).as_posix()
    review=copy.deepcopy(producer)
    review.update(completed_replay_configuration=config['producer_configuration'],
        verification_output_directory=config['review_output_directory'],
        reusable_complete_recipes=[dict(canonical_id=canonical,recipe=r['recipe'],proof_path=proof,
            proof_sha256=sha(ROOT/proof)) for r in report['records'][0]['recipes']],pending_full_recipe_count=0)
    sources=[config['producer_configuration'],producer['state_root']+'/receipt.json',
        'scripts/flow_matching/verify_grasp_completed_full_replay_v1.py',
        'scripts/flow_matching/audit_grasp_checkpoint_replay_resume_v2.py']
    sources += [p.relative_to(ROOT).as_posix() for p in output.glob('*.json')]
    review['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in dict.fromkeys(sources)]
    return review


def worker(config,path):
    producer=binding(config);report,receipt=terminal(config,producer)
    review=review_registration(config,producer,report)
    exclusive_json(ROOT/config['review_configuration'],review)
    independent_verify(review,config['review_configuration'])
    proof_path=ROOT/config['review_output_directory']/'independent_full_replay_verification.json'
    proof=read(proof_path)
    if (not proof['passed'] or not proof['all_original_forward_counts_and_recipe_budgets_verified']
            or not proof['all_metrics_independently_recomputed_from_full_replay']
            or not proof['all_saved_full_score_arrays_and_identities_reverified']
            or proof['actual_completed_recipe_count']!=producer['full_score_recipe_count']):
        raise ValueError('Every complete original replay recipe must be independently checked')
    exclusive_json(ROOT/config['output_directory']/'completed_native_and_independent_review.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),configuration=path,configuration_sha256=sha(ROOT/path),
        passed=True,canonical_id=config['canonical_id'],native_report_sha256=sha(ROOT/producer['output_directory']/'checkpoint_replay_audit.json'),
        native_terminal_receipt=receipt,independent_full_replay_proof_sha256=sha(proof_path),
        complete_original_training_and_full_model_reinference_verified=True,
        all_original_recipes_and_coordinates_independently_verified=True,
        additional_model_inference_executed_by_this_review=False,additional_training_seed_slots=0,
        original_scope=config['original_scope'],full_registered_GRASP_entity_jobs=9120,full_registered_GRASP_cohorts=480,
        required_original_cohort_repeats=10,original_training_seed=read(ROOT/producer['full_results'][0]['result_path'])['job']['seed'],full_ten_seed_cohort_completed=False,
        published_numeric_table_unchanged=True,author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);producer=binding(config)
    if args.probe_only:
        print(dict(canonical_id=config['canonical_id'],full_original_GRASP_entity_jobs=9120,cohorts=480,
            full_native_replay_required=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules));return
    if args.worker:worker(config,args.config);return
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This exact full native replay/review pipeline already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration_sha256=sha(ROOT/args.config),
        state='starting',active_pid=None,canonical_id=config['canonical_id'],all_original_experiments_complete=False)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        # The original controller owns its original mutex, admission and emergency guards.
        command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_grasp_checkpoint_replay_v1',
            '--config',config['producer_configuration']]
        with (base/'native_dispatch.stdout.log').open('xb') as out,(base/'native_dispatch.stderr.log').open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        handle=psutil.Process(child.pid);creation=handle.create_time();actual=handle.cmdline()
        update(dict(active_pid=child.pid,active_create_time=creation,actual_command=actual))
        while child.poll() is None:
            try:
                current=psutil.Process(child.pid)
                if current.create_time()!=creation or current.cmdline()!=actual:raise ValueError('Native child identity changed')
                update(dict(state='waiting_actual_full_native_replay_terminal',actual_native_controller_cpu_seconds=sum(current.cpu_times()[:2])))
            except psutil.NoSuchProcess:
                if child.poll() is None:raise
            time.sleep(2)
        code=child.wait();update(dict(active_pid=None))
        if code!=0:
            receipt=ROOT/producer['state_root']/'receipt.json'
            write(base/'receipt.json',dict(exit_code=code,complete_native_and_independent_review_passed=False,
                native_receipt=read(receipt) if receipt.exists() else {},failed_attempt_preserved=True))
            update(dict(state='failed_native_replay_preserved'));return
        terminal(config,producer)
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_grasp_registered_replay_review_pipeline_v2',
                '--config',args.config,'--worker']
            with (base/'review.stdout.log').open('xb') as out,(base/'review.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        completed=ROOT/config['output_directory']/'completed_native_and_independent_review.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and completed.exists() and read(completed)['passed'])
        write(base/'receipt.json',dict(exit_code=code,complete_native_and_independent_review_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_review_preserved',active_pid=None))
    finally:lane.close()


if __name__=='__main__':main()
