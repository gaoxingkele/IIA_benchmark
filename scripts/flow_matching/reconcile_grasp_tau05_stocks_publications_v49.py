"""Await actual complete GRASP postreview, then preserve the full Stocks cohort in the combined table."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import os
import subprocess
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT,read,sha,capped_wait
from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_grasp_tau05_postreplay_pipeline_v1 import independent_publication_review
from scripts.flow_matching.run_resource_gated_light_controller_v1 import write


def registration(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen combined-publication registration changed')
    producer=read(ROOT/config['producer_configuration'])
    if (any(config[k]!=producer[k] for k in ('settings','resource_lock','python'))
            or producer['producer_configuration']!=config['native_replay_configuration']):
        raise ValueError('Retain the original native complete postreview and all resource/environment boundaries')
    stocks=read(ROOT/config['base_publication']/'independent_publication_review.json')
    if (not stocks['passed'] or stocks['original_training_seeds']!=[42,43,44,45,46]
            or stocks['required_training_seeds']!=[42,43,44,45,46]
            or stocks['degrees_of_freedom']!=4 or not stocks['other8622rows_and_capture_times_unchanged']):
        raise ValueError('Preserve the complete independently verified five-generator Stocks publication')
    return producer


def completed_dependency(receipt,proof,review):
    if (receipt['exit_code']!=0 or receipt['resource_guard_aborted']
            or not receipt['complete_postreplay_publication_passed'] or not proof['passed']
            or not proof['both_original_native_full_model_replays_and_independent_reviews_verified']
            or proof['required_original_training_seeds']!=10 or proof['completed_original_training_seeds']!=2
            or proof['additional_training_seed_slots']!=0 or not review['passed']
            or review['original_seed_count']!=2 or review['required_original_seeds']!=10
            or review['changed_rows']!=6 or not review['other8620rows_and_capture_times_unchanged']):
        raise ValueError('Actual full native postreview and independent original two-seed publication must finish')


def worker(config,path):
    producer=registration(config)
    completed_dependency(read(ROOT/producer['state_root']/'receipt.json'),
        read(ROOT/producer['output_directory']/'postreplay_pipeline_completed.json'),
        read(ROOT/producer['publication_directory']/'independent_publication_review.json'))
    previous_path=producer['publication_configuration']
    publication=deepcopy(read(ROOT/previous_path))
    publication.update(base_publication=config['base_publication'],output_directory=config['publication_directory'],
        cohort_summary_path=config['cohort_summary_path'])
    sources=[path,previous_path,'scripts/flow_matching/publish_grasp_tau05_two_seed_listing_v49.py',
        producer['state_root']+'/receipt.json',producer['output_directory']+'/postreplay_pipeline_completed.json',
        producer['publication_directory']+'/independent_publication_review.json',
        config['base_publication']+'/all_algorithm_dataset_metrics.csv',
        config['base_publication']+'/publication_validation.json',
        config['base_publication']+'/independent_publication_review.json']
    publication['source_receipts']+=[dict(path=p,sha256=sha(ROOT/p)) for p in dict.fromkeys(sources)]
    exclusive_json(ROOT/config['publication_configuration'],publication)
    from scripts.flow_matching.publish_grasp_tau05_two_seed_listing_v49 import publish
    publish(config['publication_configuration'])
    independent_publication_review(config)
    registration(config)
    exclusive_json(ROOT/config['output_directory']/'combined_publication_completed.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=path,
        original_GRASP_tau05_training_seeds=2,required_GRASP_training_seeds=10,
        original_DiffusionTS_Stocks_training_seeds=5,required_Stocks_training_seeds=5,
        six_existing_GRASP_rows_updated_other8620rows_preserved=True,
        original_complete_five_generator_Stocks_rows_preserved=True,
        additional_training_seed_slots=0,additional_model_inference_executed=False,
        original_scope=config['original_scope'],author_equivalence_certified=False,
        strict_TAB_result=False,all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);producer=registration(config)
    if args.worker:
        worker(config,args.config);return
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:
        raise RuntimeError('The exact combined original result controller is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        receipt_path=ROOT/producer['state_root']/'receipt.json'
        while not receipt_path.exists():
            actual=process_identity(config['producer_pid'],config['producer_create_time'],
                'run_grasp_tau05_postreplay_pipeline_v1')
            if not actual['live'] or not actual['create_time_verified']:
                update(dict(state='missing_or_unverified_original_postreview',actual_dependency=actual));return
            update(dict(state='waiting_for_actual_complete_original_postreview',actual_dependency=actual))
            time.sleep(10)
        completed_dependency(read(receipt_path),read(ROOT/producer['output_directory']/'postreplay_pipeline_completed.json'),
            read(ROOT/producer['publication_directory']/'independent_publication_review.json'))
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m',
                'scripts.flow_matching.reconcile_grasp_tau05_stocks_publications_v49','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,
                    PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                    CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                handle=psutil.Process(child.pid)
                update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        proof=ROOT/config['output_directory']/'combined_publication_completed.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed'])
        write(base/'receipt.json',dict(exit_code=code,combined_publication_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
        if not passed:
            raise RuntimeError('Preserve failed complete native publication reconciliation')
    finally:
        lane.close()


if __name__=='__main__':
    main()
