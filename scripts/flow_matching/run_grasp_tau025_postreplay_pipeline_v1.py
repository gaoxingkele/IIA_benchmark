"""Publish the original tau0.25 seed pair only after full native replay succeeds."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime,timezone
import json
import math
import os
import statistics
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT,read,sha
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.verify_grasp_completed_full_replay_v1 import check_terminal,verify


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen original tau0.25 post-replay sources changed')
    producer=read(ROOT/config['producer_configuration'])
    if (config['settings']!=producer['settings'] or config['resource_lock']!=producer['resource_lock']
            or config['python']!=producer['python'] or producer['full_job_count']!=1
            or producer['full_results'][0]['canonical_id']!='grasp_protocol_v3__tau_0p25__swan__swan_sf__seed1104'):
        raise ValueError('Retain the original full native tau0.25 replay and resource boundaries')
    old=read(ROOT/config['first_seed_review_proof'])
    if (not old['passed'] or not old['all_metrics_independently_recomputed_from_full_replay']
            or old['recipes'][0]['seed']!=1103 or old['actual_completed_recipe_count']!=1):
        raise ValueError('Original seed1103 must already have independent complete replay verification')
    return producer


def register_completed_review(config,producer):
    output=ROOT/producer['output_directory']
    report=read(output/'checkpoint_replay_audit.json');receipt=read(ROOT/producer['state_root']/'receipt.json')
    canonical=producer['full_results'][0]['canonical_id']
    proof=(output/(canonical+'.json')).relative_to(ROOT).as_posix()
    review=deepcopy(producer)
    review.update(completed_replay_configuration=config['producer_configuration'],
        verification_output_directory=config['second_seed_review_directory'],
        reusable_complete_recipes=[dict(canonical_id=canonical,recipe=r['recipe'],proof_path=proof,proof_sha256=sha(ROOT/proof))
            for r in report['records'][0]['recipes']],pending_full_recipe_count=0)
    sources=[config['producer_configuration'],'scripts/flow_matching/verify_grasp_completed_full_replay_v1.py',
        'scripts/flow_matching/audit_grasp_checkpoint_replay_resume_v2.py','tests/test_fm_grasp_completed_full_replay_v1.py',
        producer['state_root']+'/receipt.json',producer['state_root']+'/status.json']
    sources += [p.relative_to(ROOT).as_posix() for p in output.glob('*.json')]
    review['source_receipts'] += [dict(path=p,sha256=sha(ROOT/p)) for p in dict.fromkeys(sources)]
    check_terminal(report,receipt,review)
    exclusive_json(ROOT/config['second_seed_review_configuration'],review)
    verify(review,config['second_seed_review_configuration'])


def register_publication(config,producer):
    paths={1103:config['first_seed_review_configuration'],1104:config['second_seed_review_configuration']}
    proofs={1103:config['first_seed_review_proof'],1104:config['second_seed_review_directory']+'/independent_full_replay_verification.json'}
    bindings={}
    sources=list(paths.values())+list(proofs.values())+[config['base_publication']+'/all_algorithm_dataset_metrics.csv',
        config['base_publication']+'/publication_validation.json',producer['queue'],
        'scripts/flow_matching/audit_grasp_verified_seed_pair_v1.py','scripts/flow_matching/publish_grasp_tau025_two_seed_listing_v45.py',
        'scripts/flow_matching/audit_grasp_complete_replay_union_v1.py','scripts/flow_matching/audit_cfm_full_arrays_v1.py',
        'scripts/flow_matching/publish_pi_original_source_listing_v42.py','scripts/flow_matching/publish_brits_two_seed_listing_v27.py',
        'tests/test_fm_grasp_verified_seed_pair_v1.py','tests/test_fm_grasp_tau025_two_seed_listing_v45.py']
    for seed,path in paths.items():
        full=read(ROOT/path)['full_results'][0]
        execution_queue=config['first_seed_execution_queue'] if seed==1103 else producer['queue']
        bindings[str(seed)]=dict(**full,completed_review_configuration=path,
            completed_full_replay_proof=proofs[seed],execution_queue=execution_queue)
        sources.extend([full['result_path'],execution_queue])
    publication=dict(profile='tau_0p25',grasp_queue=producer['queue'],completed_seed_bindings=bindings,
        base_publication=config['base_publication'],output_directory=config['publication_directory'],
        cohort_summary_path=config['cohort_summary_path'],score_tolerance=producer['score_tolerance'],
        metric_tolerance=producer['metric_tolerance'],original_scope=producer['original_scope'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in dict.fromkeys(sources)])
    exclusive_json(ROOT/config['publication_configuration'],publication)
    from scripts.flow_matching.publish_grasp_tau025_two_seed_listing_v45 import publish
    publish(config['publication_configuration'])
    return publication


def independent_publication_review(config,publication):
    before=list(csv.DictReader((ROOT/config['base_publication']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig')))
    reader=csv.DictReader((ROOT/config['publication_directory']/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig'))
    after=list(reader)
    if len(before)!=8626 or len(after)!=8626 or len(reader.fieldnames)!=31:raise ValueError('Keep every original published metric slot and field')
    changed=[i for i,(old,new) in enumerate(zip(before,after)) if old!=new]
    if len(changed)!=6 or any(before[i]['algorithm']!='GRASP/tau_0p25' or before[i]['n']!='1'
            or after[i]['n']!='2' or after[i]['required_repeats']!='10' for i in changed):
        raise ValueError('Exactly six existing tau0.25 rows may advance from one to two original seeds')
    from scripts.flow_matching.publish_grasp_tau025_two_seed_listing_v45 import check_delta
    if check_delta(before,after)!=changed:raise ValueError('The entire publication delta must be independently checked')
    summary=read(ROOT/config['cohort_summary_path']);reviews=[]
    for item in summary['metric_statistics']:
        field='paper_retrospective' if item['protocol']=='paper_test_label_oracle_no_PA' else 'validation_only_control'
        values=[read(ROOT/publication['completed_seed_bindings'][str(seed)]['result_path'])['metrics'][0]['metrics'][field][item['metric']]
            for seed in (1103,1104)]
        if item['original_seed_values']!=values or item['n']!=2:raise ValueError('Preserve unique actual original seed values')
        avg=statistics.mean(values);std=statistics.stdev(values);se=std/math.sqrt(2)
        if any(not math.isclose(item[k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in [('mean',avg),('std',std),('se',se)]):
            raise ValueError('Independent sampleSTD/SE disagree')
        if se:
            q=(item['ci95_high']-avg)/se
            if abs(.5+math.atan(q)/math.pi-.975)>1e-12 or not math.isclose(item['ci95_low'],avg-q*se,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('Independent df1 Cauchy/Student-t95percentCI disagrees')
        elif item['ci95_low']!=avg or item['ci95_high']!=avg:raise ValueError('Zero-variance CI differs')
        row=next(r for r in after if r['algorithm']=='GRASP/tau_0p25' and r['dataset']=='swan'
            and r['protocol']==item['protocol'] and r['metric']==item['metric'])
        if any(float(row[k])!=item[k] for k in ['n','mean','std','se','ci95_low','ci95_high']):raise ValueError('Published metric/statistic differs')
        reviews.append(dict(metric=item['metric'],protocol=item['protocol'],original_values_and_independent_statistics_verified=True,
            recovery_alias_does_not_add_seed=True))
    exclusive_json(ROOT/config['publication_directory']/'independent_publication_review.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,original_rows=8626,fields=31,changed_rows=6,other8620rows_and_capture_times_unchanged=True,
        measured_rows=sum(bool(r['mean']) for r in after),pending_rows=sum(not bool(r['mean']) for r in after),
        statistics=reviews,original_seed_count=2,required_original_seeds=10,additional_training_seed_slots=0,
        all_original_experiments_complete=False))


def worker(config,path):
    producer=verify_registration(config)
    register_completed_review(config,producer)
    publication=register_publication(config,producer)
    independent_publication_review(config,publication)
    verify_registration(config)
    exclusive_json(ROOT/config['output_directory']/'postreplay_pipeline_completed.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=path,configuration_sha256=sha(ROOT/path),
        both_original_native_full_model_replays_and_independent_reviews_verified=True,
        six_existing_numeric_rows_updated_to_two_original_seeds=True,recovery_alias_counted_as_original_seed1103_once=True,
        required_original_training_seeds=10,completed_original_training_seeds=2,
        additional_model_inference_executed_by_this_postreview=False,additional_training_seed_slots=0,
        original_scope=config['original_scope'],all_original_experiments_complete=False,
        author_equivalence_certified=False,strict_TAB_result=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);producer=verify_registration(config)
    if args.probe_only:
        print(dict(original_seed_pair=[1103,1104],required_original_seeds=10,full_native_replay_required=True,
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules));return
    if args.worker:worker(config,args.config);return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This original full-replay publication controller is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    from scripts.flow_matching.run_resource_gated_light_controller_v1 import write
    try:
        receipt_path=ROOT/producer['state_root']/'receipt.json'
        while not receipt_path.exists():
            handle=process_identity(config['producer_controller_pid'],config['producer_controller_create_time'],'audit_grasp_checkpoint_replay_v1')
            if not handle['live'] or not handle['create_time_verified']:
                update(dict(state='missing_or_unverified_original_native_replay',actual_producer=handle));return
            update(dict(state='waiting_for_actual_full_native_replay_receipt',actual_producer=handle));time.sleep(10)
        check_terminal(read(ROOT/producer['output_directory']/'checkpoint_replay_audit.json'),read(receipt_path),producer)
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_grasp_tau025_postreplay_pipeline_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid)
                update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        completed=ROOT/config['output_directory']/'postreplay_pipeline_completed.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and completed.exists() and read(completed)['passed'])
        write(base/'receipt.json',dict(exit_code=code,complete_postreplay_publication_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,complete_postreplay_publication_passed=passed))
    finally:lane.close()


if __name__=='__main__':main()
