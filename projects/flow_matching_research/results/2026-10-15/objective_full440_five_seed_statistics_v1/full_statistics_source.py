"""Full five-seed local-FM cohorts, paired controls and seven-dataset macros.

Wait for native/independent proofs; never pool original failures with repairs.
No training/inference/retry or historical-paper/TAB certification occurs here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.flow_matching import run_objective_full440_checkpoint_replay_v1 as producer
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT, read, sha, write = producer.ROOT, producer.read, producer.sha, producer.write
PRODUCER = 'configs/runtime/fm_objective_full440_checkpoint_replay.v1.json'
MODULE = 'scripts.flow_matching.run_objective_full440_statistics_v1'
SEEDS = [1103, 1104, 1105, 1106, 1107]
DATASETS = ['psm', 'smap', 'msl', 'smd', 'swat', 'swan', 'cicids']
T_CRITICAL = 2.7764451051977987


def method(case):
    return Path(case['job']['model_config']).stem


def tasks(cases):
    cohorts = {}
    for identity, case in cases.items():
        key = case['family']+'__'+method(case)+'__'+case['job']['dataset']
        group = cohorts.setdefault(key, dict(kind='cohort', id=key, family=case['family'],
            method=method(case), dataset=case['job']['dataset'], cases=[]))
        group['cases'].append(identity)
    for group in cohorts.values():
        group['cases'].sort(key=lambda i: cases[i]['job']['seed'])
        if [cases[i]['job']['seed'] for i in group['cases']] != SEEDS:
            raise ValueError('Exactly five original seeds per cohort required')
    if len(cases) != 440 or len(cohorts) != 88:
        raise ValueError('Every original,repair and preprocessing-control cohort required')
    result = dict(cohorts)
    for key, group in cohorts.items():
        if group['family'].startswith('constant_fit'):
            basecases = [cases[i]['job']['preprocessing_control_original_id'] for i in group['cases']]
            if any(i not in cases for i in basecases): raise ValueError('Missing frozen preprocessing baseline')
            identity = 'preprocessing__'+key
            result[identity] = dict(kind='paired', contrast='preprocessing', id=identity,
                cases=group['cases']+basecases, difference='constant_fit_minus_registered_original')
        reference_family = 'constant_fit' if group['family'].startswith('constant_fit') else 'original'
        baseline = reference_family+'__fm_objective_independent__'+group['dataset']
        if key != baseline:
            if baseline not in cohorts: raise ValueError('Missing independent-CFM reference')
            identity = 'versus_independent__'+key
            result[identity] = dict(kind='paired', contrast='method', id=identity,
                cases=group['cases']+cohorts[baseline]['cases'], difference='variant_minus_independent')
        if group['method'] in ['fm_objective_sf2m', 'fm_objective_sf2m_stable_v2']:
            off = group['method'].replace('sf2m', 'sf2m_flow_only')
            baseline = group['family']+'__'+off+'__'+group['dataset']
            if baseline not in cohorts: raise ValueError('Missing actual score-head-off ablation')
            identity = 'score_head__'+key
            result[identity] = dict(kind='paired', contrast='score_head', id=identity,
                cases=group['cases']+cohorts[baseline]['cases'], difference='score_head_on_minus_flow_only')
    for family in ['original', 'repair']:
        methods = list(dict.fromkeys(g['method'] for g in cohorts.values() if g['family'] == family))
        for name in methods:
            dependencies = [family+'__'+name+'__'+dataset for dataset in DATASETS]
            if any(d not in cohorts for d in dependencies): raise ValueError('All seven datasets required for macro')
            identity = 'macro__'+family+'__'+name
            result[identity] = dict(kind='macro', id=identity, dependencies=dependencies,
                cases=[i for d in dependencies for i in cohorts[d]['cases']],
                weighting='equal_dataset_within_matching_seed')
    counts = {k:sum(t['kind'] == k for t in result.values()) for k in ['cohort','paired','macro']}
    if counts != dict(cohort=88, paired=107, macro=11) or len(result) != 206:
        raise ValueError('All88cohorts,107paired controls and11macros required')
    return result


def five(values):
    if len(values) != 5 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Five finite observations required; no dropped seeds')
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd/math.sqrt(5); half = T_CRITICAL*se
    return dict(seeds=SEEDS, values=values, n=5, mean=mean, sample_standard_deviation=sd,
        standard_error=se, student_t_df4_95_interval=[mean-half,mean+half], interval_is_unclipped=True,
        median=statistics.median(values), minimum=min(values), maximum=max(values),
        leave_one_seed_out_means={str(s):statistics.mean(values[:i]+values[i+1:]) for i,s in enumerate(SEEDS)},
        interval_scope='descriptive_training_seed_variability_only',
        hypothesis_test_or_generalization_guarantee=False, multiplicity_correction_applied=False)


def summarize(vectors):
    if len(vectors) != 5 or any(set(v) != set(vectors[0]) for v in vectors):
        raise ValueError('Same endpoints for every one of five seeds required')
    rows = {}
    for key in vectors[0]:
        values = [v[key] for v in vectors]
        if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in values):
            raise ValueError('Endpoint observations must be finite numeric values or explicit nulls')
        if any(v is None for v in values):
            rows[key] = dict(seeds=SEEDS, values=values, summary=None,
                reason='undefined_original_endpoint_no_seed_dropped', defined_seeds=sum(v is not None for v in values))
        else:
            rows[key] = dict(summary=five(values), defined_seeds=5)
    return rows


def metrics_vector(metrics):
    """Keep strict protocols, event measures and label-selected diagnostics separate."""
    result = {}
    for ratio in ['0.1','1','5']:
        group = metrics['strict'][ratio]
        if (group['protocol'] != 'strict_validation_only_no_pa_v1' or group['point_adjustment']
                or group['calibration']['calibration_split'] != 'validation'):
            raise ValueError('Frozen validation-only unadjusted primary protocol required')
        for key in ['precision','recall','f1','auroc','average_precision']:
            result[f'strict.{ratio}.micro.{key}'] = group['micro'][key]
        result[f'strict.{ratio}.macro_f1'] = group['macro_f1']
        for key in ['event_recall','events','detected_events','mean_detection_delay_points','false_alarms_per_1000_normal_points']:
            result[f'strict.{ratio}.event.{key}'] = group['event_metrics'][key]
        for entity, row in group['per_entity'].items():
            result[f'strict.{ratio}.entity.{entity}.f1'] = row['f1']
    tab = metrics['tab_core_diagnostic']
    if not tab['test_scores_used_in_calibration'] or not tab['test_labels_used_to_select_best_grid']:
        raise ValueError('Original test-pooled and test-label-selected diagnostic must be disclosed')
    if [r['ratio_percent'] for r in tab['ratio_grid']] != [.1,.5,1,2,3,5,10,15,20,25]:
        raise ValueError('All ten original diagnostic ratios required')
    for row in tab['ratio_grid']:
        for key in ['f_score','adjust_f_score']:
            result[f'test_pooled_diagnostic.ratio{row["ratio_percent"]}.{key}'] = row[key]
    for key in ['best_grid_f_score','best_grid_adjust_f_score','auc_roc','auc_pr']:
        result['test_label_selected_diagnostic.'+key] = tab[key]
    return result


def scalar_endpoints(vector):
    return {k:v for k,v in vector.items() if '.entity.' not in k and not k.endswith(('.events','.detected_events'))}


def difference(left, right):
    if set(left) != set(right): raise ValueError('Paired endpoints differ')
    return {key:None if left[key] is None or right[key] is None else left[key]-right[key] for key in left}


def macro_vectors(rows):
    if len(rows) != 7 or any(len(v) != 5 for v in rows): raise ValueError('All seven datasets and five matched seeds required')
    rows = [[scalar_endpoints(v) for v in group] for group in rows]
    keys = set(rows[0][0])
    if any(set(v) != keys for group in rows for v in group): raise ValueError('Macro endpoint schemas differ')
    return [{key:None if any(group[index][key] is None for group in rows)
        else statistics.mean(group[index][key] for group in rows) for key in rows[0][0]} for index in range(5)]


def verify(runtime):
    source = read(ROOT/PRODUCER)
    if (runtime['producer'] != PRODUCER or runtime['producer_sha256'] != sha(ROOT/PRODUCER)
            or runtime['settings'] != source['settings'] or runtime['resource_lock'] != source['resource_lock']
            or runtime['original_scope'] != source['original_scope'] or runtime['required_cases'] != 440
            or runtime['required_tasks'] != 206 or runtime['seeds'] != SEEDS or runtime['datasets'] != DATASETS
            or runtime['student_t_df4_975'] != T_CRITICAL or runtime['primary_endpoint'] != 'strict.1.micro.f1'
            or runtime['python'] != str(ROOT/read(ROOT/producer.QUEUES['original'][0])['python'])):
        raise ValueError('Complete frozen source, statistics and resource contracts required')
    for pin in runtime['source_receipts']:
        if sha(ROOT/pin['path']) != pin['sha256']: raise ValueError('Statistics dependency changed: '+pin['path'])
    cases = producer.verify(source); selected = tasks(cases)
    if runtime['tasks'] != list(selected.values()): raise ValueError('Full task bindings differ')
    return source,cases,selected


def phase_paths(source, identity, phase):
    base = Path(source['artifact_root'])/identity
    proof = base/('native/native_replay.json' if phase == 'native' else 'independent/review.json')
    guard = ROOT/source['state_root']/'attempts'/identity/phase/'guard_receipt.json'
    return proof,guard


def readiness(source, cases, task):
    """Return pending or ready; actual terminal negative evidence is not retried."""
    negative = {}
    ready = True
    for identity in task['cases']:
        if identity in source['preserved_original_failures']:
            negative[identity] = source['preserved_original_failures'][identity]
            continue
        case = cases[identity]
        for path, guard in producer.source_guard_records(source,case):
            if guard.get('exit_code') is not None and (guard['exit_code'] != 0 or guard.get('resource_guard_aborted')):
                negative[identity] = dict(source_guard=str(path),sha256=sha(path),outcome='terminal_source_failure')
        phases = ['independent'] if identity == source['existing_native_certificate']['id'] else ['native','independent']
        for phase in phases:
            proof,guard = phase_paths(source,identity,phase)
            if guard.exists():
                g = read(guard)
                if not g['passed'] or g['exit_code'] != 0 or g['resource_guard_aborted']:
                    negative[identity] = dict(phase=phase,guard=str(guard),sha256=sha(guard),outcome='preserved_failed_full_verification')
                elif not proof.exists(): raise ValueError('Passing guard with missing full proof')
            else: ready = False
    return ('unestimable',negative) if negative else ('ready' if ready else 'pending',{})


def full_case(source, case):
    result,dataset,binding = producer.check_source(source,case)
    identity = case['job']['id']; reused = identity == source['existing_native_certificate']['id']
    proof_paths = {}
    for phase in (['independent'] if reused else ['native','independent']):
        path,guard_path = phase_paths(source,identity,phase)
        proof,guard = read(path),read(guard_path)
        command = [str(ROOT/case['queue']['python']),'-B','-X','utf8','-u','-m',producer.MODULE,
            '--config',PRODUCER,'--stage',phase,'--job-id',identity]
        if (not guard['passed'] or guard['exit_code'] != 0 or guard['resource_guard_aborted']
                or guard['proof_sha256'] != sha(path) or guard['exact_child_handle']['command'] != command
                or not proof['passed'] or proof['job'] != case['job'] or proof['source_binding'] != binding):
            raise ValueError('Unchanged guarded complete native and independent proofs required')
        proof_paths[phase] = dict(path=str(path),sha256=sha(path),guard_sha256=sha(guard_path))
    independent = read(proof_paths['independent']['path'])
    if (not independent['all_original_arrays_and_labels_reviewed']
            or not independent['metrics_events_and_bootstrap_replayed_exactly']
            or independent['training_updates_performed'] != 0
            or independent['existing_native_certificate_reused'] != reused):
        raise ValueError('Full independent verification required')
    if reused:
        if independent['native_proof_binding'] != producer.reused_native(source,case,result):
            raise ValueError('Reused original native proof changed')
    else:
        native = read(proof_paths['native']['path'])
        if (native['runtime_sha256'] != sha(ROOT/PRODUCER) or not native['all_original_arrays_identical']
                or not native['all_native_groups_reinferred'] or native['fit_calls'] != 0
                or native['training_updates_performed'] != 0 or native['math_tolerance'] != source['math_tolerance']
                or sha(native['scores_path']) != native['scores_sha256']
                or independent['native_proof_binding'] != {k:proof_paths['native'][k] for k in ['path','sha256']}):
            raise ValueError('Complete unchanged full native inference proof required')
    return result,dataset,dict(id=identity,seed=case['job']['seed'],family=case['family'],
        source_binding=binding,proofs=proof_paths,
        ideal_float64_path_equivalence_certified=independent['ideal_float64_path_equivalence_certified'],
        full_native_split_coverage=independent['full_native_split_coverage'])


def support(dataset, evidence):
    import numpy as np
    from scripts.flow_matching.anomaly_transformer_independent_math_v1 import array_digest
    rows = []
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as arrays:
        for entity in dataset['entities']:
            prefix = entity['array_prefix']
            row = dict(entity_id=entity['entity_id'],metadata={})
            for suffix in ['_labels','_timestamps','_source_rows','_fault_labels']:
                if prefix+suffix in arrays.files: row['metadata'][suffix] = array_digest(arrays[prefix+suffix])
            if '_labels' not in row['metadata']: raise ValueError('Complete test labels required for paired support')
            rows.append(row)
    return dict(entities=rows,coverage=evidence['full_native_split_coverage'])


def worker(runtime, source, cases, task):
    if readiness(source,cases,task)[0] != 'ready': raise ValueError('Every required full proof must be ready')
    vectors,evidence,supports = {},{},{}
    for identity in dict.fromkeys(task['cases']):
        result,dataset,item = full_case(source,cases[identity])
        vectors[identity] = metrics_vector(result['metrics']); evidence[identity] = item
        if dataset['input_sha256'] not in supports:
            supports[dataset['input_sha256']] = support(dataset,item)
        item['support'] = supports[dataset['input_sha256']]
    ids = task['cases']
    if task['kind'] == 'paired':
        data = []
        for left,right in zip(ids[:5],ids[5:]):
            if (cases[left]['job']['seed'] != cases[right]['job']['seed']
                    or cases[left]['job']['dataset'] != cases[right]['job']['dataset']
                    or evidence[left]['support'] != evidence[right]['support']):
                raise ValueError('Exact matching original seed,labels,rows,timestamps and full split coverage required')
            data.append(difference(vectors[left],vectors[right]))
    elif task['kind'] == 'macro':
        data = macro_vectors([[vectors[i] for i in ids[k:k+5]] for k in range(0,35,5)])
    else:
        if any(evidence[i]['support'] != evidence[ids[0]]['support'] for i in ids):
            raise ValueError('All five seeds must cover the identical full grouped data')
        data = [vectors[i] for i in ids]
    output = Path(runtime['artifact_root'])/task['id']/'statistics.json'
    if output.exists(): raise FileExistsError('Preserve prior statistics attempt')
    write(output,dict(passed=True,task=task,runtime_sha256=sha(ROOT/runtime['_config_path']),
        vectors_in_original_seed_order=data,statistics=summarize(data),full_case_evidence=evidence,
        primary_endpoint=runtime['primary_endpoint'],test_pooled_PA_and_grid_controls_are_diagnostic=True,
        source_temporal_bootstrap_intervals_not_reinterpreted_as_seed_repetitions=True,
        original_failures_not_replaced_by_repair_results=True,raw_preprocessing_values_independently_reconstructed=False,
        no_new_fit_inference_or_seed=True,paper_equivalence_certified=False,strict_TAB_harness_equivalence=False,
        all_original_experiments_complete=False))


def controller(runtime, source, cases, selected):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api = install_facades(); base = ROOT/runtime['state_root']; base.mkdir(parents=True,exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('Statistics controller already live')
    publish = status_writer(base/'status.json',write)
    completed,unestimable,failed = [],{},{}
    current = {}
    common = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_cases=440,
        required_tasks=206,original_scope=runtime['original_scope'],all_original_experiments_complete=False)
    def update(value=None,**kw):
        current.update(value or {}); current.update(kw)
        row = dict(common,completed_tasks=list(completed),completed_count=len(completed),
            terminal_unestimable=unestimable,unestimable_count=len(unestimable),failed_statistics=failed,
            captured_utc=datetime.now(timezone.utc).isoformat(),**current)
        publish(base/'status.json',row)
    try:
        while len(completed)+len(unestimable)+len(failed) < len(selected):
            for identity,task in selected.items():
                if identity in completed or identity in unestimable or identity in failed: continue
                state,negative = readiness(source,cases,task)
                if state == 'unestimable': unestimable[identity] = negative; continue
                if state != 'ready': continue
                attempt = base/'attempts'/identity; output = Path(runtime['artifact_root'])/identity/'statistics.json'
                try:
                    if attempt.exists() or output.exists(): raise ValueError('Prior statistics attempt preserved; no automatic retry')
                    def waiting(value): update(value,state='waiting_for_memory',active_pid=None,job=identity)
                    with api['resource_slot'](ROOT/runtime['resource_lock'],runtime['settings'],waiting,poll_seconds=10):
                        attempt.mkdir(parents=True,exist_ok=False)
                        command = [runtime['python'],'-B','-X','utf8','-u','-m',MODULE,'--config',runtime['_config_path'],'--worker','--task-id',identity]
                        with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            handle = dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command)
                            write(attempt/'launch.json',handle); update(state='running_full_proof_bound_statistics',active_pid=child.pid,job=identity)
                            code,usage = capped_wait(child,runtime,api,update)
                        passed = code == 0 and not usage['resource_guard_aborted'] and output.exists() and read(output).get('passed') is True
                        write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,
                            proof_sha256=sha(output) if output.exists() else None,exact_child_handle=handle))
                        if not passed: raise ValueError('Full statistics guard or worker failed; preserved without retry')
                    completed.append(identity)
                except Exception as error: failed[identity] = str(error)
            update(state='waiting_all_five_actual_full_model_proofs',active_pid=None,job=None)
            time.sleep(10)
        update(state='all206_tasks_terminal_statistics_or_preserved_failure_other_experiments_remain',active_pid=None,job=None)
    finally: lane.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True)
    p.add_argument('--probe-only',action='store_true'); p.add_argument('--worker',action='store_true'); p.add_argument('--task-id')
    args=p.parse_args(); runtime=read(ROOT/args.config); source,cases,selected=verify(runtime); runtime['_config_path']=args.config
    if args.probe_only:
        print(json.dumps(dict(passed=True,cases=len(cases),tasks=len(selected),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,numerical_workers_started=0)))
    elif args.worker: worker(runtime,source,cases,selected[args.task_id])
    else: controller(runtime,source,cases,selected)


if __name__ == '__main__': main()
