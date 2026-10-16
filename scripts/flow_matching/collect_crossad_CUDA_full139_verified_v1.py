"""Independently bind all139 CUDA results and complete five-seed cohorts.

No numerical model imports. Complete compressed source arrays are compared to
fresh native replay arrays. Partial cohorts never produce means or intervals.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import zipfile

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.collect_crossad_CUDA_full139_verified_v1'
CONFIG='configs/reproducibility/fm_crossad_CUDA_full139_independent_results.v1.json'
QUEUE='configs/experiments/fm_crossad_original_CUDA_full139.v2.json'
SEEDS=[1103,1104,1105,1106,1107]
T95_DF4=2.7764451051977987


def registration():
    q=read(ROOT/QUEUE)
    first=q['delegated_first_configuration']
    paths=[QUEUE,first,MODULE.replace('.','/')+'.py','tests/test_fm_crossad_CUDA_full139_verified_v1.py']
    return dict(queue=QUEUE,queue_sha256=sha(ROOT/QUEUE),required_conditions=139,
        primary_seeds=SEEDS,expected_complete_primary_cohorts=22,
        Student_t95_df4=T95_DF4,interval_interpretation='Across the five fixed model seeds on one frozen split;not unseen-entity or historical-paper uncertainty',
        original_scope=q['original_scope'],source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],
        state_root='experiments/runs/fm_crossad_CUDA_full139_independent_results_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/crossad_CUDA_full139_independent_results_v1',
        citation='https://arxiv.org/abs/2510.12489',new_training_seed_slots=0,
        original_CPU_completion_credit=0,strict_TAB_result=False,paper_equivalence=False,
        all_original_experiments_complete=False)


def npy_header(stream):
    if stream.read(6)!=b'\x93NUMPY':raise ValueError('Actual NPY array required')
    version=stream.read(2)
    if version==b'\x01\x00':length=struct.unpack('<H',stream.read(2))[0]
    elif version in [b'\x02\x00',b'\x03\x00']:length=struct.unpack('<I',stream.read(4))[0]
    else:raise ValueError('Unsupported frozen native NPY version')
    if length>65536:raise ValueError('Invalid native NPY header')
    header=ast.literal_eval(stream.read(length).decode('utf-8' if version==b'\x03\x00' else 'latin1').strip())
    if set(header)!={'descr','fortran_order','shape'} or header['fortran_order'] or len(header['shape'])!=1:
        raise ValueError('Complete one-dimensional original native score/label/row array required')
    return header


def complete_array(left,right,points):
    a,b=npy_header(left),npy_header(right)
    if a!=b or a['shape']!=(points,) or a['descr'] not in ['<f4','<f8']:
        raise ValueError('Every score dtype and full-population shape must match')
    size=4 if a['descr']=='<f4' else 8;fmt='<f' if size==4 else '<d';remaining=points*size
    while remaining:
        count=min(65536,remaining);x,y=left.read(count),right.read(count)
        if len(x)!=count or x!=y or not all(math.isfinite(v[0]) for v in struct.iter_unpack(fmt,x)):
            raise ValueError('Complete source and independently replayed score bytes differ/nonfinite')
        remaining-=count
    if left.read(1) or right.read(1):raise ValueError('Unexpected trailing score bytes')
    return dict(points=points,dtype=a['descr'],all_score_payload_bytes_identical=True,all_finite=True)


def labels_and_rows(archive,points,start):
    digest=hashlib.sha256();positive=0
    with archive.open('labels.npy') as stream:
        h=npy_header(stream)
        if h['shape']!=(points,) or h['descr']!='|u1':raise ValueError('Every full native binary test label required')
        remaining=points
        while remaining:
            part=stream.read(min(65536,remaining))
            if not part or any(v not in [0,1] for v in part):raise ValueError('Incomplete/nonbinary original label array')
            digest.update(part);positive+=sum(part);remaining-=len(part)
        if stream.read(1):raise ValueError('Unexpected trailing label bytes')
    row_digest=hashlib.sha256();offset=0
    with archive.open('source_rows.npy') as stream:
        h=npy_header(stream)
        if h['shape']!=(points,) or h['descr']!='<i8':raise ValueError('All original source row identifiers required')
        remaining=points*8
        while remaining:
            part=stream.read(min(65536,remaining))
            if not part or len(part)%8:raise ValueError('Incomplete original source rows')
            row_digest.update(part)
            for value, in struct.iter_unpack('<q',part):
                if value!=start+offset:raise ValueError('Original source row order/boundary changed')
                offset+=1
            remaining-=len(part)
        if stream.read(1):raise ValueError('Unexpected trailing row bytes')
    return dict(full_test_points=points,positive_labels=positive,labels_payload_sha256=digest.hexdigest(),
        source_rows_payload_sha256=row_digest.hexdigest(),all_source_rows_verified=True)


def same_metrics(a,b,tolerance):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Every metric/uncertainty field required')
        for key in a:same_metrics(a[key],b[key],tolerance)
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):raise ValueError('Every metric list coordinate required')
        for x,y in zip(a,b):same_metrics(x,y,tolerance)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=tolerance['rtol'],abs_tol=tolerance['atol']):
            raise ValueError('Full native metric/uncertainty differs')
    elif a!=b:raise ValueError('Metric provenance or coverage differs')


def checked_guard(path):
    guard=read(path)
    if not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']:
        raise ValueError('Actual numerical guard did not pass')
    if sha(guard['report_path'])!=guard['report_sha256']:raise ValueError('Actual numerical report changed')
    return guard


def condition(c,q,job):
    canonical=job['canonical_CPU_condition_reference']
    if canonical==q['delegated_first_original_condition']:
        first=read(ROOT/q['delegated_first_configuration']);state=ROOT/first['state_root'];expected=first['job']
        guards=[state/(phase+'.guard_receipt.json') for phase in ['train','replay']]
        artifact=Path(first['artifact_root'])/'native_replay'
        score_name=lambda role:role+'_scores.npy'
        prefix_complete_key='full_test_labels_and_source_rows_equal'
        metrics_complete_key='full_native_validation_metrics_and_uncertainty_recomputed'
    else:
        state=ROOT/q['state_root']/'jobs'/canonical;expected=job
        guards=[state/(phase+'.guard.json') for phase in ['CUDA_probe','CPU_reference','author','replay']]
        artifact=Path(q['artifact_root'])/canonical/'native_replay'
        score_name=lambda role:role+'.npy'
        prefix_complete_key='all_labels_source_rows_and_tail_points_verified'
        metrics_complete_key='all_native_validation_metrics_and_uncertainty_recomputed'
    if not all(p.exists() for p in guards):return None
    actual=[checked_guard(path) for path in guards]
    source=read(actual[-2]['report_path']);native=read(actual[-1]['report_path'])
    source_path=ROOT/expected['output_directory']/'result.json'
    if (source['status']!='completed' or source['id']!=expected['id']
            or source['experiment_sha256']!=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
            or native['source_result_sha256']!=sha(source_path) or not native['passed']
            or not native[prefix_complete_key] or not native[metrics_complete_key]):
        raise ValueError('Full original source condition and actual checkpoint replay required')
    for item in source['artifacts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('A complete source artifact changed')
    proof={v['role']:v for v in native['all_roles']}
    counts=source['details']['native_points']
    if set(proof)!={'train','validation','test'} or set(counts)!=set(proof):raise ValueError('Every original split required')
    arrays=[]
    with zipfile.ZipFile(ROOT/expected['output_directory']/'native_scores.npz') as saved:
        if set(saved.namelist())!={'train.npy','validation.npy','test.npy','labels.npy','source_rows.npy'}:
            raise ValueError('Every full native score, label and row array required')
        for role in ['train','validation','test']:
            path=artifact/score_name(role)
            if (not proof[role]['all_score_bytes_equal'] or proof[role]['points']!=counts[role]
                    or sha(path)!=proof[role]['score_file_sha256']):
                raise ValueError('Native replay array receipt differs')
            with saved.open(role+'.npy') as left,path.open('rb') as right:
                arrays.append(dict(role=role,**complete_array(left,right,counts[role])))
        labels=labels_and_rows(saved,counts['test'],expected['train_length'])
    same_metrics(source['metrics']['native_validation_control'],native['native_validation_control'],q['array_tolerance'])
    parameters=source['details']['declared_train_parameters']
    if parameters!=expected['train_parameters']:raise ValueError('Original full epoch/batch/earlystop/LR recipe differs')
    return dict(canonical_CPU_condition_reference=canonical,actual_execution_id=expected['id'],dataset=job['dataset'],seed=job['seed'],
        mode=job['mode'],ablation_row=job.get('ablation_row'),reconstructed_ablation=job['reconstructed_ablation'],
        source_result_sha256=sha(source_path),actual_guard_sha256=[sha(p) for p in guards],
        native_report_sha256=sha(actual[-1]['report_path']),full_original_score_arrays_independently_compared=arrays,
        label_and_source_row_review=labels,raw_sha256=expected['raw_sha256'],native_points=counts,
        native_validation_control=source['metrics']['native_validation_control'],original_author_metrics=source['metrics']['original_author'],
        actual_epochs=source['details']['epochs_completed'],original_train_parameters=parameters,
        original_CPU_completion_credit=0,strict_TAB_result=False,paper_equivalence=False)


def interval(values):
    if len(values)!=5 or not all(math.isfinite(x) for x in values):raise ValueError('Exactly all five finite seed values required')
    mean=statistics.mean(values);sd=statistics.stdev(values);half=T95_DF4*sd/math.sqrt(5)
    return dict(n=5,mean=mean,seed_sample_SD=sd,mean_t95_interval=[mean-half,mean+half])


def metric_vector(record):
    values={}
    for ratio,item in record['native_validation_control']['strict'].items():
        for key in ['f1','precision','recall','auroc','average_precision']:
            if item['micro'].get(key) is not None:values['validation_false_alarm_percent/'+ratio+'/micro/'+key]=item['micro'][key]
        values['validation_false_alarm_percent/'+ratio+'/macro_f1']=item['macro_f1']
    return values


def cohort_key(job):return job['dataset']+'/'+('main' if not job['reconstructed_ablation'] else 'paper_reconstruction_row'+str(job['ablation_row']))


def cohorts(jobs,verified):
    planned={}
    for job in jobs:
        if job['mode']=='fresh_train' and job['seed'] in SEEDS:planned.setdefault(cohort_key(job),{})[job['seed']]=job['canonical_CPU_condition_reference']
    result={}
    for name,seeds in sorted(planned.items()):
        if set(seeds)!=set(SEEDS):raise ValueError('All five original planned seeds must be present')
        present={seed:verified[identity] for seed,identity in seeds.items() if identity in verified}
        row=dict(primary_seeds=SEEDS,verified_seed_ids=sorted(present),complete=len(present)==5)
        if row['complete']:
            signatures={json.dumps(dict(raw=v['raw_sha256'],points=v['native_points'],labels=v['label_and_source_row_review']),sort_keys=True) for v in present.values()}
            if len(signatures)!=1:raise ValueError('Five matched seeds must share the full original split/data/labels/rows')
            vectors={seed:metric_vector(present[seed]) for seed in SEEDS}
            if len({tuple(sorted(v)) for v in vectors.values()})!=1:raise ValueError('Full native metric coverage differs across seeds')
            row['metrics']={key:interval([vectors[seed][key] for seed in SEEDS]) for key in vectors[SEEDS[0]]}
            row['metric_vectors_by_seed']=vectors
        result[name]=row
    paired={}
    for name,row in result.items():
        main=result[name.split('/')[0]+'/main']
        if '/paper_reconstruction_row' in name and row['complete'] and main['complete']:
            paired[name+'_minus_main']={key:interval([row['metric_vectors_by_seed'][seed][key]-main['metric_vectors_by_seed'][seed][key] for seed in SEEDS]) for key in row['metrics']}
    return dict(primary_cohorts=result,paired_seed_ablation_differences=paired,
        complete_primary_cohorts=sum(row['complete'] for row in result.values()),
        historical_seed2025_and_release_checks_excluded_from_primary_means=True,
        intervals_descriptive_no_multiple_comparison_significance_claim=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG)
    if c!=registration():raise ValueError('Full139/five-seed/source/interval contract changed')
    for item in c['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen collector dependency changed')
    q=read(ROOT/c['queue'])
    if (len(q['jobs'])!=139 or len({j['canonical_CPU_condition_reference'] for j in q['jobs']})!=139
            or len(cohorts(q['jobs'],{})['primary_cohorts'])!=22):
        raise ValueError('All139 original conditions and all22 primary five-seed cohorts required')
    if args.probe_only:
        print(json.dumps(dict(required_conditions=139,primary_cohorts=len(cohorts(q['jobs'],{})['primary_cohorts']),
            numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,completed_conditions=0)));return
    import psutil
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    base=ROOT/c['state_root'];base.mkdir(parents=True,exist_ok=True);lock=exclusive_lock(base/'controller.lock')
    if lock is None or (base/'status.json').exists():
        if lock:lock.close()
        raise FileExistsError('Never duplicate or overwrite a collector attempt')
    out=Path(c['artifact_root']);out.mkdir(parents=True,exist_ok=False);verified={};errors={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),required_original_conditions=139,all_original_experiments_complete=False)
    def update():
        summary=cohorts(q['jobs'],verified)
        write(out/'complete_primary_cohorts.json',dict(summary,verified_original_CUDA_conditions=len(verified),
            new_training_seed_slots=0,original_CPU_completion_credit=0,strict_TAB_result=False,all_original_experiments_complete=False))
        write(base/'status.json',dict(state,state='waiting_actual_full_source_and_checkpoint_arrays' if len(verified)<139 else 'all139_original_CUDA_results_independently_verified_other_experiments_pending',
            verified_original_CUDA_ids=list(verified),verified_count=len(verified),complete_primary_cohorts=summary['complete_primary_cohorts'],
            errors_preserved=errors,captured_utc=datetime.now(timezone.utc).isoformat()))
    try:
        while len(verified)<139:
            for job in q['jobs']:
                identity=job['canonical_CPU_condition_reference']
                if identity in verified or identity in errors:continue
                try:
                    proof=condition(c,q,job)
                    if proof is None:continue
                    exclusive(out/'conditions'/(identity+'.json'),proof);verified[identity]=proof
                    update()
                except Exception as error:errors[identity]=repr(error)
            update();time.sleep(10)
    finally:lock.close()


if __name__=='__main__':main()
