"""Independently review every released-weight value and all untouched records."""
import argparse
from collections import defaultdict
import csv
import json

import numpy as np
from sklearn.metrics import precision_recall_curve,roc_auc_score,average_precision_score

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT,read,digest


def require(value,message):
    if not value:raise ValueError(message)


def load_rows(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);return reader.fieldnames,list(reader)


def key(row):return tuple(row[k] for k in ('algorithm','dataset','task','protocol','metric'))


def review(config_path):
    config=read(config_path)
    for item in config['source_receipts']:require(digest(item['path'])==item['sha256'],'Frozen review source changed')
    publication=read(config['publication_config'])
    for item in publication['source_receipts']:require(digest(item['path'])==item['sha256'],'Frozen publication source changed')
    base=ROOT/publication['base_publication'];target=ROOT/publication['output_directory'];receipt=read(target/'publication_validation.json')
    for name,expected in receipt['outputs'].items():require(digest(target/name)==expected,'Published output changed')
    h,old=load_rows(base/'all_algorithm_dataset_metrics.csv');newh,rows=load_rows(target/'all_algorithm_dataset_metrics.csv')
    require(h==newh and len(h)==31 and len(old)==len(rows)==8098,'All8098scientific slots and31fields required')
    changed=[i for i,(a,b) in enumerate(zip(old,rows)) if a!=b]
    require(len(changed)==32 and changed==receipt['changed_row_positions'],'Exactly32 previously missing released-weight records required')
    require(rows[:7283]==old[:7283],'All previous7283records and capture times must be exact')
    require((target/'all_algorithm_dataset_metrics.csv').read_bytes().startswith((ROOT/publication['preserved_original7283_csv']).read_bytes()),'Original7283CSV prefix must be byte exact')
    for i in changed:
        a,b=old[i],rows[i]
        require(not a['mean'] and a['n']=='0' and b['n']=='1' and b['required_repeats']=='1','Only missing fixed-weight slots may be filled')
        require(b['algorithm']=='crossad_complete/release_checkpoint' and b['dataset'] in ['GECCO','MSL'],'Exactly original completed releases required')
        require(b['author_equivalence_certified']=='False' and b['strict_TAB_result']=='False','Native flat source is not a certified strict leaderboard')
        require(all(not b[k] for k in ['std','se','ci95_low','ci95_high']),'Fixed released weights have no training-seed interval')
        require({k:v for k,v in a.items() if k not in ['mean','n','captured_utc','source','uncertainty_scope']}==
            {k:v for k,v in b.items() if k not in ['mean','n','captured_utc','source','uncertainty_scope']},'Scientific dimensions changed')
    queue=read(publication['queue']);groups=defaultdict(list)
    require(len(queue['jobs'])==139 and len({j['id'] for j in queue['jobs']})==139,'Every original native job required')
    for job in queue['jobs']:groups[job['author_recipe'],job['dataset']].append(job)
    require(len(groups)==29,'All original29cohorts required')
    native=[r for r in rows if r['algorithm'].startswith('crossad_complete/')]
    require(len(native)==841 and len({key(r) for r in native})==841,'Every841native indicator required')
    _,grid=load_rows(target/'crossad_complete_native_metric_grid.csv')
    require({key(r):r for r in native}=={key(r):r for r in grid},'Complete native grid differs from total table')
    for (recipe,dataset),jobs in groups.items():
        seeds=[2025] if recipe=='release_checkpoint' else [2025,1103,1104,1105,1106,1107]
        require(sorted(j['seed'] for j in jobs)==sorted(seeds),'Original seed roster changed')
        cohort=[r for r in native if r['algorithm']=='crossad_complete/'+recipe and r['dataset']==dataset]
        require(len(cohort)==29 and all(int(r['required_repeats'])==len(seeds) for r in cohort),'Full cohort obligations changed')
    checked=0
    for case in publication['completed_releases']:
        point=read(case['point_audit']);model=read(case['model_replay']);guard=read(case['guard_receipt'])
        require(point['passed'] and point['full_raw_labels_and_source_rows_checked'] and point['completed_training_seed_slots']==0,'Complete raw-label release audit required')
        require(model['full_original_model_reinference_passed'] and model['full_original_author_metric_reexecution_passed']
            and model['original_full_state_dictionary_loaded_strictly'] and model['fixed_release_checkpoint_evaluation']
            and model['original_SPOT_init_model_inference_and_threshold_reexecution_passed']
            and model['completed_training_seed_slots']==0 and model['additional_training_seed_slots']==0,'Full native fixed-weight replay required')
        require(guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_native_model_replay_passed'],'Complete guarded execution required')
        for name,expected in model['replay_outputs'].items():require(digest(ROOT/case['model_artifact_directory']/name)==expected,'Actual replay artifact changed')
        job=next(j for j in queue['jobs'] if j['id']==model['job_id']);source=ROOT/job['output_directory']
        require(job['dataset']==case['dataset'] and job['mode']=='release_checkpoint' and point['job_id']==job['id'],'Original released-weight job required')
        source_result=read(source/'result.json')
        require(digest(source/'result.json')==model['source_result_sha256'] and source_result['details']['epochs_completed']==0,'No source training may be introduced')
        lookup={(r['protocol'],r['metric']):r for r in native if r['dataset']==case['dataset'] and r['algorithm']=='crossad_complete/release_checkpoint'}
        require(len(lookup)==29 and all(r['n']=='1' and r['mean'] for r in lookup.values()),'All29fixed-weight indicators must be available')
        def compare(protocol,metric,value):
            nonlocal checked
            row=lookup[protocol,metric]
            require(np.isclose(float(row['mean']),value,rtol=1e-12,atol=1e-12),'Independent full metric mismatch: '+case['dataset']+'/'+protocol+'/'+metric)
            checked+=1
        arrays=np.load(source/'native_scores.npz',allow_pickle=False);test=arrays['test'].astype(float);validation=arrays['validation'].astype(float);labels=arrays['labels'].astype(int)
        require({role:len(arrays[role]) for role in ['train','validation','test']}==point['full_native_points']==model['full_score_points'],'All original full points required')
        require(np.array_equal(arrays['source_rows'],np.arange(job['train_length'],job['train_length']+len(labels))),'Full original source rows required')
        for ratio,percentile in [(.1,'99.9'),(1,'99'),(5,'95')]:
            predicted=test>np.percentile(validation,100-ratio,method='linear')
            tp=int(np.sum(predicted&(labels==1)));fp=int(np.sum(predicted&(labels==0)));fn=int(np.sum(~predicted&(labels==1)))
            values=dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,auroc=roc_auc_score(labels,test),average_precision=average_precision_score(labels,test))
            for metric,value in values.items():compare('release_checkpoint_validation'+percentile+'_no_PA',metric,value)
        directory=source/'workspace/test_results'/job['dataset']/'results_0'
        energy=np.load(directory/'a_test_energy.npy',allow_pickle=False).mean(axis=-1);truth=np.load(directory/'a_gt.npy',allow_pickle=False).reshape(-1).astype(int)
        require(len(energy)==source_result['details']['original_scored_test_points'] and np.array_equal(truth,labels[:len(energy)]),'Complete original nonoverlap scores required')
        p,r,thresholds=precision_recall_curve(truth,energy);f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r!=0);best=int(np.argmax(f))
        for metric,value in dict(precision=p[best],recall=r[best],f1=f[best],accuracy=np.mean((energy>thresholds[best])==truth)).items():compare('release_checkpoint_test_label_oracle',metric,value)
        compare('release_checkpoint_author_evaluation','auroc',roc_auc_score(truth,energy));compare('release_checkpoint_author_evaluation','average_precision',average_precision_score(truth,energy))
        for stem,fields in [('_r_auc',['R_AUC_ROC','R_AUC_PR']),('_vus',['VUS_ROC','VUS_PR']),('_affiliation',['Affiliation_ACC','Affiliation_P','Affiliation_R','Affiliation_F1'])]:
            with (directory/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as stream:original=list(csv.DictReader(stream))
            require(len(original)==1,'One complete source metric row required')
            for field in fields:
                value=float(original[0][field])
                require(np.isclose(value,model['original_author_metrics'][stem][0][field],rtol=1e-12,atol=1e-12),'Original full evaluator replay differs')
                compare('release_checkpoint_author_evaluation',field,value)
    require(checked==58 and sum(bool(r['mean']) for r in rows)==2120,'All58native release values and2120total measured rows required')
    for file in base.iterdir():
        if file.is_file() and file.name not in {'all_algorithm_dataset_metrics.csv','crossad_complete_native_metric_grid.csv','all_metrics.md','results.html','README.rst','publication_validation.json'}:
            require(file.read_bytes()==(target/file.name).read_bytes(),'Other track export changed')
    require(receipt['additional_training_seed_slots']==0 and receipt['newly_completed_canonical_training_slots']==0,'Released weights never create new training evidence')
    report=dict(passed=True,publication=publication['output_directory'],registered_metric_rows=8098,
        measured_metric_rows=2120,missing_metric_rows=5978,changed_missing_release_slots=32,
        independently_checked_complete_release_values=58,other8066rows_and_capture_times_exact=True,
        original7283CSVprefix_byte_exact=True,all139native_jobs_and29cohorts_retained=True,
        new_training_seed_slots=0,all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False)
    output=ROOT/config['output'];output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);review(parser.parse_args().config)
