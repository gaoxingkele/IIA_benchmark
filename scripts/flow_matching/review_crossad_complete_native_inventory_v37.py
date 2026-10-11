"""Independently check the complete native inventory and full-point metrics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import ROOT,read,digest


def require(value,message):
    if not value:raise ValueError(message)


def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        return reader.fieldnames,list(reader)


def key(row):return tuple(row[k] for k in ('algorithm','dataset','task','protocol','metric'))


def review(config_path):
    config=read(config_path)
    for item in config['source_receipts']:
        require(digest(item['path'])==item['sha256'],'Pinned review source changed')
    publication=read(config['publication_config']);base=ROOT/publication['base_publication']
    target=ROOT/publication['output_directory'];receipt=read(target/'publication_validation.json')
    for name,expected in receipt['outputs'].items():require(digest(target/name)==expected,'Publication output changed')
    for item in publication['source_receipts']:
        require(digest(item['path'])==item['sha256'],'Publication source changed')
    header,old=rows(base/'all_algorithm_dataset_metrics.csv');newheader,allrows=rows(target/'all_algorithm_dataset_metrics.csv')
    require(len(header)==31 and header==newheader and len(old)==7283 and len(allrows)==8098,'Full original geometry required')
    require(allrows[:7283]==old,'Every old field and capture time must remain')
    require((target/'all_algorithm_dataset_metrics.csv').read_bytes().startswith((base/'all_algorithm_dataset_metrics.csv').read_bytes()),'Original CSV prefix must be byte exact')
    for file in base.iterdir():
        if file.name not in {'all_algorithm_dataset_metrics.csv','all_metrics.md','results.html','README.rst','publication_validation.json'}:
            require(file.read_bytes()==(target/file.name).read_bytes(),'Other export changed')
    queue=read(publication['queue']);groups=defaultdict(list)
    require(len(queue['jobs'])==139 and len({j['id'] for j in queue['jobs']})==139,'139 original jobs required')
    for job in queue['jobs']:groups[job['author_recipe'],job['dataset']].append(job)
    require(len(groups)==29 and queue['settings']['evaluation']['strict_false_alarm_percent']==[.1,1,5],'29 cohorts and all operating points required')
    native=[r for r in allrows if r['algorithm'].startswith('crossad_complete/')]
    require(len(native)==841 and len({key(r) for r in native})==841,'841 unique native slots required')
    _,grid=rows(target/'crossad_complete_native_metric_grid.csv')
    require({key(r):r for r in native}=={key(r):r for r in grid},'Complete native grid mismatch')
    for (recipe,dataset),jobs in groups.items():
        expected=[2025] if recipe=='release_checkpoint' else [2025,1103,1104,1105,1106,1107]
        require(sorted(j['seed'] for j in jobs)==sorted(expected),'All original unique seeds required')
        cohort=[r for r in native if r['algorithm']=='crossad_complete/'+recipe and r['dataset']==dataset]
        require(len(cohort)==29 and all(int(r['required_repeats'])==len(expected) for r in cohort),'Complete metric obligations required')
        expected_metrics={(recipe+'_test_label_oracle',m) for m in ['precision','recall','f1','accuracy']}
        expected_metrics|={(recipe+'_author_evaluation',m) for m in ['auroc','average_precision','R_AUC_ROC','R_AUC_PR','VUS_ROC','VUS_PR','Affiliation_ACC','Affiliation_P','Affiliation_R','Affiliation_F1']}
        for percentile in ['99.9','99','95']:
            expected_metrics|={(recipe+'_validation'+percentile+'_no_PA',m) for m in ['precision','recall','f1','auroc','average_precision']}
        require({(r['protocol'],r['metric']) for r in cohort}==expected_metrics,'Native protocols or metrics omitted')
    tail=allrows[7283:];measured=[r for r in tail if r['mean']]
    require(len(tail)==815 and len(measured)==29,'Exactly815 appended obligations and29 measured metrics required')
    for row in tail:
        require(row['author_equivalence_certified']=='False' and row['strict_TAB_result']=='False','Uncertified source must remain separate')
        require(all(not row[k] for k in ['std','se','ci95_low','ci95_high']),'Single-seed results have no seed confidence interval')
        require(row['n']==('1' if row['mean'] else '0'),'Missing values must remain unavailable')
    require(all(r['algorithm']=='crossad_complete/fresh_train' and r['dataset']=='MSL' and r['required_repeats']=='6' for r in measured),'Only audited MSL one-of-six training seed required')
    point=read(publication['completed_point_audit']);model=read(publication['completed_model_replay']);guard=read(publication['completed_model_guard_receipt'])
    require(point['passed'] and model['full_original_model_reinference_passed'] and model['full_original_author_metric_reexecution_passed'] and model['original_full_state_dictionary_loaded_strictly'],'Full native model and source evaluator execution required')
    require(guard['exit_code']==0 and not guard['resource_guard_aborted'] and guard['full_native_model_replay_passed'],'Full model guard failed')
    artifact=ROOT/publication['model_artifact_directory']
    for name,expected in model['replay_outputs'].items():require(digest(artifact/name)==expected,'Full model replay output changed')
    job=next(j for j in queue['jobs'] if j['id']==model['job_id'])
    original=ROOT/job['output_directory'];scores=np.load(original/'native_scores.npz',allow_pickle=False)
    labels=scores['labels'].astype(int);test=scores['test'].astype(float);validation=scores['validation'].astype(float)
    require(len(test)==73729 and len(validation)==11664 and len(scores['train'])==46653 and np.isfinite(test).all(),'All original complete native points required')
    lookup={(r['protocol'],r['metric']):float(r['mean']) for r in measured}
    checked=0
    def compare(protocol,metric,value):
        nonlocal checked
        require(np.isclose(lookup[protocol,metric],value,rtol=1e-12,atol=1e-12),'Independent metric mismatch: '+protocol+'/'+metric)
        checked+=1
    for ratio,percentile in [(.1,'99.9'),(1,'99'),(5,'95')]:
        prediction=test>np.percentile(validation,100-ratio,method='linear')
        tp=int(np.sum(prediction & (labels==1)));fp=int(np.sum(prediction & (labels==0)));fn=int(np.sum(~prediction & (labels==1)))
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        values=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,auroc=roc_auc_score(labels,test),average_precision=average_precision_score(labels,test))
        for metric,value in values.items():compare('fresh_train_validation'+percentile+'_no_PA',metric,value)
    energy=np.load(original/'workspace/test_results/MSL/results_0/a_test_energy.npy',allow_pickle=False).reshape(-1)
    authorlabels=np.load(original/'workspace/test_results/MSL/results_0/a_gt.npy',allow_pickle=False).reshape(-1).astype(int)
    require(len(energy)==len(authorlabels)==73728 and np.array_equal(authorlabels,labels[:-1]),'Full author nonoverlap geometry required')
    p,r,thresholds=precision_recall_curve(authorlabels,energy);f1=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r!=0);best=int(np.argmax(f1))
    for metric,value in dict(precision=p[best],recall=r[best],f1=f1[best],accuracy=np.mean((energy>=thresholds[best])==authorlabels)).items():compare('fresh_train_test_label_oracle',metric,value)
    compare('fresh_train_author_evaluation','auroc',roc_auc_score(authorlabels,energy));compare('fresh_train_author_evaluation','average_precision',average_precision_score(authorlabels,energy))
    for stem,fields in [('_r_auc',['R_AUC_ROC','R_AUC_PR']),('_vus',['VUS_ROC','VUS_PR']),('_affiliation',['Affiliation_ACC','Affiliation_P','Affiliation_R','Affiliation_F1'])]:
        with (original/'workspace/test_results/MSL/results_0'/ (stem+'.csv')).open(encoding='utf-8-sig',newline='') as stream:source=list(csv.DictReader(stream))
        require(len(source)==1,'Single native source row required')
        for field in fields:
            require(float(source[0][field])==model['original_author_metrics'][stem][0][field],'Source evaluator replay differs')
            compare('fresh_train_author_evaluation',field,float(source[0][field]))
    require(checked==29 and sum(bool(r['mean']) for r in allrows)==2088,'29 verified appended results and2088 total measured metrics required')
    require(receipt['changed_row_positions']==[] and receipt['base_publication_utc']==read(base/'publication_validation.json')['published_utc'],'Append-only publication receipt required')
    output=dict(passed=True,publication=publication['output_directory'],original_rows_byte_exact=7283,registered_original_native_jobs=139,native_cohorts=29,native_slots=841,appended_slots=815,independently_checked_measured_values=29,aggregate_metric_rows=8098,measured_metric_rows=2088,missing_metric_rows=6010,all_original_experiments_complete=False,author_equivalence_certified=False,strict_TAB_result=False)
    path=ROOT/config['output'];path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);review(parser.parse_args().config)
