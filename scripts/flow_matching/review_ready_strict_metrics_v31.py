"""Independently review frozen sources, raw point increments and all statistics."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.flow_matching.publish_full_range_increment_v25 import ROOT, SCOPE, PROTOCOL, stream_sha

POINT = ('precision','recall','f1','auroc','average_precision')
RANGE = ('VUS_ROC','VUS_PR','affiliation_f')
KEY_FIELDS = ('task','algorithm','dataset','protocol','metric','epochs','pattern','missing_ratio','score_recipe')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(row):return tuple(str(row.get(f) or '') for f in KEY_FIELDS)


def stats(samples):
    from scipy.stats import t
    if not samples or not all(math.isfinite(x) for x in samples):
        raise ValueError('Finite defined original metric samples required')
    n=len(samples);mean=statistics.mean(samples)
    sd=statistics.stdev(samples) if n>1 else None
    se=sd/math.sqrt(n) if n>1 else None
    half=float(t.ppf(.975,n-1))*se if n>1 else None
    return dict(n=n,mean=mean,std=sd,se=se,ci95_low=mean-half if n>1 else None,
        ci95_high=mean+half if n>1 else None)


def same(actual,expected):
    if actual in ('',None) or expected is None:
        if actual not in ('',None) or expected is not None:
            raise ValueError('Missing or single-seed uncertainty became a number')
    elif not math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Independent native seed statistic differs')


def review_delta(before,after,samples):
    if [identity(r) for r in before]!=[identity(r) for r in after]:
        raise ValueError('Every original metric identity and order required')
    changed=[]
    permitted={'n','mean','std','se','ci95_low','ci95_high','captured_utc','source',
        'author_equivalence_certified','strict_TAB_result'}
    for index,(old,new) in enumerate(zip(before,after)):
        selected=samples.get((new['algorithm'],new['dataset'],new['metric'])) if new['protocol']==PROTOCOL else None
        expected=stats(selected) if selected else None
        should_change=expected is not None and expected['n']>int(old['n'])
        if not should_change:
            if old!=new:raise ValueError('Unrelated or same-count historical row changed')
            continue
        if (new['task']!='anomaly_detection' or new['metric'] not in POINT+RANGE
                or not int(old['n'])<int(new['n'])<=int(old['required_repeats'])
                or str(new['author_equivalence_certified']).lower()!='false'
                or str(new['strict_TAB_result']).lower()!='false'
                or any(old.get(f)!=new.get(f) for f in set(old)|set(new) if f not in permitted)):
            raise ValueError('Frozen protocol, scope, model budget or uncertainty identity changed')
        for field,value in expected.items():same(new[field],value)
        changed.append(index)
    return changed


def independent_raw_points(result,job,dataset):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,average_precision_score
    source=ROOT/job['output_directory']
    calibration=result['metrics']['strict']['1']['calibration']
    truths,tests,validations=[],[],[]
    with np.load(source/'scores.npz',allow_pickle=False) as scores, np.load(ROOT/dataset['input_npz'],allow_pickle=False) as prepared:
        for entity in dataset['entities']:
            prefix=entity['array_prefix'];labels=scores[prefix+'_labels'];test=scores[prefix+'_test']
            if (not np.array_equal(labels,prepared[prefix+'_labels']) or labels.shape!=test.shape
                    or len(test)!=entity['test_points'] or not np.isfinite(test).all()):
                raise ValueError('Independent complete original saved-score arrays required')
            truths.append(labels);tests.append(test);validations.append(scores[prefix+'_validation'].astype(np.float64))
    validation=np.concatenate(validations)
    if (len(validation)!=calibration['calibration_points'] or calibration['calibration_split']!='validation'
            or hashlib.sha256(validation.astype('<f8').tobytes()).hexdigest()!=calibration['calibration_sha256']):
        raise ValueError('Exactly the original validation array required')
    threshold=float(np.percentile(validation,99,method='linear'))
    same(calibration['threshold'],threshold)
    truth=np.concatenate(truths).astype(bool);test=np.concatenate(tests)
    p,r,f,_=precision_recall_fscore_support(truth,test>threshold,average='binary',zero_division=0)
    metrics=dict(precision=float(p),recall=float(r),f1=float(f),
        auroc=float(roc_auc_score(truth,test)) if np.unique(truth).size==2 else None,
        average_precision=float(average_precision_score(truth,test)) if truth.any() else None)
    for name,value in metrics.items():same(result['metrics']['strict']['1']['micro'][name],value)
    return dict(id=job['id'],full_test_points=len(truth),validation_points=len(validation),metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(ROOT/args.config)
    for item in config['source_receipts']:
        if stream_sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Independent frozen review input changed')
    publication=read(ROOT/config['publication_config'])
    if publication['original_scope']!=SCOPE:raise ValueError('All original experimental obligations required')
    for item in publication['source_receipts']:
        if stream_sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen original source changed')
    base=ROOT/publication['base_publication'];target=ROOT/publication['output_directory']
    receipts=[]
    for folder in (base,target):
        receipt=read(folder/'publication_validation.json')
        for filename,digest in receipt['outputs'].items():
            if stream_sha(folder/filename)!=digest:raise ValueError('Immutable complete export changed')
        receipts.append(receipt)
    def load(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);return reader.fieldnames,list(reader)
    headers,before=load(base/'all_algorithm_dataset_metrics.csv')
    new_headers,after=load(target/'all_algorithm_dataset_metrics.csv')
    if headers!=new_headers or len(headers)!=31 or len(after)!=7283:raise ValueError('Full existing CSV schema/scope required')
    proof=read(ROOT/publication['proof_directory']/'independent_complete_saved_score_audit.json')
    expected_ids=[r['id'] for c in publication['cohorts'] for r in c['runs']]
    actual_ids=[r['id'] for r in proof['records']]
    if actual_ids!=expected_ids or len(set(actual_ids))!=publication['full_saved_score_runs']:
        raise ValueError('Every original unique complete registered audit required')
    hashes={}
    for item in proof['raw_prepared_scientific_source_receipts']:
        digest=stream_sha(ROOT/item['path']);hashes[item['path']]=digest
        if digest!=item['sha256']:raise ValueError('Independent full raw/prepared data checksum differs')
    samples=defaultdict(list);raw_point_reviews=[];range_count=0
    positions={(r['algorithm'],r['dataset'],r['metric']):r for r in before if r['protocol']==PROTOCOL}
    for cohort in publication['cohorts']:
        seeds=set()
        for entry in cohort['runs']:
            if entry['seed'] in seeds:raise ValueError('A retry or fixed weight cannot add a seed')
            seeds.add(entry['seed']);result=read(ROOT/entry['result']);job=result['job']
            for filename,field in (('scores.npz','scores_sha256'),('model.pt','checkpoint_sha256')):
                if stream_sha(ROOT/job['output_directory']/filename)!=result[field]:raise ValueError('Complete original artifact changed')
            micro=result['metrics']['strict']['1']['micro']
            for name in POINT:
                if micro[name] is not None:samples[cohort['algorithm'],cohort['dataset'],name].append(micro[name])
            # Recompute every newly published point cohort from its full original arrays,
            # independently of the producer's manual confusion-count implementation.
            if len(cohort['runs'])>int(positions[cohort['algorithm'],cohort['dataset'],'f1']['n']):
                dataset=next(d for d in read(ROOT/job['dataset_manifest_path'])['datasets'] if d['id']==job['dataset'])
                raw_point_reviews.append(independent_raw_points(result,job,dataset))
            if entry['evaluation'] is not None:
                evaluation=read(ROOT/entry['evaluation']);range_count+=1
                if (evaluation['status']!='completed_range_evaluation' or evaluation['id']!=entry['id']
                        or evaluation['seed']!=entry['seed'] or evaluation['model_result_sha256']!=entry['result_sha256']
                        or evaluation['scores_sha256']!=result['scores_sha256']):
                    raise ValueError('Full native range result identity changed')
                for name in RANGE:
                    aggregate=(evaluation['VUS_entity_macro'] if name.startswith('VUS') else
                               evaluation['strict_affiliation_entity_macro']['1'])[name]
                    entities=(evaluation['VUS_per_entity'] if name.startswith('VUS') else
                        {k:v['1'] for k,v in evaluation['strict_affiliation_per_entity'].items()})
                    values=[r[name] for r in entities.values() if r[name] is not None]
                    if aggregate['defined_entities']!=len(values) or aggregate['undefined_entities']!=len(entities)-len(values):
                        raise ValueError('Undefined/no-event entity counts changed')
                    mean=statistics.mean(values) if values else None
                    same(aggregate['mean_over_defined_entities'],mean)
                    if mean is not None:samples[cohort['algorithm'],cohort['dataset'],name].append(mean)
    changed=review_delta(before,after,samples)
    if changed!=proof['changed_positions'] or len(changed)!=receipts[1]['changed_numeric_metric_rows']:
        raise ValueError('Every published update must have independent source evidence')
    for filename in ('original_generation_jobs.csv','original_energy_jobs.csv','corrected_two_seed_metric_evidence.json'):
        if (base/filename).read_bytes()!=(target/filename).read_bytes():raise ValueError('Other experiment track export changed')
    # Exact field preservation: derive each expected cell from the actual CSV,
    # accepting the renderer's declared unavailable cell for blank CSV values.
    text=(target/'all_metrics.md').read_text(encoding='utf-8')
    table_rows=[line for line in text.splitlines() if line.startswith('| ')][2:]
    if len(table_rows)!=len(after):raise ValueError('Every scientific metric row must be displayed')
    for row,line in zip(after,table_rows):
        expected=['—' if row[c]=='' else row[c].replace('|','\\|').replace('\n',' ') for c in headers]
        raw=['' if row[c]=='' else row[c].replace('|','\\|').replace('\n',' ') for c in headers]
        # Newly generated None cells become —; original absent strings remain empty.
        import re
        cells=re.split(r'(?<!\\) \| ',line[2:-2])
        if len(cells)!=31 or any(cell not in (wanted,empty) for cell,wanted,empty in zip(cells,expected,raw)):
            raise ValueError('A complete CSV field is missing or changed in the text')
    output=ROOT/config['output_directory']
    if output.exists():raise FileExistsError('Preserve every previous independent review')
    output.mkdir(parents=True)
    review=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,original_scope=SCOPE,
        all_full_saved_score_artifact_files_verified=len(actual_ids),native_full_range_evaluations_verified=range_count,
        newly_published_point_full_array_reviews=raw_point_reviews,independently_derived_seed_statistics=True,
        changed_metric_rows=len(changed),unchanged_metric_rows=len(after)-len(changed),all31_CSV_fields_preserved=True,
        measured_metric_rows=sum(bool(r['mean']) for r in after),registered_metric_rows=len(after),
        all_other_tracks_exact=True,additional_training_seed_slots=0,model_reinference_performed=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False)
    path=output/'independent_publication_review.json'
    path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'validation.json').write_text(json.dumps(dict(passed=True,source_receipts=config['source_receipts'],
        outputs={path.name:stream_sha(path)}),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,full_saved_score_files=len(actual_ids),range_evaluations=range_count,
        new_point_array_reviews=len(raw_point_reviews),changed_metrics=len(changed),measured=review['measured_metric_rows'])))


if __name__=='__main__':main()
