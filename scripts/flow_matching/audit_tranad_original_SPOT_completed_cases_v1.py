"""Independent standard-library array/count audit of complete source-SPOT cases.

Not a new calibration, inference or training run. Source PA/binary ROC semantics
are preserved; undefined full-population results remain undefined.
"""
from __future__ import annotations
import argparse
import array
import ast
from datetime import datetime,timezone
import hashlib
import math
from pathlib import Path
import struct
import sys
import zipfile
from scripts.flow_matching import evaluate_tranad_original_SPOT_full30_v1 as source


def vector(archive,name,digest=None):
    with archive.open(name+'.npy') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('NPY vector required')
        version=stream.read(2)
        if version==b'\x01\x00':n=struct.unpack('<H',stream.read(2))[0]
        elif version in [b'\x02\x00',b'\x03\x00']:n=struct.unpack('<I',stream.read(4))[0]
        else:raise ValueError('Unknown NPY version')
        if n>65536:raise ValueError('Oversized NPY header')
        header=ast.literal_eval(stream.read(n).decode('utf-8' if version==b'\x03\x00' else 'latin1'))
        shape=header['shape'];descr=header['descr']
        if header['fortran_order'] or len(shape)!=1 or shape[0]<=0:raise ValueError('Complete one-dimensional vector required')
        code,width={'<f8':('d',8),'<f4':('f',4),'|u1':('B',1),'|b1':('B',1)}[descr]
        payload=stream.read(shape[0]*width+1)
        if len(payload)!=shape[0]*width or stream.read(1):raise ValueError('Truncated or trailing vector data')
    actual=dict(dtype=descr,shape=list(shape),content_sha256=hashlib.sha256(payload).hexdigest())
    if digest is not None and actual!=digest:raise ValueError('Exact vector shape/dtype/payload lineage required')
    values=array.array(code);values.frombytes(payload)
    if sys.byteorder!='little' and width>1:values.byteswap()
    if not all(map(math.isfinite,values)):raise ValueError('Every original coordinate must be finite')
    if width==1 and any(v not in [0,1] for v in values):raise ValueError('Complete binary labels/predictions required')
    return values


def source_pa(raw,labels):
    if len(raw)!=len(labels):raise ValueError('Full aligned arrays required')
    result=list(map(bool,raw));start=0
    while start<len(labels):
        if not labels[start]:start+=1;continue
        stop=start+1
        while stop<len(labels) and labels[stop]:stop+=1
        if any(raw[start:stop]):
            # Source backward range excludes index0; preserve its original bit.
            result[max(start,1):stop]=[True]*(stop-max(start,1))
        start=stop
    return result


def counts(pred,labels):
    if len(pred)!=len(labels):raise ValueError('Full aligned arrays required')
    tp=sum(bool(a and b) for a,b in zip(pred,labels));fp=sum(bool(a and not b) for a,b in zip(pred,labels))
    fn=sum(bool(not a and b) for a,b in zip(pred,labels));tn=len(labels)-tp-fp-fn
    p=tp/(tp+fp+1e-5);r=tp/(tp+fn+1e-5)
    return dict(f1=2*p*r/(p+r+1e-5),precision=p,recall=r,TP=tp,TN=tn,FP=fp,FN=fn,
        **{'ROC/AUC':.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None})


def compare(expected,actual):
    if set(expected)!=set(actual):raise ValueError('Every source point metric required')
    for key,value in expected.items():
        if actual[key] is None:
            if value not in [None,0]:raise ValueError('Undefined single-class source binary ROC behavior changed')
        elif value is None or not math.isclose(value,actual[key],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Independent full count/epsilon metric differs: '+key)


def audit_case(c,record):
    case=record['entry']['case'];base=source.ROOT/c['state_root'];out=Path(c['artifact_root'])/case['id']
    report_path=out/source.REPORT;guard_path=base/'attempts'/case['id']/'guard_receipt.json'
    report,guard=source.read(report_path),source.read(guard_path);bound=source.ready(record)
    if (bound is None or not report['passed'] or report['case']!=case or report['source_lineage']!=bound['lineage']
            or report['configuration_sha256']!=source.sha(source.ROOT/source.CONFIG)
            or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['proof_sha256']!=source.sha(report_path)
            or source.sha(report['prediction_file'])!=report['prediction_sha256']):
        raise ValueError('Complete exact source/native/all-label/SPOT/prediction/guard chain required')
    dataset=bound['dataset'];entities=dataset['entities'];expected_keys=set();rows=[];undefined=[]
    if len(report['entities'])!=len(entities):raise ValueError('Full original entity denominator required')
    with zipfile.ZipFile(bound['lineage']['native_scores']) as scores,zipfile.ZipFile(report['prediction_file']) as predictions:
        for entity,result in zip(entities,report['entities']):
            prefix=entity['array_prefix']
            if result['entity']!=entity['entity_id'] or result['array_prefix']!=prefix:raise ValueError('Original entity order required')
            train=vector(scores,prefix+'_train',result['train_scores']);test=vector(scores,prefix+'_test',result['test_scores'])
            if len(test)!=entity['test_points']:raise ValueError('All original test points required')
            if result['threshold'] is None:
                if (result['status']!='undefined_constant_training_scores_no_source_peaks' or len(set(train))!=1
                        or result['controls'] or not result['calibration_undefined_not_percentile_imputed']):
                    raise ValueError('Only actual constant calibration populations are explicitly undefined')
                undefined.append(dict(entity=result['entity'],test_points=len(test)));continue
            if set(result['controls'])!=set(case['required_label_policies']):raise ValueError('All original label views required')
            for policy,info in dataset['labels_only_controls'].items():
                if source.sha(info['path'])!=info['sha256']:raise ValueError('Frozen labels changed')
                with zipfile.ZipFile(info['path']) as label_file:labels=vector(label_file,prefix+'_labels',result['label_digests'][policy])
                raw=[v>result['threshold'] for v in test];pa=source_pa(raw,labels)
                for suffix,values,key in [('raw',raw,'raw_source_point_metrics'),('source_PA',pa,'source_PA_point_metrics')]:
                    name=prefix+'__'+policy+'__'+suffix;expected_keys.add(name+'.npy')
                    saved=vector(predictions,name)
                    if list(saved)!=list(values):raise ValueError('All prediction bits must match independent threshold/PA')
                    independent=counts(values,labels);compare(result['controls'][policy][key]['values'],independent)
                    rows.append(dict(entity=result['entity'],label_policy=policy,mode=suffix,test_points=len(labels),**independent))
        if set(predictions.namelist())!=expected_keys:raise ValueError('Missing or surplus per-entity predictions')
    summaries=[]
    for policy in case['required_label_policies']:
        for mode in ['raw','source_PA']:
            selected=[r for r in rows if r['label_policy']==policy and r['mode']==mode]
            summaries.append(dict(label_policy=policy,mode=mode,required_entities=len(entities),defined_entities=len(selected),
                undefined_entities=len(undefined),full_entity_macro_f1=math.fsum(r['f1'] for r in selected)/len(entities) if not undefined else None,
                explicitly_defined_subset_macro_f1=math.fsum(r['f1'] for r in selected)/len(selected) if selected else None,
                full_macro_status='defined' if not undefined else 'undefined_constant_calibration_entity',
                independent_training_repetitions=0,training_seed_interval=None))
    return dict(case=case,source_report_sha256=source.sha(report_path),source_guard_sha256=source.sha(guard_path),
        full_entities=len(entities),all_test_points=sum(e['test_points'] for e in entities),undefined_entities=undefined,
        independent_point_metric_cells=len(rows)*8,all_prediction_bits_independently_recomputed=True,
        source_epsilon_and_PA_index0_preserved=True,binary_prediction_ROC_is_not_continuous_score_AUROC=True,
        entity_metric_rows=rows,full_denominator_macro_controls=summaries,all_original_experiments_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    c=source.read(source.ROOT/source.CONFIG);source.verify(c)
    state=source.read(source.ROOT/c['state_root']/'status.json');completed=state['completed_ids']
    results=[audit_case(c,r) for r in c['entries'] if r['entry']['case']['id'] in completed]
    if len(results)!=len(completed):raise ValueError('Every captured complete source SPOT case required')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    source.exclusive(output/'independent_full_array_audit.json',dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        original_scope=c['original_scope'],required_unique_cases=25,required_comparison_slots=30,complete_cases=len(results),
        cases=results,source_model_training_or_inference_performed=False,all_original_experiments_complete=False))
    print(__import__('json').dumps(dict(cases=len(results),point_metric_cells=sum(r['independent_point_metric_cells'] for r in results),
        undefined_entities=sum(len(r['undefined_entities']) for r in results),output=str(output))))


if __name__=='__main__':main()
