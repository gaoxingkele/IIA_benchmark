"""Complete ready-cohort audits with original grouped industrial score layouts."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

import psutil

from scripts.flow_matching import publish_ready_strict_metrics_v31 as original

ROOT=original.ROOT
ORDINARY_POINTS=original.audit_points


def industrial_points(scores,prepared,archive,dataset,result):
    import numpy as np
    if not dataset.get('grouped_splits_verified'):
        raise ValueError('Original grouped industrial split audit required')
    strict=result['metrics']['strict']['1'];calibration=strict['calibration']
    if (strict['protocol']!='strict_validation_only_no_pa_v1' or strict['coverage']!='complete'
            or strict['point_adjustment'] is not False or calibration['calibration_split']!='validation'
            or calibration['false_alarm_percent']!=1
            or calibration['rule']!='validation_score_percentile_linear_strict_greater'
            or calibration['calibration_dataset_sha256']!=dataset['input_sha256']
            or calibration['model_artifact_sha256']!=result['checkpoint_sha256']):
        raise ValueError('Original complete validation-only no-PA industrial protocol required')
    roles=[('train',dataset['training_groups']),('validation',dataset['validation_groups']),('test',dataset['entities'])]
    expected=set();validation=[];truths=[];tests=[];groups=[];entities=[]
    if set(strict['per_entity'])!={e['entity_id'] for e in dataset['entities']}:
        raise ValueError('Every original industrial test entity required')
    occupied={}
    for role,segments in roles:
        for segment in segments:
            prefix=segment['array_prefix'];score_name=prefix+'_'+role
            input_name=prefix+('_test' if role=='test' else '_values')
            expected.add(score_name);values=scores[score_name]
            if (values.ndim!=1 or len(values)!=segment['points'] or not np.isfinite(values).all()
                    or len(values)!=original.array_shape(archive,input_name)[0]):
                raise ValueError('Every full original industrial segment timestamp required')
            verified=[]
            for suffix in ('_labels','_timestamps','_source_rows','_fault_labels'):
                name=prefix+suffix
                if name in prepared.files:
                    expected.add(name)
                    if name not in scores.files or scores[name].shape!=prepared[name].shape or not np.array_equal(scores[name],prepared[name]):
                        raise ValueError('Original industrial labels, timestamps, source row order or fault labels changed')
                    if len(scores[name])!=len(values):
                        raise ValueError('Every original industrial provenance coordinate required')
                    verified.append(suffix)
            # Preserve the source-file interval identity, including purged temporal boundaries.
            start,end=segment['source_interval'];source=segment['source_path']
            if end-start!=len(values):raise ValueError('Original full source interval required')
            for old_role,old_start,old_end in occupied.get(source,[]):
                if old_role!=role and max(start,old_start)<min(end,old_end):
                    raise ValueError('Training, validation and test source intervals must not overlap')
            occupied.setdefault(source,[]).append((role,start,end))
            groups.append(dict(role=role,entity_id=segment['entity_id'],points=len(values),
                source_path=source,source_interval=segment['source_interval'],provenance_arrays_verified=verified))
            if role=='validation':validation.append(values.astype(np.float64))
            if role=='test':
                labels=scores[prefix+'_labels']
                if (labels.shape!=values.shape or len(values)!=segment['test_points'] or not np.isin(labels,[0,1]).all()):
                    raise ValueError('All original industrial test labels required')
                metrics=original.independent_points(labels,values,calibration['threshold'])
                for metric,value in metrics.items():original.assert_close(value,strict['per_entity'][segment['entity_id']][metric],metric)
                entities.append(dict(entity_id=segment['entity_id'],test_points=len(values),independent_point_metrics=metrics))
                truths.append(labels);tests.append(values)
    if set(scores.files)!=expected:raise ValueError('Exactly all original industrial score and provenance arrays required')
    validation=np.concatenate(validation)
    if (len(validation)!=calibration['calibration_points']
            or hashlib.sha256(validation.astype('<f8').tobytes()).hexdigest()!=calibration['calibration_sha256']):
        raise ValueError('All original independent industrial validation groups required')
    threshold=float(np.percentile(validation,99,method='linear'))
    original.assert_close(threshold,calibration['threshold'],'industrial validation threshold')
    micro=original.independent_points(np.concatenate(truths),np.concatenate(tests),threshold)
    for metric,value in micro.items():original.assert_close(value,strict['micro'][metric],metric)
    original.assert_close(float(np.mean([e['independent_point_metrics']['f1'] for e in entities])),strict['macro_f1'],'industrial macro F1')
    return dict(validation_points=len(validation),threshold=threshold,full_entities=len(entities),
        full_test_points=sum(e['test_points'] for e in entities),micro=micro,entities=entities,
        original_grouped_train_validation_test_segments=groups,
        every_original_timestamp_source_row_and_fault_label_verified=True,
        original_grouped_source_intervals_disjoint=True,no_point_adjustment=True,no_test_calibration=True)


def full_points(scores,prepared,archive,dataset,result):
    if 'training_groups' in dataset or 'validation_groups' in dataset:
        if 'training_groups' not in dataset or 'validation_groups' not in dataset:
            raise ValueError('Both original industrial training and validation group sets required')
        return industrial_points(scores,prepared,archive,dataset,result)
    return ORDINARY_POINTS(scores,prepared,archive,dataset,result)


def worker(config,path):
    # The frozen v31 worker retains all original run/data/hash/statistic checks;
    # only the validator dispatch recognizes the original industrial producer layout.
    original.audit_points=full_points
    original.worker(config,path)
    target=ROOT/config['output_directory']
    with (target/'all_algorithm_dataset_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);headers,rows=reader.fieldnames,list(reader)
    text=(target/'all_metrics.md').read_text(encoding='utf-8')
    labels=[line for line in text.splitlines() if line.startswith('| ')][0][2:-2].split(' | ')
    if len(labels)!=31 or len(headers)!=31:raise ValueError('Every original scientific CSV field required')
    intro=text.split('\n| ',1)[0].replace('指标 v31','指标 v32')
    (target/'all_metrics.md').write_text(intro+'\n\n'+original.legacy.table(labels,
        [[r.get(h) or '—' for h in headers] for r in rows])+'\n',encoding='utf-8')
    readme=(target/'README.rst').read_text(encoding='utf-8').replace('指标 v31','指标 v32')
    readme+='\n工业片段保持原训练/验证/测试分组；每一个原时间戳、行号、故障标签和来源区间均独立核验。v31首次核验失败记录与268份进度保留，未缩减351份重核验范围。\n'
    (target/'README.rst').write_text(readme,encoding='utf-8')
    receipt=original.read(target/'publication_validation.json')
    receipt['industrial_original_grouped_score_layout_and_provenance_verified']=True
    receipt['previous_failed_audit_preserved']=config['previous_failed_audit']
    receipt['outputs']={p.name:original.stream_sha(p) for p in target.iterdir() if p.is_file() and p.name!='publication_validation.json'}
    (target/'publication_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    config=original.read(ROOT/args.config);original.verify_registration(config)
    if args.worker:return worker(config,args.config)
    api=original.resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full ready-cohort audit already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None,
        configuration_sha256=original.stream_sha(ROOT/args.config))
    def update(values):state.update(values);original.legacy.write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1',
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen([config['python'],'-X','utf8','-u','-m','scripts.flow_matching.publish_ready_strict_metrics_v32',
                    '--config',args.config,'--worker'],cwd=ROOT,env=env,stdout=out,stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid)
                update(dict(state='auditing_all_full_saved_scores',active_pid=child.pid,active_create_time=handle.create_time(),
                    actual_command=handle.cmdline()))
                def guard_update(values):
                    values=dict(values)
                    if values.get('state')=='replaying_full_checkpoints':values['state']='auditing_all_full_saved_scores'
                    update(values)
                code,usage=original.capped_wait(child,config,api,guard_update)
            receipt=dict(exit_code=code,full_saved_score_audit_passed=code==0,**usage)
            original.exclusive_write(base/'receipt.json',receipt)
            update(dict(state='completed' if code==0 else 'failed_preserved',active_pid=None,**receipt))
            if code:raise RuntimeError('Full ready-cohort audit failed; complete attempt preserved')
    finally:owner.close()


if __name__=='__main__':main()
