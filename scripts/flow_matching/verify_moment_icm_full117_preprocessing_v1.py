"""Close full117 transformed datasets with stored-byte and independent label audits.

No source resampling/scaling rewrite, model inference, SVM or training occurs.
"""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path

from scripts.flow_matching import prepare_moment_icm_classification_v1 as producer

ROOT,read,sha=producer.ROOT,producer.read,producer.sha


def raw_labels(path,expected_channels=None):
    labels=[];body=False
    with Path(path).open('rb') as stream:
        for line in stream:
            stripped=line.decode('utf-8').strip()
            if not body:
                if stripped.lower()=='@data':body=True
                continue
            if not stripped or stripped.startswith('#'):continue
            # Independently read only class fields. Match the pinned parser's
            # documented lowercasing, including its lack of class-field lstrip.
            source=stripped.lower().replace('?', 'NaN')
            if expected_channels is not None and (expected_channels<=0 or source.count(':')!=expected_channels):
                raise ValueError('Independent original channel-count reconciliation failed')
            labels.append(source[source.rfind(':')+1:])
    if not body or not labels:raise ValueError('Complete raw class-label stream required')
    return labels


def close_case(config,job):
    import numpy as np
    identity=job['dataset'];base=Path(config['artifact_root'])/identity
    path=base/'preprocessing.json';proof=read(path)
    guard_path=ROOT/config['state_root']/'attempts'/identity/'guard_receipt.json';guard=read(guard_path)
    expected=[config['python'],'-B','-X','utf8','-u','-m',producer.MODULE,
        '--config',config['_config_path'],'--dataset',identity]
    if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted']
            or guard['proof_sha256']!=sha(path) or guard['exact_child_handle']['command']!=expected
            or not proof['passed'] or proof['job']!=job or proof['configuration_sha256']!=sha(ROOT/config['_config_path'])
            or proof['environment']!=config['environment'] or proof['strict_train_only_preprocessing']
            or not proof['source_AST_reused'] or proof['independent_preprocessing_value_reconstruction']
            or proof['neural_model_training_or_inference_or_SVM_performed'] or proof['raw_data_overwritten']
            or proof['exact_historical_ICM_preprocessing_certified']):
        raise ValueError('Exact guarded source-control transformation and unchanged evidence required')
    raw={}
    for split in ['TRAIN','TEST']:
        source=job['splits'][split]
        if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Complete original raw file changed')
        raw[split]=raw_labels(ROOT/source['path'],proof['records'][split]['channels'])
        if len(raw[split])!=source['instances']:raise ValueError('Original full class-label support differs')
    mapping={label:i for i,label in enumerate(sorted(set(raw['TRAIN'])))}
    records={}
    for split in ['TRAIN','TEST']:
        record=proof['records'][split];n=job['splits'][split]['instances'];c=record['channels']
        files={'values.npy','input_mask.npy','labels.npy','original_instance_rows.npy'}
        if (set(record['files_sha256'])!=files or record['instances']!=n or record['sequence_length']!=256
                or record['values_shape']!=[n,c,256] or record['source_coordinates_copied_and_checked']!=n*c*256
                or not record['official_instance_order_and_all_channels_retained']):
            raise ValueError('All original transformed coordinates and source instances required')
        for name,digest in record['files_sha256'].items():
            if sha(base/split/name)!=digest:raise ValueError('Stored processed array hash differs')
        values=np.load(base/split/'values.npy',mmap_mode='r',allow_pickle=False)
        masks=np.load(base/split/'input_mask.npy',mmap_mode='r',allow_pickle=False)
        labels=np.load(base/split/'labels.npy',allow_pickle=False)
        rows=np.load(base/split/'original_instance_rows.npy',allow_pickle=False)
        if values.shape!=(n,c,256) or str(values.dtype)!=record['values_dtype'] or masks.shape!=(n,256):
            raise ValueError('Complete native dtype/shape required')
        if labels.dtype!=np.dtype('int64') or rows.dtype!=np.dtype('int64') or not np.array_equal(rows,np.arange(n)):
            raise ValueError('Every original instance row must remain in order')
        try:expected_labels=np.asarray([mapping[label] for label in raw[split]],dtype=np.int64)
        except KeyError as error:raise ValueError('Test class absent from official training classes') from error
        if not np.array_equal(labels,expected_labels):raise ValueError('Independent raw-label reconciliation failed')
        for i in range(n):
            if not np.isfinite(values[i]).all() or not np.isin(masks[i],[0,1]).all():
                raise ValueError('Every stored signal coordinate must be finite and mask binary')
        records[split]=dict(instances=n,channels=c,coordinates=n*c*256,values_dtype=str(values.dtype),
            independent_original_labels_and_rows_match=True,all_values_finite=True,all_masks_binary=True,
            file_sha256=record['files_sha256'])
        del values,masks
    return dict(dataset=identity,records=records,proof_sha256=sha(path),guard_sha256=sha(guard_path),
        original_data_standardizer_fitting_performed=proof['original_data_standardizer_fitting_performed'],
        source_TEST_fit_policy_preserved=True,strict_train_only_preprocessing=False,
        independent_label_and_stored_coordinate_audit=True,independent_resampling_and_scaling_rewrite=False,
        original_ICM_pretraining_or_performance_certified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    config=read(ROOT/args.config);jobs=producer.verify(config);config['_config_path']=args.config
    state=read(ROOT/config['state_root']/'status.json')
    if state['completed_count']!=117 or set(state['completed_datasets'])!={j['dataset'] for j in jobs} or state['failed_preserved']:
        raise ValueError('All117 actual complete preprocessing tasks required before closure')
    output=Path(config['artifact_root'])/'full117_processed_data_closure_v1.json'
    if output.exists():raise FileExistsError('Preserve existing full closure')
    records=[close_case(config,job) for job in jobs]
    report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
        configuration_sha256=sha(ROOT/args.config),verifier_sha256=sha(Path(__file__)),datasets=117,official_splits=234,
        records=records,total_source_instances=sum(r['instances'] for c in records for r in c['records'].values()),
        total_transformed_signal_coordinates=sum(r['coordinates'] for c in records for r in c['records'].values()),
        original_source_control_preprocessing=True,strict_train_only_preprocessing=False,
        independent_resampling_and_scaling_rewrite=False,new_model_fit_inference_SVM=0,all_original_experiments_complete=False)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ['passed','datasets','official_splits','total_source_instances','total_transformed_signal_coordinates','new_model_fit_inference_SVM']}))


if __name__=='__main__':main()
