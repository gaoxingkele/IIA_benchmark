"""Read every real NPZ header;check all873 fixed-window input geometries.

This does not load data arrays,fit models,skip jobs or certify all data values.
"""
from __future__ import annotations
import ast,json,struct,zipfile
from pathlib import Path
from scripts.flow_matching import audit_original_full873_label_metrics_v1 as scalar


def array_header(stream):
    if stream.read(6)!=b'\x93NUMPY':raise ValueError('Original NPY magic required')
    version=tuple(stream.read(2))
    if version not in [(1,0),(2,0),(3,0)]:raise ValueError('Known NPY schema required')
    size=2 if version==(1,0) else 4
    raw=stream.read(size)
    if len(raw)!=size:raise ValueError('Complete original NPY header size required')
    length=struct.unpack('<H' if size==2 else '<I',raw)[0]
    if length<1 or length>1024*1024:raise ValueError('Bounded original NPY metadata required')
    raw=stream.read(length)
    if len(raw)!=length:raise ValueError('Truncated NPY metadata')
    record=ast.literal_eval(raw.decode('utf-8' if version==(3,0) else 'latin1'))
    if set(record)!={'descr','fortran_order','shape'}:raise ValueError('Complete original NPY fields required')
    shape=record['shape']
    if not isinstance(shape,tuple) or any(not isinstance(n,int) or n<0 for n in shape):raise ValueError('Explicit nonnegative array shape required')
    return dict(shape=list(shape),dtype=record['descr'],fortran_order=record['fortran_order'])


def inspect_case(case,dataset,headers):
    window=case['window'];incompatible=[]
    for entity in dataset['entities']:
        for role,interval in [('train','fit_interval'),('validation','validation_interval'),('test',None)]:
            key=entity['array_prefix']+'_'+role;header=headers[key];shape=header['shape']
            declared=entity['test_points'] if interval is None else entity[interval][1]-entity[interval][0]
            if shape!=[declared,entity['features']]:raise ValueError('Actual array header must match every original entity interval')
            if shape[0]<window:
                incompatible.append(dict(entity=entity['entity_id'],role=role,points=shape[0],window=window,
                    missing_points_for_one_full_window=window-shape[0]))
    return dict(case=case,incompatible_role_inputs=incompatible,all_entities_pass_fixed_window_length_gate=not incompatible,
        training_rejected_before_fit=any(x['role']=='train' for x in incompatible),
        possible_later_scoring_rejection_after_fit=bool(incompatible) and not any(x['role']=='train' for x in incompatible),
        actual_job_terminal_failure_observed=False)


def audit():
    from collections import Counter
    original=scalar.read(scalar.ROOT/scalar.CONFIG);scalar.verify(original)
    prepared=scalar.read(scalar.ROOT/scalar.read(scalar.ROOT/original['source_execution'])['full_entity_preparation'])
    # Derive the actual prepared manifest from a frozen materialized source job.
    queue,job=scalar.resolve_case(original['cases'][0]);manifest_path=scalar.ROOT/job['dataset_manifest_path']
    manifest=scalar.read(manifest_path);datasets={d['id']:d for d in manifest['datasets']};headers={}
    for name,dataset in datasets.items():
        headers[name]={}
        with zipfile.ZipFile(scalar.ROOT/dataset['input_npz']) as archive:
            for entity in dataset['entities']:
                for role in ['train','validation','test','labels']:
                    key=entity['array_prefix']+'_'+role
                    with archive.open(key+'.npy') as stream:headers[name][key]=array_header(stream)
    records=[inspect_case(case,datasets[case['dataset']],headers[case['dataset']]) for case in original['cases']]
    failed=[row for row in records if not row['all_entities_pass_fixed_window_length_gate']]
    return dict(passed=True,source_configuration=scalar.CONFIG,source_configuration_sha256=scalar.sha(scalar.ROOT/scalar.CONFIG),
        manifest_path=str(manifest_path),manifest_sha256=scalar.sha(manifest_path),full_original_scope=original['original_scope'],
        all_source_cases=len(records),actual_arrays_headers_checked=sum(len(x) for x in headers.values()),
        length_incompatible_cases=len(failed),incompatible_case_counts_by_family=dict(Counter(x['case']['family'] for x in failed)),
        training_rejected_before_fit=sum(x['training_rejected_before_fit'] for x in failed),
        scoring_rejection_after_fit=sum(x['possible_later_scoring_rejection_after_fit'] for x in failed),
        records=records,full_array_value_or_checksum_audit_performed=False,
        independently_predicted_window_gate_rejections_are_observed_terminal_failures=False,
        source_jobs_modified_cancelled_or_skipped=False,padding_window_reduction_or_entity_removal_performed=False,
        full_original_data_proof_and_actual_training_guards_still_required=True,all_original_experiments_complete=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError('Preserve prior geometry observation')
    value=audit();scalar.write(target,value)
    print(json.dumps({key:value[key] for key in ['all_source_cases','actual_arrays_headers_checked','length_incompatible_cases',
        'incompatible_case_counts_by_family','training_rejected_before_fit','scoring_rejection_after_fit']}))
