"""Read-only source-entity coverage audit; no scientific arrays are rewritten."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

ROOT=Path(__file__).resolve().parents[2]
CONFIG='configs/datasets/fm_original_entity_coverage_inventory.2026-10-16_v1.json'


def digest_stream(stream):
    h=hashlib.sha256();size=0
    for block in iter(lambda:stream.read(1024*1024),b''):
        h.update(block);size+=len(block)
    return dict(sha256=h.hexdigest(),bytes=size)


def file_fact(path):
    with Path(path).open('rb') as stream:return dict(path=str(path),**digest_stream(stream))


def npy_header(path):
    with Path(path).open('rb') as stream:
        if stream.read(6)!=b'\x93NUMPY':raise ValueError('Original file must have a NumPy header')
        version=stream.read(2)
        if version not in [b'\x01\x00',b'\x02\x00',b'\x03\x00']:raise ValueError('Unknown NPY version')
        width=2 if version[0]==1 else 4
        length=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
        if length>65536:raise ValueError('Unexpectedly large source header')
        info=ast.literal_eval(stream.read(length).decode('utf-8' if version[0]==3 else 'latin1'))
        offset=stream.tell()
    shape=info['shape'];descriptor=info['descr']
    if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('Full time/channel array required')
    if descriptor not in ['<f4','<f8','>f4','>f8']:raise ValueError('Expected numeric source telemetry dtype')
    if offset+shape[0]*shape[1]*int(descriptor[-1])!=Path(path).stat().st_size:
        raise ValueError('Array payload byte count does not match complete header')
    return dict(shape=list(shape),dtype=descriptor,fortran_order=info['fortran_order'],header_bytes=offset,
        complete_payload_byte_count_consistent=True)


def text_fact(path,header=False):
    fact=file_fact(path)
    with Path(path).open('rb') as stream:
        first=stream.readline();lines=1 if first else 0;last=first[-1:] if first else b''
        for block in iter(lambda:stream.read(1024*1024),b''):
            lines+=block.count(b'\n');last=block[-1:]
        # readline already counted the first record; account for a final
        # unterminated record only if bytes follow the first line.
        if fact['bytes']>len(first) and last!=b'\n':lines+=1
    columns=next(csv.reader([first.decode('utf-8-sig').rstrip('\r\n')])) if first else []
    return dict(**fact,physical_data_lines=lines-int(header),first_record_columns=columns,
        all_numeric_rows_parsed=False)


def audit(config):
    base=ROOT/config['tranad_data_root'];metadata=ROOT/config['nasa_metadata']
    rows=list(csv.DictReader(io.StringIO(metadata.read_text(encoding='utf-8-sig'))))
    counts=Counter(r['spacecraft'] for r in rows)
    if dict(counts)!=config['expected_label_rows']:raise ValueError('All original label rows required')
    duplicates=Counter((r['spacecraft'],r['chan_id']) for r in rows)
    raw_files={};row_records=[];summary=[]
    for index,row in enumerate(rows):
        if row['spacecraft'] not in ['SMAP','MSL']:raise ValueError('Unknown original spacecraft')
        facts={}
        for role in ['train','test']:
            p=base/'SMAP_MSL'/role/(row['chan_id']+'.npy');key=p.relative_to(ROOT).as_posix()
            if key not in raw_files:raw_files[key]=dict(**file_fact(p),**npy_header(p))
            facts[role]=raw_files[key]
        if facts['test']['shape'][0]!=int(row['num_values']):raise ValueError('Metadata and complete test length differ')
        if facts['train']['shape'][1]!=facts['test']['shape'][1]:raise ValueError('Complete channel geometry must agree')
        intervals=ast.literal_eval(row['anomaly_sequences'])
        if not intervals or any(len(v)!=2 or not 0<=v[0]<=v[1]<int(row['num_values']) for v in intervals):
            raise ValueError('Anomaly metadata is out of original test bounds')
        row_records.append(dict(source_row_index=index,metadata=row,raw=facts,
            same_channel_metadata_rows=duplicates[(row['spacecraft'],row['chan_id'])],
            labels_generated=False,endpoint_convention_selected=False))
    mirror_path=ROOT/config['nasa_mirror_archive'];archive_fact=file_fact(mirror_path)
    with zipfile.ZipFile(mirror_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Archive contains duplicate paths')
        mirror_rows=list(csv.DictReader(io.StringIO(archive.read('labeled_anomalies.csv').decode('utf-8-sig'))))
        members={}
        for role in ['train','test']:
            for name in names:
                if '/'+role+'/' not in name or not name.endswith('.npy'):continue
                channel=Path(name).stem;key=role+'/'+channel
                if key in members:raise ValueError('Multiple candidate raw members for same role/channel')
                with archive.open(name) as stream:facts=digest_stream(stream)
                local=base/'SMAP_MSL'/role/(channel+'.npy')
                existing=raw_files.get(local.relative_to(ROOT).as_posix())
                members[key]=dict(member=name,**facts,matching_TranAD_raw_byte_hash=(existing['sha256']==facts['sha256']) if existing else None,
                    labelled_channel=any(r['chan_id']==channel for r in rows))
    manifest=json.loads((ROOT/config['registered_prepared_manifest']).read_text(encoding='utf-8'))
    registered=[]
    for dataset in config['datasets']:
        root=ROOT/config['registered_CSV_root']/dataset
        train={p.name[:-10].removeprefix(dataset+'_'):p for p in root.glob('*_train.csv')}
        test={p.name[:-9].removeprefix(dataset+'_'):p for p in root.glob('*_test.csv')}
        if dataset=='SMD':
            raw={role:{p.stem:p for p in (base/'SMD'/role).glob('*.txt')} for role in ['train','test','labels']}
            if not all(set(v)==set(raw['train']) for v in raw.values()) or len(raw['train'])!=config['expected_raw_SMD_pairs']:
                raise ValueError('All28raw SMD training/test/label files must pair')
            expected=set(raw['train']);complete_test_points=0
            for entity in sorted(expected):
                role_facts={role:text_fact(raw[role][entity]) for role in raw}
                if role_facts['test']['physical_data_lines']!=role_facts['labels']['physical_data_lines']:
                    raise ValueError('Full SMD test and label physical row counts differ')
                if len(role_facts['train']['first_record_columns'])!=len(role_facts['test']['first_record_columns']):
                    raise ValueError('Raw SMD feature count mismatch')
                complete_test_points+=role_facts['test']['physical_data_lines']
                for role,facts in role_facts.items():raw_files[raw[role][entity].relative_to(ROOT).as_posix()]=facts
        else:
            expected={r['chan_id'] for r in rows if r['spacecraft']==dataset}
            if len(expected)!=config['expected_unique_labelled_channels'][dataset]:raise ValueError('All distinct labelled channels required')
            complete_test_points=sum(next(r['raw']['test']['shape'][0] for r in row_records if r['metadata']['spacecraft']==dataset and r['metadata']['chan_id']==entity) for entity in expected)
        for role,mapping in [('train',train),('test',test)]:
            for entity,p in sorted(mapping.items()):registered.append(dict(dataset=dataset,entity=entity,role=role,**text_fact(p,header=True)))
        prepared=next(d for d in manifest['datasets'] if d['id']==dataset.lower())
        if len(prepared['entities'])!=len(test):raise ValueError('Live registered manifest entity coverage differs from inventory')
        registered_points=sum(r['physical_data_lines'] for r in registered if r['dataset']==dataset and r['role']=='test')
        if sum(r['test_points'] for r in prepared['entities'])!=registered_points:raise ValueError('Registered test counts must match actual CSV inventory')
        summary.append(dict(dataset=dataset,raw_distinct_labelled_entities=len(expected),raw_metadata_rows=counts.get(dataset),
            raw_distinct_test_points=complete_test_points,registered_CSV_train_entities=len(train),registered_CSV_test_entities=len(test),
            registered_CSV_test_points=registered_points,missing_registered_test_entities=sorted(expected-set(test)),
            registered_test_entities_without_raw_identity=sorted(set(test)-expected),registered_train_only_entities=sorted(set(train)-set(test)),
            same_dataset_coverage_proven=expected==set(test),numerical_coordinate_or_label_equivalence_checked=False))
    return dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),dataset_coverage=summary,
        original_label_metadata=file_fact(metadata),original_label_rows=row_records,
        duplicate_channel_labels=[dict(spacecraft=k[0],channel=k[1],rows=[r for r in rows if (r['spacecraft'],r['chan_id'])==k])
            for k,count in duplicates.items() if count>1],raw_files=raw_files,registered_CSV_files=registered,
        mirror_archive=archive_fact,mirror_raw_members=members,mirror_metadata_rows_equal_to_TranAD=mirror_rows==rows,
        mirror_metadata_content=dict(rows=len(mirror_rows),spacecraft_counts=dict(Counter(r['spacecraft'] for r in mirror_rows))),
        mirror_metadata_comparison_not_a_canonical_label_resolution=True,
        labels_merged_or_selected=False,raw_files_modified=False,numerical_coordinate_audit_complete=False,
        preprocessing_or_training_or_model_inference_executed=False,all_original_experiments_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG);args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'));folder=ROOT/config['report_root']
    if folder.exists():raise FileExistsError('Preserve every existing coverage inventory')
    proof=audit(config);folder.mkdir(parents=True,exist_ok=False)
    proof['configuration_sha256']=file_fact(ROOT/args.config)['sha256']
    (folder/'actual_original_entity_coverage.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,coverage=proof['dataset_coverage'],duplicate_label_channels=proof['duplicate_channel_labels'],
        raw_files_hashed=len(proof['raw_files']),CSV_files_hashed=len(proof['registered_CSV_files']),
        mirror_raw_members_hashed=len(proof['mirror_raw_members']),numerical_audit_complete=False,all_original_experiments_complete=False)))


if __name__=='__main__':main()
