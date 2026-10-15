"""Index every ICM appendix result and rehash its local raw evaluation data.

No training,inference,preprocessing-value changes,downloads or result claims.
The separate pretrained AD queues do not cover these forecast/SVM experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
DATASETS=['Exchange','ETTh1','ETTh2','ETTm1','ETTm2','Weather']
MODELS4=['Pre-trained MT + Infini-Channel Mixer (8 folds)',
    'Pre-trained MT + Infini-Channel Mixer (4 folds)','Pre-trained Univariate MT']
MODELS5=['MT + Graph Transformer layer','NBeats','CrossFormer','iTransformer',
    'Infini-Channel Mixer MT (Ours)','Infini-Channel Mixer + Static MT (Ours)',
    'Channel Concatenation','Univariate MT']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def classification_rows(text,table):
    pattern=r'(?m)^([A-Za-z][A-Za-z0-9-]*)\n(0\.\d{3}|1\.\d{3})\n(0\.\d{3}|1\.\d{3})$'
    return [dict(table=table,pdf_page=page,dataset=name,metric='accuracy',model=model,reported=float(value))
        for page,content in text for name,a,b in re.findall(pattern,content)
        for model,value in [('MOMENT-Tiny',a),('MOMENT-Tiny + ICM',b)]]


def forecasting_rows(text,models,table):
    lines=[line.strip() for line in text.splitlines() if line.strip()];result=[]
    for model in models:
        if lines.count(model)!=1:raise ValueError('Exactly one original model block required: '+model)
        cursor=lines.index(model)+1
        for horizon in [96,192,384]:
            if lines[cursor]!=str(horizon):raise ValueError('Original horizon ordering changed')
            cursor+=1
            for dataset in DATASETS:
                for metric in ['MAE','MSE']:
                    value=lines[cursor];cursor+=1
                    if not re.fullmatch(r'\d+\.\d{3}',value):raise ValueError('Three-decimal original table cell required')
                    result.append(dict(table=table,pdf_page=9,model=model,dataset=dataset,
                        horizon=horizon,metric=metric,reported=float(value)))
    return result


def targets(paper):
    import fitz
    with fitz.open(paper) as document:
        if len(document)!=12:raise ValueError('Exact original12page version required')
        pages=[p.get_text() for p in document]
    table4=forecasting_rows(pages[8].split('Table 4:')[0],MODELS4,4)
    table5=forecasting_rows(pages[8].split('Table 4:')[1].split('Table 5:')[0],MODELS5,5)
    table6=classification_rows([(9,pages[8].split('D.2\nClassification')[1]),(10,pages[9].split('Table 6:')[0])],6)
    table7=classification_rows([(10,pages[9].split('Table 6:')[1]),(11,pages[10]),(12,pages[11])],7)
    result=table4+table5+table6+table7
    if dict(Counter(r['table'] for r in result))!={4:108,5:288,6:52,7:182}:
        raise ValueError('All630scalar targets required,including91univariate datasets')
    identities=[(r['table'],r['model'],r['dataset'],r.get('horizon'),r['metric']) for r in result]
    if len(set(identities))!=630:raise ValueError('Duplicate or missing original target cell')
    return result


def file_audit(path,expected,kind):
    """One streaming pass: checksum, TS header/dimensions/labels or CSV row count.

    Does not parse series floats,impute,resample,normalize or verify leakage.
    """
    path=Path(path)
    if not path.is_file():return dict(path=str(path),status='missing')
    before=path.stat();digest=hashlib.sha256();blob=hashlib.sha1(f'blob {before.st_size}\0'.encode())
    header={};in_data=False;instances=0;channels=Counter();labels=Counter();csv_header=None
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line);blob.update(line)
            if kind=='csv':
                if csv_header is None:csv_header=next(csv.reader([line.decode('utf-8-sig').strip()]))
                elif line.strip():instances+=1
                continue
            if not in_data:
                text=line.decode('utf-8-sig').strip()
                if not text or text.startswith('#'):continue
                key,_,value=text.partition(' ');header[key.lower()]=value
                if key.lower()=='@data':in_data=True
            elif line.strip() and not line.lstrip().startswith(b'#'):
                instances+=1
                if header.get('@timestamps','').lower()=='false':
                    # One colon between channels, plus the final class label.
                    dimensions=line.count(b':');channels[dimensions]+=1
                    index=line.rfind(b':');labels[line[index+1:].strip().decode('utf-8')]+=1
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Raw file changed during audit')
    valid=(expected['status']=='verified' and digest.hexdigest()==expected['sha256']
        and before.st_size==expected['size_bytes'] and blob.hexdigest()==expected['git_blob_sha1'])
    if kind=='ts':
        declared=header.get('@classlabel','').split()
        valid=valid and in_data and instances>0 and bool(declared) and declared[0].lower()=='true'
        if labels:valid=valid and set(labels).issubset(set(declared[1:])) and min(channels)>0
    else:valid=valid and instances>0 and bool(csv_header)
    return dict(path=str(path),status='verified' if valid else 'audit_failed',bytes=before.st_size,
        sha256=digest.hexdigest(),git_blob_sha1=blob.hexdigest(),observed_instances=instances,
        TS_header=header if kind=='ts' else None,observed_dimensions=dict(channels),observed_class_counts=dict(labels),
        CSV_columns=csv_header,series_float_values_parsed=False,raw_preprocessing_or_split_leakage_certified=False)


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen source changed: '+pin['path'])
    rows=targets(ROOT/config['paper'])
    if rows!=config['all630_reported_targets']:raise ValueError('Frozen original target transcription differs')
    return rows


def execute(config):
    rows=verify(config);original=read(ROOT/config['pile_integrity'])
    references={r['path']:r for r in original['records']};files=[]
    classification=list(dict.fromkeys((r['table'],r['dataset']) for r in rows if r['table'] in [6,7]))
    for table,dataset in classification:
        for split in ['TRAIN','TEST']:
            logical=config['classification_root']+'/'+dataset+'/'+dataset+'_'+split+'.ts'
            if logical not in references:raise ValueError('No previous registered checksum for '+logical)
            row=file_audit(ROOT/logical,references[logical],'ts');row.update(dataset=dataset,split=split,table=table,logical_path=logical)
            files.append(row)
    for dataset,name in config['forecast_files'].items():
        logical=config['forecast_root']+'/'+name
        if logical not in references:raise ValueError('No previous registered checksum for '+logical)
        row=file_audit(ROOT/logical,references[logical],'csv');row.update(dataset=dataset,logical_path=logical);files.append(row)
    if len(files)!=240:raise ValueError('234classification splits plus6forecast files required')
    folder=ROOT/config['output_directory'];folder.mkdir(parents=True,exist_ok=False)
    fields=['table','pdf_page','model','dataset','horizon','metric','reported']
    with (folder/'all630_paper_reported_metrics.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),paper_sha256=sha(ROOT/config['paper']),
        reported_scalar_targets=len(rows),reported_evaluation_cells=432,
        multivariate_classification_datasets=26,univariate_classification_datasets=91,
        forecast_datasets=6,classification_files=234,total_audited_files=len(files),
        file_status_counts=dict(Counter(r['status'] for r in files)),files=files,
        original_pretraining_and_forecasting_and_SVM_experiments_run=False,new_fit_or_inference=0,
        original_paper_raw_data_version_certified=False,raw_files_overwritten=False,
        raw_preprocessing_or_split_leakage_certified=False,all_original_experiments_complete=False)
    (folder/'complete_raw_data_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    report=execute(read(ROOT/args.config))
    print(json.dumps({k:report[k] for k in ['reported_scalar_targets','reported_evaluation_cells','multivariate_classification_datasets',
        'univariate_classification_datasets','total_audited_files','file_status_counts','new_fit_or_inference']}))


if __name__=='__main__':main()
