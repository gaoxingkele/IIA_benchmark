"""Read-only full322 source missingness audit in a private compatible environment.

A successful audit may establish that original transformed data remain nonfinite;
it never certifies a repaired cache or neural result.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as recovery
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard_template
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=recovery.ROOT,recovery.read,recovery.sha,recovery.write
MODULE='scripts.flow_matching.audit_moment_icm_anomaly_missingness_v2'
CONFIG='configs/reproducibility/fm_moment_ICM_all322_source_missingness_audit.2026-10-15_v2.json'
REPORT='all322_source_missingness_audit.json'


def missing_ranges(mask):
    import numpy as np
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    cuts=np.flatnonzero(np.diff(indices)>1)+1
    return [[int(group[0]),int(group[-1])+1] for group in np.split(indices,cuts)]


def registration():
    previous=read(ROOT/recovery.producer.CONFIG);jobs=recovery.producer.verify(previous)
    success,failed,pins=recovery.source_partition(previous,jobs)
    package=read(ROOT/recovery.PACKAGE)
    files=[recovery.producer.CONFIG,recovery.PACKAGE,
        'scripts/flow_matching/recover_moment_icm_anomaly_scipy1101_v2.py',
        'scripts/flow_matching/audit_moment_icm_anomaly_missingness_v2.py',
        'tests/test_fm_moment_icm_anomaly_scipy1101_v2.py',
        'tests/test_fm_moment_icm_anomaly_missingness_v2.py',
        str(Path(package['directory'])/'installed_manifest.json'),
        str(Path(package['directory'])/'package_verified.json'),
        'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py',guard_template.CONFIG]
    files += [p['path'] for p in read(ROOT/guard_template.CONFIG)['source_receipts']]
    pins += [dict(path=p,sha256=sha(ROOT/p)) for p in files]
    return dict(schema_version=2,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        producer_configuration=recovery.producer.CONFIG,original_successful_sources=len(success),
        required_source_audits=len(failed),jobs=failed,environment=dict(previous['environment'],scipy='1.10.1'),
        python=previous['python'],settings=previous['settings'],resource_lock=previous['resource_lock'],source_receipts=pins,
        state_root='experiments/runs/fm_moment_ICM_all322_source_missingness_audit_v2',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_ICM_all322_source_missingness_audit_v2',
        boundary=('Readonlyfull322diagnosedcases/all3source splits using SHA256verified privateSciPy1.10.1 '
            'andcomplete unchangedsourceclass. Count every raw/postcubic/postscaler finite/nonfinite '
            'coordinate andcomplete indexranges. Do not fill/drop/normalizeagain or create modelcaches. '
            'Successful audit !=successfulrepair:residualNaNsremain material obligations. '
            'Also independently close all1627successful source caches/all14643arrays without '
            'promoting322failed sources to cache success. Originalfailure/success/rawpreserved, '
            'same16/21/16/3GiBdatapolicy,no neuralfit/results.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen missingness audit input changed: '+pin['path'])
    previous=read(ROOT/config['producer_configuration']);jobs=recovery.producer.verify(previous)
    success,failed,pins=recovery.source_partition(previous,jobs)
    if (config['jobs']!=failed or config['required_source_audits']!=len(failed) or len(failed)!=322
            or config['original_successful_sources']!=len(success) or len(success)!=1627
            or config['environment']!=dict(previous['environment'],scipy='1.10.1') or config['settings']!=previous['settings']):
        raise ValueError('Exact322failed-only readonlyaudit and originaldata policy required')
    return previous,failed


def worker(config):
    previous,jobs=verify(config);recovery.install_private_path(config)
    import numpy as np
    import pandas as pd
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact isolated compatible environment required')
    cls=recovery.producer.source_anomaly_class(ROOT/previous['source_root'])
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    publish=status_writer(output/'progress.json',write);results=[];closed=[]
    original_jobs=recovery.producer.verify(previous)
    success,_,_=recovery.source_partition(previous,original_jobs)
    for job in original_jobs:
        if job['id'] not in success:continue
        directory=Path(previous['artifact_root'])/job['family']/job['id']
        proof_path=directory/'pretraining_preparation.json';proof=read(proof_path)
        if (not proof['passed'] or proof['job']!=job or sha(proof_path)!=success[job['id']]['sha256']
                or proof['configuration_sha256']!=sha(ROOT/recovery.producer.CONFIG)
                or not proof['original_complete_class_AST_reused'] or not proof['original_header_split_scaler_unmodified']
                or proof['neural_fit_inference_SVM']!=0 or proof['historical_ICM_pipeline_certified']):
            raise ValueError('Exact original successful full source cache proof required')
        row_ids,raw=guard_template.raw_records(job['raw']);train_ids=train_raw=None
        if job['train_raw']:train_ids,train_raw=guard_template.raw_records(job['train_raw'])
        expected,fit_rows=guard_template.expected_splits(job,row_ids,train_ids);splits={}
        for split in recovery.producer.SPLITS:
            source_path=job['train_raw']['path'] if job['train_raw'] and split!='test' else job['raw']['path']
            splits[split]=guard_template.close_split(directory/split,proof['splits'][split],expected[split],fit_rows,job['family'],source_path)
        closed.append(dict(id=job['id'],family=job['family'],raw=raw,NASA_train_raw=train_raw,
            cache_directory=str(directory),source_proof_sha256=sha(proof_path),splits=splits))
        publish(output/'progress.json',dict(closed_original_source_caches=len(closed),required_original_source_caches=1627,audited_missing_sources=0))
    for job in jobs:
        raw,rows=recovery.producer.csv_identity(job['raw'])
        raw_frame=pd.read_csv(ROOT/job['raw']['path'])
        raw_values=raw_frame.iloc[:,0].to_numpy()
        interpolated=raw_frame.interpolate(method='cubic').iloc[:,0].to_numpy()
        raw_bad=~np.isfinite(raw_values);splits={}
        if len(raw_values)!=len(rows):raise ValueError('Complete raw parser/record identity required')
        for split in recovery.producer.SPLITS:
            data=recovery.producer.source_dataset(cls,ROOT/job['raw']['path'],split)
            ids,fit_rows=recovery.producer.original_split_rows(data,job,rows)
            bad=~np.isfinite(data.data)
            if len(ids)!=len(data.data) or int(data.scaler.n_samples_seen_)!=int(np.isfinite(interpolated[:fit_rows]).sum()):
                raise ValueError('All source row identities and finite-only StandardScaler sample count required')
            splits[split]=dict(time_rows=len(data.data),nonfinite_coordinates=int(bad.sum()),
                finite_coordinates=int((~bad).sum()),complete_nonfinite_position_ranges=missing_ranges(bad),
                complete_nonfinite_raw_parsed_position_ranges=missing_ranges(np.isin(rows,ids[bad])),
                nonfinite_raw_record_ids_sha256=hashlib.sha256(ids[bad].tobytes()).hexdigest(),
                source_scaler_finite_samples=int(data.scaler.n_samples_seen_),
                original_training_fit_extent_rows=fit_rows,source_scaler_mean=data.scaler.mean_.tolist(),
                source_scaler_scale=data.scaler.scale_.tolist(),all_output_coordinates_finite=not bool(bad.any()))
            del data;gc.collect()
        if sha(ROOT/job['raw']['path'])!=job['raw']['sha256']:raise ValueError('Original raw changed')
        results.append(dict(id=job['id'],family=job['family'],raw_path=job['raw']['path'],raw_sha256=job['raw']['sha256'],
            original_failure=job['original_failure'],raw_parsed_rows=len(rows),raw_nonfinite_coordinates=int(raw_bad.sum()),
            complete_raw_nonfinite_position_ranges=missing_ranges(raw_bad),raw_first_numeric_record_omitted=raw['source_discards_numeric_first_record'],splits=splits))
        publish(output/'progress.json',dict(closed_original_source_caches=len(closed),audited_sources=len(results),required_sources=322))
    if len(closed)!=1627 or len(results)!=322:raise ValueError('All1627successful caches and322failed sources required')
    write(output/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),original_successful_cache_sources=1627,source_audits=322,
        source_split_audits=966,records=results,private_environment=actual,
        original_successful_caches_independently_closed=len(closed),original_cache_records=closed,
        complete_original_cache_arrays=9*len(closed),original_cache_signal_coordinates=sum(s['signal_coordinates'] for r in closed for s in r['splits'].values()),
        original_cache_training_gate_selected=sum(r['splits']['train']['source_length_gate_selected'] for r in closed),
        original_cache_training_windows=sum(r['splits']['train']['eligible_windows'] for r in closed),
        sources_with_residual_nonfinite=sum(any(r['nonfinite_coordinates'] for r in row['splits'].values()) for row in results),
        residual_nonfinite_coordinates=sum(r['nonfinite_coordinates'] for row in results for r in row['splits'].values()),
        full_post_source_signal_coordinates=sum(r['time_rows'] for row in results for r in row['splits'].values()),
        repaired_model_cache_sources=0,all_source_outputs_finite=all(not r['nonfinite_coordinates'] for row in results for r in row['splits'].values()),
        neural_fit_inference_SVM=0,original_source_method_numerics_unmodified=True,raw_or_source_cache_mutated=False,
        all_original_experiments_complete=False))


def controller(config):
    path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
    replacements={'closing_all1949_anomaly_sources':'auditing_all322_original_nonfinite_outputs',
        'all1949_anomaly_cache_closure_passed':'all322_source_missingness_audit_passed'}
    class StatusNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    node=StatusNames().visit(node);ast.fix_missing_locations(node)
    ns=dict(guard_template.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<readonly-audit-guard>','exec'),ns)
    ns['controller'](config)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen source missingness audit')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,readonly_source_audits=322,model_cache_repair=False)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
