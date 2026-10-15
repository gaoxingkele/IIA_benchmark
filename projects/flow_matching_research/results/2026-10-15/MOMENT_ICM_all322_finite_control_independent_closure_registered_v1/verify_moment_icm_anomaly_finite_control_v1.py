"""Independent complete322 finite-control cache reconstruction and validity audit.

Rebuild CSV interpolation/scaling without the source data class, using the same
frozen libraries. Compare all values, original NaN bytes, raw identities and
target masks exactly. This verifies a disclosed preprocessing control, not the
unchanged historical author pipeline or a neural performance result.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as producer
from scripts.flow_matching import run_moment_icm_finite_control_cold_metadata_v4 as metadata
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as template
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=producer.ROOT,producer.read,producer.sha,producer.write
MODULE='scripts.flow_matching.verify_moment_icm_anomaly_finite_control_v1'
CONFIG='configs/reproducibility/fm_moment_ICM_all322_finite_control_closure.2026-10-15_v1.json'
REPORT='all322_independent_finite_control_closure.json'
NAMES=dict(values='values_T.npy',source_values='original_source_values_with_nonfinite_T.npy',
    raw_rows='raw_CSV_record_indices.npy',raw_observed='raw_observed_T.npy',
    valid_targets='valid_targets_T.npy',residual_filled='residual_filled_T.npy',mask='input_mask_window.npy')


def independently_reconstruct(raw_path):
    """No source data class or its restoration/mapping helpers are used here."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    frame=pd.read_csv(raw_path)
    if frame.shape[1]!=2:raise ValueError('Original two-column signal/label CSV required')
    observed=np.isfinite(frame.iloc[:,0].to_numpy())
    frame.interpolate(inplace=True,method='cubic')
    signal=frame.iloc[:,0].to_numpy().reshape(-1,1)
    n=len(signal);end=int(.6*n);test=n-int(.3*n)
    scaler=StandardScaler().fit(signal[:end])
    values=scaler.transform(signal).squeeze()
    if values.ndim!=1:raise ValueError('Complete original univariate vector required')
    splits=dict(train=slice(0,end),val=slice(end,test),test=slice(test,-1))
    return values,observed,splits,dict(fit_extent_rows=end,finite_fit_samples=int(scaler.n_samples_seen_),
        mean=float(scaler.mean_[0]),scale=float(scaler.scale_[0]))


def inspect_split(directory,record,source_values,raw_rows,raw_observed,scaler):
    import numpy as np
    directory=Path(directory);n=len(source_values);finite=np.isfinite(source_values)
    restored=np.where(finite,source_values,0.)
    valid=raw_observed & finite
    expected=dict(values=restored,source_values=source_values,raw_rows=raw_rows,
        raw_observed=raw_observed,valid_targets=valid,residual_filled=~finite,mask=np.ones(256))
    if (record['time_rows']!=n or record['source_dunder_len']!=n-255
            or record['source_length_gate_selected']!=(n>=256) or record['source_stride1_window_count']!=max(0,n-255)
            or record['source_fit_extent_rows']!=scaler['fit_extent_rows']
            or record['source_scaler_finite_samples']!=scaler['finite_fit_samples']
            or record['source_scaler_mean']!=scaler['mean'] or record['source_scaler_scale']!=scaler['scale']
            or record['replaced_residual_nonfinite_coordinates']!=int((~finite).sum())
            or record['raw_unobserved_coordinates']!=int((~raw_observed).sum())
            or record['valid_reconstruction_target_coordinates']!=int(valid.sum())
            or not record['finite_source_coordinates_byte_identical'] or record['raw_time_rows_dropped']!=0
            or record['raw_or_source_cache_overwritten'] or not record['target_validity_must_be_applied_to_random_mask_and_loss']
            or record['historical_source_equivalence_certified']):
        raise ValueError('Complete original extent and explicitly restored/valid-target identities required')
    if {p.name for p in directory.glob('*.npy')}!=set(NAMES.values()):raise ValueError('All7original/finite/mask/row arrays required')
    hashes={}
    for key,name in NAMES.items():
        path=directory/name;before=path.stat();pin=record['files'][key]
        if pin['name']!=name or sha(path)!=pin['sha256']:raise ValueError('Frozen complete finite-control array changed')
        saved=np.load(path,allow_pickle=False);target=expected[key]
        if (list(saved.shape)!=pin['shape'] or str(saved.dtype)!=pin['dtype'] or saved.dtype!=target.dtype
                or saved.shape!=target.shape or saved.tobytes()!=target.tobytes()):
            raise ValueError('Independent every-coordinate values,NaN bytes,identities and masks must be exact')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Array changed during closure')
        hashes[key]=pin['sha256']
    if not np.isfinite(restored).all():raise ValueError('Every model input must be finite')
    return dict(time_rows=n,signal_coordinates=n,all7arrays_independently_reconstructed_byte_equal=True,
        all_model_inputs_finite=True,original_source_NaN_bytes_preserved=True,
        finite_source_coordinates_unchanged=True,raw_row_identity_equal=True,
        raw_missing_and_source_invalid_targets_excluded=True,
        replaced_coordinates=int((~finite).sum()),raw_unobserved_coordinates=int((~raw_observed).sum()),
        valid_reconstruction_targets=int(valid.sum()),source_eligible_window_count=max(0,n-255),files_sha256=hashes)


def registration():
    previous=read(ROOT/producer.CONFIG);runtime=read(ROOT/metadata.CONFIG)
    acquisition='configs/acquisition/fm_project_moment_pile.v1.json';raw_registry=read(ROOT/acquisition)
    files=[producer.CONFIG,metadata.CONFIG,
        'scripts/flow_matching/verify_moment_icm_anomaly_finite_control_v1.py',
        'tests/test_fm_moment_icm_anomaly_finite_control_closure_v1.py',acquisition]
    files += [p['path'] for p in previous['source_receipts']]+[p['path'] for p in runtime['source_receipts']]
    return dict(schema_version=1,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        data_citation=dict(source_url=raw_registry['sources'][0]['evidence_url'],snapshot_revision=raw_registry['upstream_revision'],
            acquisition_registry=acquisition,acquisition_registry_sha256=sha(ROOT/acquisition)),
        producer_configuration=producer.CONFIG,actual_producer_guard_configuration=metadata.CONFIG,
        sources=322,cells=966,arrays=6762,environment=previous['environment'],python=previous['python'],
        settings=previous['settings'],resource_lock=previous['resource_lock'],
        source_receipts=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))],
        state_root='experiments/runs/fm_moment_icm_all322_finite_control_closure_v1',
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_all322_finite_control_closure_v1',
        boundary=('Full322cases/all966splits/all6762arrays,exact rawCSV identity and original '
            'pandas header/cubic thentrainingfitStandardScaler independently assembled without '
            'source dataclass/restoration helpers. Same frozen libraries,not independentlowlevel '
            'interpolationimplementation. All originalNaNbytes,finitevalues,trainmeanzero '
            'replacements,raw_observed/validtarget/residual masks/rawrows/padmask exactly checked. '
            'No tolerancerelaxation,source/historical/TABcausal equivalence or neuralresults. '
            'Full actual sourceaudit/1627closure and first322preparation guard required. '
            'Same16/21/16/3GiBpolicy,no inputmutation/neuralfit/retry.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen full322closure input changed: '+pin['path'])
    previous=read(ROOT/config['producer_configuration']);jobs=producer.verify(previous)
    runtime=read(ROOT/config['actual_producer_guard_configuration']);metadata.pre_guard(runtime)
    if (config['sources']!=322 or config['cells']!=966 or config['arrays']!=6762
            or config['settings']!=previous['settings'] or config['environment']!=previous['environment']):
        raise ValueError('Full322/966/6762scope and unchanged data protection required')
    return previous,jobs,runtime


def preparation_ready(config):
    previous,jobs,runtime=verify(config)
    proof=Path(previous['artifact_root'])/producer.REPORT
    guard=ROOT/runtime['state_root']/'guard_receipt.json'
    if not proof.exists() or not guard.exists():return False
    p,g=read(proof),read(guard)
    if (not p['passed'] or p['failed_preserved'] or p['required_sources']!=322 or len(p['records'])!=322
            or p['configuration_sha256']!=sha(ROOT/producer.CONFIG)
            or [r['id'] for r in p['records']]!=[j['id'] for j in jobs]
            or not g['passed'] or g['exit_code']!=0 or g['resource_guard_aborted'] or g['proof_sha256']!=sha(proof)
            or g['exact_child_handle']['command']!=[previous['python'],'-B','-X','utf8','-u','-m',producer.MODULE,'--worker']):
        raise ValueError('Exact first successful full322data worker and realv4guard chain required')
    return True


def worker(config):
    previous,jobs,runtime=verify(config)
    if not preparation_ready(config):raise ValueError('Full322 actual preparation is not complete')
    producer.source.install_private_path(config)
    import numpy as np
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact compatible environment required')
    source_proof=read(Path(previous['artifact_root'])/producer.REPORT)
    registry={r['id']:r for r in source_proof['records']}
    audit_config=read(ROOT/producer.audit.CONFIG)
    audited=read(Path(audit_config['artifact_root'])/producer.audit.REPORT)
    missingness={r['id']:r for r in audited['records']}
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    publish=status_writer(output/'progress.json',write);records=[]
    for job in jobs:
        if job['family']!='OPPORTUNITY':raise ValueError('Complete registered failed-source family scope required')
        directory=Path(previous['artifact_root'])/job['id'];proof_path=directory/'finite_control_preparation.json';proof=read(proof_path)
        if (not proof['passed'] or proof['job']!=job or proof['configuration_sha256']!=sha(ROOT/producer.CONFIG)
                or proof['environment']!=actual or sha(proof_path)!=registry[job['id']]['sha256']
                or proof['historical_source_equivalence_certified'] or proof['neural_fit_inference_SVM']!=0):
            raise ValueError('Exact full original-case transformed-control source proof required')
        rows,raw=template.raw_records(job['raw'])
        values,observed,slices,scaler=independently_reconstruct(ROOT/job['raw']['path'])
        if len(values)!=len(rows):raise ValueError('Every original rawCSVrecord required')
        splits={}
        for split,bounds in slices.items():
            expected=values[bounds];expected_rows=rows[bounds]
            splits[split]=inspect_split(directory/split,proof['splits'][split],expected,expected_rows,observed[bounds],scaler)
            old=missingness[job['id']]['splits'][split]
            if (old['time_rows']!=len(expected) or old['nonfinite_coordinates']!=int((~np.isfinite(expected)).sum())
                    or old['source_scaler_finite_samples']!=scaler['finite_fit_samples']
                    or old['source_scaler_mean']!=[scaler['mean']] or old['source_scaler_scale']!=[scaler['scale']]):
                raise ValueError('Independent reconstruction must retain all prior actual missingness observations')
        if sha(ROOT/job['raw']['path'])!=job['raw']['sha256']:raise ValueError('Original raw data changed')
        records.append(dict(id=job['id'],family=job['family'],raw=job['raw'],source_proof_sha256=sha(proof_path),
            cache_directory=str(directory),splits=splits))
        publish(output/'progress.json',dict(closed_sources=len(records),required=322))
    write(output/REPORT,dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        configuration_sha256=sha(ROOT/CONFIG),sources=322,cells=966,arrays=6762,records=records,environment=actual,
        all_source_steps_independently_reconstructed_with_same_libraries=True,independent_lowlevel_interpolation_implementation=False,
        source_signal_coordinates=sum(s['signal_coordinates'] for r in records for s in r['splits'].values()),
        residual_replaced_coordinates=sum(s['replaced_coordinates'] for r in records for s in r['splits'].values()),
        raw_unobserved_coordinates=sum(s['raw_unobserved_coordinates'] for r in records for s in r['splits'].values()),
        valid_reconstruction_targets=sum(s['valid_reconstruction_targets'] for r in records for s in r['splits'].values()),
        all_model_inputs_finite=True,original1627_caches_and322failures_untouched=True,
        full_neural_training_runner_integration=False,neural_fit_inference_SVM=0,
        historical_source_equivalence_certified=False,strict_temporal_independence_certified=False,all_original_experiments_complete=False))


def controller(config):
    import msvcrt
    import psutil
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lock=(base/'prerequisite_metadata.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lock.close();raise RuntimeError('Existing full322closure owner preserved')
    publish=status_writer(base/'status.json',write)
    try:
        while not preparation_ready(config):
            publish(base/'status.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
                state='waiting_actual_full322_finite_preparation',required_sources=322))
            runtime=read(ROOT/metadata.CONFIG);path=ROOT/runtime['state_root']/'status.json'
            if path.exists() and read(path).get('state')=='failed_preserved_without_retry':
                raise ValueError('First sourcepreparation failed;independent closure not launched')
            time.sleep(10)
        # No resource facade was installed above. The unchanged guard installs once.
        path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
        replacements={'closing_all1949_anomaly_sources':'independently_closing_all322_finite_controls',
            'all1949_anomaly_cache_closure_passed':'all322_independent_finite_control_closure_passed'}
        class Names(ast.NodeTransformer):
            def visit_Constant(self,n):
                if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
                return n
        node=Names().visit(node);ast.fix_missing_locations(node)
        ns=dict(template.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<full322-independent-guard>','exec'),ns)
        ns['controller'](config)
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen full322independent registration')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,sources=322,cells=966,arrays=6762)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
