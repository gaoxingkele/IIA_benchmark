"""Explicit finite-input control for322 source cases; never historical equivalence.

Preserve the entire source output (including NaNs), replace residual nonfinite
standardized inputs with training-mean zero, and exclude all raw-missing targets
from reconstruction loss. Do not alter any originally finite source coordinate.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from scripts.flow_matching import audit_moment_icm_anomaly_missingness_v2 as audit
from scripts.flow_matching import recover_moment_icm_anomaly_scipy1101_v2 as source
from scripts.flow_matching import verify_moment_icm_PILE_anomaly_v1 as guard_template
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=source.ROOT,source.read,source.sha,source.write
MODULE='scripts.flow_matching.prepare_moment_icm_anomaly_finite_control_v3'
CONFIG='configs/reproducibility/fm_moment_ICM_all322_anomaly_finite_control.2026-10-15_v3.json'
REPORT='all322_finite_control_preparation.json'
ARRAYS=dict(values='values_T.npy',source_values='original_source_values_with_nonfinite_T.npy',
    raw_rows='raw_CSV_record_indices.npy',raw_observed='raw_observed_T.npy',
    valid_targets='valid_targets_T.npy',residual_filled='residual_filled_T.npy',mask='input_mask_window.npy')


def restored_vectors(values,raw_observed,scaler_mean,scaler_scale):
    import numpy as np
    values=np.asarray(values);raw_observed=np.asarray(raw_observed)
    if values.ndim!=1 or values.dtype!=np.dtype('float64') or raw_observed.shape!=values.shape or raw_observed.dtype!=np.dtype(bool):
        raise ValueError('Complete native float64 source vector and explicit bool raw observation identity required')
    if not np.isfinite(scaler_mean) or not np.isfinite(scaler_scale) or scaler_scale<=0:
        raise ValueError('Finite actual source training-fit scaler required;no all-missing fallback invented')
    finite=np.isfinite(values);restored=values.copy();restored[~finite]=0.
    targets=raw_observed & finite
    if restored[finite].tobytes()!=values[finite].tobytes() or not np.isfinite(restored).all():
        raise ValueError('Every finite source coordinate must stay byte identical')
    return restored,targets,~finite


def export_split(dataset,job,rows,raw_observed,output):
    import numpy as np
    if job['family']!='OPPORTUNITY':raise ValueError('This control is registered for the322 diagnosed OPPORTUNITY cases only')
    ids,fit_rows=source.producer.original_split_rows(dataset,job,rows)
    positions=np.searchsorted(rows,ids)
    if not np.array_equal(rows[positions],ids):raise ValueError('Complete raw row identity required')
    observed=raw_observed[positions]
    mean=float(dataset.scaler.mean_[0]);scale=float(dataset.scaler.scale_[0])
    restored,targets,filled=restored_vectors(dataset.data,observed,mean,scale)
    if len(restored)!=dataset.length_timeseries:raise ValueError('No original source time row may be dropped')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arrays=dict(values=restored,source_values=dataset.data,raw_rows=ids,
        raw_observed=observed,valid_targets=targets,residual_filled=filled,mask=np.ones(256))
    files={}
    for key,name in ARRAYS.items():
        array=arrays[key];path=output/name;np.save(path,array,allow_pickle=False)
        saved=np.load(path,allow_pickle=False)
        if saved.dtype!=array.dtype or saved.shape!=array.shape or saved.tobytes()!=array.tobytes():
            raise ValueError('Full native arrays, including original NaN bytes, must be preserved')
        files[key]=dict(name=name,sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype))
    return dict(time_rows=len(restored),source_dunder_len=dataset.__len__(),source_length_gate_selected=len(restored)>=256,
        source_stride1_window_count=max(0,dataset.__len__()),source_fit_extent_rows=fit_rows,
        source_scaler_finite_samples=int(dataset.scaler.n_samples_seen_),source_scaler_mean=mean,source_scaler_scale=scale,
        replaced_residual_nonfinite_coordinates=int(filled.sum()),raw_unobserved_coordinates=int((~observed).sum()),
        valid_reconstruction_target_coordinates=int(targets.sum()),finite_source_coordinates_byte_identical=True,
        raw_time_rows_dropped=0,raw_or_source_cache_overwritten=False,files=files,
        source_full_CSV_interpolation_before_split_retained=True,strict_temporal_independence_certified=False,
        target_validity_must_be_applied_to_random_mask_and_loss=True,historical_source_equivalence_certified=False)


def reconstruction_loss_mask(random_keep_mask,valid_targets,input_mask):
    """Original MOMENT mask1=keep,mask0=predict; supervise only observed valid targets."""
    import numpy as np
    keep=np.asarray(random_keep_mask);valid=np.asarray(valid_targets);padding=np.asarray(input_mask)
    if valid.dtype!=np.dtype(bool) or not np.isin(keep,[0,1]).all() or not np.isin(padding,[0,1]).all():
        raise ValueError('Explicit bool valid targets and binary original mask convention required')
    if valid.ndim==3:
        if keep.ndim==2:keep=keep[:,None,:]
        if padding.ndim==2:padding=padding[:,None,:]
    try:return np.broadcast_to((keep==0)&(padding==1),valid.shape)&valid
    except ValueError as error:raise ValueError('Random/padding masks must preserve batch/channel/time identity') from error


def read_window(directory,index,sequence_length=256):
    import numpy as np
    directory=Path(directory)
    values=np.load(directory/ARRAYS['values'],mmap_mode='r',allow_pickle=False)
    valid=np.load(directory/ARRAYS['valid_targets'],mmap_mode='r',allow_pickle=False)
    rows=np.load(directory/ARRAYS['raw_rows'],mmap_mode='r',allow_pickle=False)
    try:
        if sequence_length!=256 or not 0<=index<=len(values)-256:raise ValueError('One complete original stride1window required')
        if valid.shape!=values.shape or rows.shape!=values.shape:raise ValueError('Complete target/source-row identities required')
        return dict(timeseries=values[index:index+256].reshape(1,256).copy(),
            valid_targets=valid[index:index+256].reshape(1,256).copy(),
            input_mask=np.ones(256),raw_CSV_rows=rows[index:index+256].copy(),window_index=index)
    finally:
        for array in [values,valid,rows]:array._mmap.close()


def registration():
    previous=read(ROOT/source.producer.CONFIG);jobs=source.producer.verify(previous)
    success,failed,pins=source.source_partition(previous,jobs)
    files=[source.producer.CONFIG,audit.CONFIG,
        'scripts/flow_matching/prepare_moment_icm_anomaly_finite_control_v3.py',
        'tests/test_fm_moment_icm_anomaly_finite_control_v3.py']
    files += [p['path'] for p in read(ROOT/audit.CONFIG)['source_receipts']]
    pins += [dict(path=p,sha256=sha(ROOT/p)) for p in sorted(set(files))]
    return dict(schema_version=3,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        original_preparation_configuration=source.producer.CONFIG,required_readonly_audit=audit.CONFIG,
        original_successful_sources=len(success),required_restored_sources=len(failed),jobs=failed,
        track='source_cubic_residual_training_mean_with_raw_observation_target_validity',
        environment=dict(previous['environment'],scipy='1.10.1'),source_root=previous['source_root'],
        python=previous['python'],settings=previous['settings'],resource_lock=previous['resource_lock'],source_receipts=pins,
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_anomaly_finite_control_v3',
        state_root='experiments/runs/fm_moment_icm_anomaly_finite_control_v3',
        boundary=('Separate explicitlychanged preprocessing control for322failedsource cases,not historical '
            'source reproduction. Preservecomplete originalNaNoutputs;finite sourcecoordinatesbyteequal. '
            'Replace only nonfinite standardizedinputs withtrainingmeanzero,keep allrowIDs andexact '
            'source3splits/scaler/cubic/fullCSVinterpolation. RawunobservedORsourceinvalid targets '
            'excluded byexplicit validtargetmask frommaskedreconstructionloss;never supervise '
            'fabricatedzeros or interpolatedrawmissingtargets. Originalinputpaddingmask remains '
            'separate. Allmissingtrain scaler errors preserved. Sourcefuturecontext notcausal/TAB '
            'certified. Original1627successes/322failures untouched. Require actual full322readonly '
            'audit and1627cacheclosure first. Same16/21/16/3GiBpolicy,no neuralfit/results or '
            'completePILEtrainingrunner integration claimed.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen finite-control input changed: '+pin['path'])
    previous=read(ROOT/config['original_preparation_configuration']);jobs=source.producer.verify(previous)
    success,failed,pins=source.source_partition(previous,jobs)
    if (config['jobs']!=failed or len(failed)!=322 or len(success)!=1627
            or config['required_restored_sources']!=322 or config['original_successful_sources']!=1627
            or config['environment']!=dict(previous['environment'],scipy='1.10.1')
            or config['settings']!=previous['settings'] or any(j['family']!='OPPORTUNITY' for j in failed)
            or config['track']!='source_cubic_residual_training_mean_with_raw_observation_target_validity'):
        raise ValueError('Exact full322cases and explicitly different validity-aware preprocessing required')
    return failed


def audit_prerequisite(config):
    c=read(ROOT/config['required_readonly_audit']);proof=Path(c['artifact_root'])/audit.REPORT
    guard=ROOT/c['state_root']/'guard_receipt.json'
    if not proof.exists() or not guard.exists():return False
    p,g=read(proof),read(guard)
    if (not p['passed'] or p['source_audits']!=322 or p['original_successful_caches_independently_closed']!=1627
            or p['configuration_sha256']!=sha(ROOT/audit.CONFIG) or not g['passed'] or g['exit_code']!=0
            or g['resource_guard_aborted'] or g['proof_sha256']!=sha(proof)):
        raise ValueError('Complete successful actual readonly source audit prerequisite required')
    return True


def worker(config):
    jobs=verify(config)
    if not audit_prerequisite(config):raise ValueError('Actual complete readonlyaudit prerequisite missing')
    source.install_private_path(config)
    import numpy as np
    import pandas as pd
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact private compatible environment required')
    cls=source.producer.source_anomaly_class(ROOT/config['source_root'])
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    progress=status_writer(output/'progress.json',write);records=[];failed={}
    for job in jobs:
        folder=output/job['id']
        try:
            raw,rows=source.producer.csv_identity(job['raw'])
            raw_values=pd.read_csv(ROOT/job['raw']['path']).iloc[:,0].to_numpy();observed=np.isfinite(raw_values)
            if len(rows)!=len(observed):raise ValueError('Complete raw signal observation identity required')
            splits={}
            for split in source.producer.SPLITS:
                data=source.producer.source_dataset(cls,ROOT/job['raw']['path'],split)
                splits[split]=export_split(data,job,rows,observed,folder/split)
                del data;gc.collect()
            proof=folder/'finite_control_preparation.json'
            write(proof,dict(passed=True,job=job,configuration_sha256=sha(ROOT/CONFIG),
                environment=actual,raw_CSV_audit=raw,splits=splits,
                historical_source_equivalence_certified=False,neural_fit_inference_SVM=0))
            records.append(dict(id=job['id'],proof=str(proof),sha256=sha(proof)))
        except Exception as error:
            failed[job['id']]=dict(type=type(error).__name__,message=str(error))
            folder.mkdir(parents=True,exist_ok=True);write(folder/'failed_preserved.json',failed[job['id']])
        progress(output/'progress.json',dict(prepared_sources=len(records),required=322,failed_preserved=failed))
    write(output/REPORT,dict(passed=not failed and len(records)==322,configuration_sha256=sha(ROOT/CONFIG),
        required_sources=322,records=records,failed_preserved=failed,environment=actual,
        source_cache_or_raw_mutated=False,neural_fit_inference_SVM=0,
        independent_complete_cache_closure=False,complete_PILE_neural_training_integration=False,
        historical_source_equivalence_certified=False,all_original_experiments_complete=False))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'prerequisite_owner.lock')
    if owner is None:raise RuntimeError('Existing finite-control prerequisite owner preserved')
    publish=status_writer(base/'status.json',write)
    try:
        while not audit_prerequisite(config):
            publish(base/'status.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
                active_pid=None,state='waiting_actual322_source_audit_and1627_cache_closure',required_sources=322,
                all_original_experiments_complete=False))
            prior=read(ROOT/read(ROOT/audit.CONFIG)['state_root']/'status.json')
            if prior.get('state')=='failed_preserved_without_retry':
                raise ValueError('Readonly prerequisite failed;finite control not launched')
            time.sleep(10)
        path=ROOT/'scripts/flow_matching/verify_moment_icm_PILE_anomaly_v1.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='controller')
        replacements={'closing_all1949_anomaly_sources':'preparing_all322_explicit_finite_controls',
            'all1949_anomaly_cache_closure_passed':'all322_finite_control_preparation_passed'}
        class Names(ast.NodeTransformer):
            def visit_Constant(self,n):
                if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
                return n
        node=Names().visit(node);ast.fix_missing_locations(node)
        ns=dict(guard_template.controller.__globals__);ns.update(MODULE=MODULE,CONFIG=CONFIG,REPORT=REPORT)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<finite-control-guard>','exec'),ns)
        ns['controller'](config)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true')
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen explicit finite-control registration')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,explicit_finite_control_sources=322,original_source_files_modified=False,original_source_outputs_equivalent=False,
        original_finite_coordinates_unchanged=True,all_row_ids_kept=True,target_validity_required=True,neural_integration_complete=False)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
