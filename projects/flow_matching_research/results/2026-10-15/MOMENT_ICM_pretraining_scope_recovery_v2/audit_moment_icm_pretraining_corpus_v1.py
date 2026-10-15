"""Verify the entire registered PILE snapshot and profile all47 Monash inputs.

Both source-count and independent temporal-length eligibility remain explicit.
This is an input/one-epoch-budget audit,not an actual pretraining run.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import importlib.metadata

from scripts.flow_matching import moment_icm_pretraining_inputs_v1 as inputs
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=inputs.ROOT,inputs.read,inputs.sha,inputs.write
MODULE='scripts.flow_matching.audit_moment_icm_pretraining_corpus_v1'


def check_file(path,record):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Registered raw snapshot input missing: '+str(path))
    before=path.stat();digest=hashlib.sha256();blob=hashlib.sha1(f'blob {before.st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk);blob.update(chunk)
    after=path.stat()
    if ((before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
            or record['status']!='verified' or before.st_size!=record['size_bytes']
            or digest.hexdigest()!=record['sha256'] or blob.hexdigest()!=record['git_blob_sha1']):
        raise ValueError('Complete current raw bytes differ from registered PILE snapshot: '+str(path))
    return dict(path=record['path'],bytes=before.st_size,sha256=digest.hexdigest(),git_blob_sha1=blob.hexdigest())


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen corpus audit input changed: '+pin['path'])
    if (config['context_length']!=256 or config['source_seed']!=13 or config['monash_inputs']!=47
            or config['snapshot_files']!=2441 or config['data_source_fraction']!=1.
            or config['original_paper_pretraining_epochs']!=1
            or config['settings']!=dict(minimum_available_memory_bytes=16*2**30,minimum_commit_headroom_bytes=24*2**30,
                emergency_commit_headroom_bytes=16*2**30,maximum_worker_rss_bytes=6*2**30,maximum_worker_private_bytes=6*2**30)):
        raise ValueError('Complete registered snapshot and full one-epoch scope retained')
    groups,compat=inputs.enumerate_inputs(ROOT/config['source_root'],ROOT/config['pile_root'])
    paths={k:[Path(p).relative_to(ROOT).as_posix() for p in v] for k,v in groups.items()}
    if paths!=config['all_source_getter_paths'] or len(paths['short_forecast'])!=47:
        raise ValueError('Every source-visible dataset path and optional empty branch must remain explicit')
    return groups,compat


def worker(config):
    actual=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']})
    if actual!=config['environment']:raise ValueError('Exact metadata/parser environment required')
    groups,compat=verify(config);records=read(ROOT/config['integrity'])['records']
    if len(records)!=2441:raise ValueError('No raw registry rows may be omitted')
    output=Path(config['artifact_root']);output.mkdir(parents=True,exist_ok=False)
    progress=status_writer(output/'progress.json',write)
    raw=[];profiles=[];failures={}
    for index,record in enumerate(records):
        raw.append(check_file(ROOT/record['path'],record))
        if index%100==0:progress(output/'progress.json',dict(phase='rehashing_entire_snapshot',verified_files=len(raw),profiles=0))
    cls=inputs.short_forecast_class(ROOT/config['source_root'])
    references={r['path']:r for r in records}
    for path in groups['short_forecast']:
        relative=Path(path).relative_to(ROOT).as_posix()
        if relative not in references:raise ValueError('Source dataset has no registered raw checksum')
        try:
            row=inputs.profile_short(cls,path,config['context_length'],config['source_seed'])
            row['path']=relative;row['source_sha256']=references[relative]['sha256'];profiles.append(row)
        except Exception as error:failures[relative]=dict(type=type(error).__name__,message=str(error))
        progress(output/'progress.json',dict(phase='profiling_all47_Monash_sources',verified_files=len(raw),profiles=len(profiles),failed_preserved=failures))
    # An audit may expose real source parser failures;retain all without retry.
    report=dict(passed=len(profiles)==47 and not failures,captured_utc=datetime.now(timezone.utc).isoformat(),
        configuration_sha256=sha(ROOT/config['_config_path']),environment=actual,
        snapshot_file_count=len(raw),snapshot_bytes=sum(r['bytes'] for r in raw),raw_snapshot_checks=raw,
        complete_source_visible_paths=config['all_source_getter_paths'],path_compatibility=compat,
        Monash_profiles=profiles,profile_failures_preserved=failures,
        source_count_selected=[r['path'] for r in profiles if r['original_source_field_selected']],
        observed_training_time_selected=[r['path'] for r in profiles if r['observed_training_time_selected']],
        added_by_temporal_control=[r['path'] for r in profiles if r['observed_training_time_selected'] and not r['original_source_field_selected']],
        excluded_by_temporal_control=[r['path'] for r in profiles if r['original_source_field_selected'] and not r['observed_training_time_selected']],
        source_selected_training_items=sum(r['actual_training_rows'] for r in profiles if r['original_source_field_selected']),
        temporal_selected_training_items=sum(r['actual_training_rows'] for r in profiles if r['observed_training_time_selected']),
        temporal_policy_is_separate_control=True,source_scaling_and_random_window_sampling_not_executed=True,
        historical_ICM_pretraining_corpus_equivalence_certified=False,original_pretraining_epochs_executed=0,
        neural_fit_inference_or_SVM=0,raw_data_overwritten=False,all_original_experiments_complete=False)
    write(output/'full_corpus_and_temporal_gate_audit.json',report)
    if not report['passed']:raise RuntimeError('Source profiling failures retained;no automatic retry or corpus reduction')
    print(json.dumps(dict(passed=True,raw_files=len(raw),Monash_profiles=len(profiles),
        source_count_selected=len(report['source_count_selected']),temporal_selected=len(report['observed_training_time_selected']),
        added=len(report['added_by_temporal_control']),excluded=len(report['excluded_by_temporal_control']),neural_fits=0)))


def controller(config):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('Corpus audit already has a live owner')
    publish=status_writer(base/'status.json',write)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None)
    def update(value):state.update(value);publish(base/'status.json',state)
    try:
        if (base/'worker_launch.json').exists():raise ValueError('Prior attempt preserved;no automatic retry')
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update,poll_seconds=10):
            command=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--config',config['_config_path'],'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=command)
                write(base/'worker_launch.json',handle);update(dict(state='auditing_entire_actual_pretraining_input_snapshot',active_pid=p.pid,active_create_time=handle['create_time']))
                code,usage=capped_wait(p,config,api,lambda value:update(dict(value,state='auditing_actual_corpus_bytes_and47profiles')))
            proof=Path(config['artifact_root'])/'full_corpus_and_temporal_gate_audit.json'
            passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
            write(base/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,
                proof_sha256=sha(proof) if proof.exists() else None))
            update(dict(state='complete_snapshot_and47profiles_verified' if passed else 'failed_preserved_without_retry',active_pid=None))
    finally:lane.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--probe-only',action='store_true');args=p.parse_args()
    config=read(ROOT/args.config);verify(config);config['_config_path']=args.config
    if args.probe_only:print(json.dumps(dict(passed=True,snapshot_files=2441,Monash_inputs=47,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules)))
    elif args.worker:worker(config)
    else:controller(config)


if __name__=='__main__':main()
