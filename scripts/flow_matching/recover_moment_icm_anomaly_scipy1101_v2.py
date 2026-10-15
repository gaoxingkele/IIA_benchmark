"""Separate dependency repair for source cases that could not enter cubic interpolation.

Never mutate base environments, successful caches, or the original failure proofs.
Execute an unchanged private copy of the original preparation worker with a
verified private SciPy1.10.1 package; the source data class remains unmodified.
"""
from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.flow_matching import prepare_moment_icm_PILE_anomaly_v1 as producer
from scripts.flow_matching.resume_anomaly_transformer_full500_metadata_v1 import status_writer

ROOT,read,sha,write=producer.ROOT,producer.read,producer.sha,producer.write
MODULE='scripts.flow_matching.recover_moment_icm_anomaly_scipy1101_v2'
CONFIG='configs/reproducibility/fm_moment_ICM_anomaly_scipy1101_recovery.2026-10-15_v2.json'
PACKAGE='configs/reproducibility/fm_moment_anomaly_scipy1101_private_package.2026-10-15_v1.json'
REASON="Pandas requires version '1.10.0' or newer of 'scipy' (version '1.9.3' currently installed)."


def source_partition(previous,jobs):
    state=read(ROOT/previous['state_root']/'status.json')
    if state['state']!='all18_anomaly_family_attempts_terminal' or state['active_pid'] is not None:
        raise ValueError('All original family attempts must be terminal before repair registration')
    original_success={};failed_jobs=[];family_pins=[]
    for family in sorted({j['family'] for j in jobs}):
        proof_path=Path(previous['artifact_root'])/family/'family_preparation.json';proof=read(proof_path)
        guard_path=ROOT/previous['state_root']/'attempts'/family/'guard_receipt.json';guard=read(guard_path)
        subset=[j for j in jobs if j['family']==family];expected={j['id'] for j in subset}
        ok={r['id']:r for r in proof['records']};bad=proof['failed_preserved']
        if (proof['configuration_sha256']!=sha(ROOT/producer.CONFIG) or proof['environment']!=previous['environment']
                or len(ok)!=len(proof['records']) or set(ok)&set(bad) or set(ok)|set(bad)!=expected
                or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(proof_path)
                or guard['exact_child_handle']['command']!=[previous['python'],'-B','-X','utf8','-u','-m',producer.MODULE,'--family',family]
                or guard['passed']!=(not bad) or proof['passed']!=(not bad)):
            raise ValueError('Exact original successful/failed source partition and guarded commands required')
        original_success.update(ok)
        for job in subset:
            if job['id'] in bad:
                if bad[job['id']]['type']!='ImportError' or bad[job['id']]['message']!=REASON:
                    raise ValueError('Only diagnosed pandas/SciPy optional-dependency failures may enter this repair')
                failure_path=Path(previous['artifact_root'])/family/job['id']/'failed_preserved.json'
                if read(failure_path)!=bad[job['id']]:raise ValueError('Exact failed source record required')
                failed_jobs.append(dict(**job,original_failure=dict(path=str(failure_path),sha256=sha(failure_path))))
        family_pins.extend([dict(path=str(proof_path),sha256=sha(proof_path)),dict(path=str(guard_path),sha256=sha(guard_path))])
    if len(original_success)+len(failed_jobs)!=1949 or not failed_jobs:raise ValueError('Complete original1949 partition required')
    return original_success,failed_jobs,family_pins


def install_private_path(config):
    if any(k=='scipy' or k.startswith('scipy.') for k in sys.modules):
        raise ValueError('Fresh private repair worker required;never hot-swap imported SciPy')
    package=read(ROOT/PACKAGE);verified=read(Path(package['directory'])/'package_verified.json')
    if not verified['passed'] or verified['sha256']!=package['expected_sha256']:
        raise ValueError('Official wheel checksum verification required')
    wheel=Path(package['directory'])/package['filename']
    if sha(wheel)!=package['expected_sha256']:raise ValueError('Private original wheel changed')
    target=Path(package['private_target'])
    manifest=read(Path(package['directory'])/'installed_manifest.json')
    for pin in manifest['files']:
        if sha(target/pin['path'])!=pin['sha256']:raise ValueError('Private installed package changed')
    sys.path.insert(0,str(target))
    if importlib.metadata.version('scipy')!='1.10.1':raise ValueError('Exact private SciPy version required')


def private_worker(config,family,jobs):
    install_private_path(config)
    path=ROOT/'scripts/flow_matching/prepare_moment_icm_PILE_anomaly_v1.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    namespace=dict(producer.worker.__globals__);namespace['CONFIG']=CONFIG
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'::<isolated-original-worker>', 'exec'),namespace)
    # Original jobs only; repair provenance stays in the frozen v2 registration.
    clean_jobs=[{k:value for k,value in j.items() if k!='original_failure'} for j in jobs]
    namespace['worker'](config,family,clean_jobs)


def registration():
    previous=read(ROOT/producer.CONFIG);jobs=producer.verify(previous)
    success,failed,pins=source_partition(previous,jobs)
    preflight=ROOT/'configs/reproducibility/fm_moment_anomaly_finite_repair_preflight.json'
    if not preflight.exists() or not read(preflight).get('all_source_outputs_finite',False):
        raise ValueError('Original cubic interpolation leaves residual nonfinite values; a separate audited finite-data adapter is still required. No failed source repair is launched.')
    package=read(ROOT/PACKAGE);manifest=Path(package['directory'])/'installed_manifest.json'
    if not manifest.exists():raise ValueError('Private package must be installed and fully hashed before registration')
    files=[producer.CONFIG,PACKAGE,'scripts/flow_matching/recover_moment_icm_anomaly_scipy1101_v2.py',
        'tests/test_fm_moment_icm_anomaly_scipy1101_v2.py',str(manifest),str(Path(package['directory'])/'package_verified.json')]
    pins += [dict(path=p,sha256=sha(ROOT/p)) for p in files]
    return dict(schema_version=2,paper_id='moment_long_context',paper_citation=previous['paper_citation'],
        producer_configuration=producer.CONFIG,package_configuration=PACKAGE,original_successful_sources=len(success),
        required_failed_source_repairs=len(failed),jobs=failed,original_source_scope=1949,
        original_source_root=previous['artifact_root'],source_root=previous['source_root'],
        environment=dict(previous['environment'],scipy='1.10.1'),python=previous['python'],
        settings=previous['settings'],resource_lock=previous['resource_lock'],source_receipts=pins,
        artifact_root='F:/aicoding/IIA_Data/experiments/moment_icm_anomaly_scipy1101_recovery_v2',
        state_root='experiments/runs/fm_moment_icm_anomaly_scipy1101_recovery_v2',
        boundary=('Only terminal sourcecases with exact diagnosed Pandas2.2.2/SciPy1.9.3 optionaldependency '
            'ImportError. Freshworker privately imports SHA256verified officialSciPy1.10.1,NumPy/Pandas/ '
            'sklearn/Python unchanged. Complete originalsource class/worker numericalbody reused unchanged '
            'through private AST namespace;no header/split/scaler/interpolation/precision/windowchange. '
            'Original successes/failures preserved;no source retry and no live/base environment mutation. '
            'Separate compatibleenvironment reconstruction,not proof of equality to previously failed '
            'source results/historicalpaper pipeline. Same16/21/16/3GiB data policy,no neuralfit/results.'))


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen diagnosed repair input changed: '+pin['path'])
    previous=read(ROOT/config['producer_configuration']);original=producer.verify(previous)
    success,failed,pins=source_partition(previous,original)
    if (config['jobs']!=failed or config['original_successful_sources']!=len(success)
            or config['required_failed_source_repairs']!=len(failed) or config['original_source_scope']!=1949
            or config['environment']!=dict(previous['environment'],scipy='1.10.1') or config['settings']!=previous['settings']):
        raise ValueError('Exact failed-only partition and one explicitly different dependency required')
    return failed


def controller(config,jobs):
    import psutil
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=install_facades();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Existing isolated repair owner preserved')
    publish=status_writer(base/'status.json',write);completed=[];failed={}
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,required_source_repairs=len(jobs))
    def update(value=None,**extra):
        state.update(value or {});state.update(extra);state.update(completed_families=completed,failed_preserved=failed);publish(base/'status.json',state)
    try:
        for family in sorted({j['family'] for j in jobs}):
            attempt=base/'attempts'/family;proof=Path(config['artifact_root'])/family/'family_preparation.json'
            try:
                if attempt.exists() or proof.parent.exists():raise ValueError('Existing repair attempt preserved;no automatic retry')
                with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],lambda x:update(x,state='waiting_original_data_memory_policy'),poll_seconds=10):
                    attempt.mkdir(parents=True,exist_ok=False);cmd=[config['python'],'-B','-X','utf8','-u','-m',MODULE,'--family',family]
                    with (attempt/'stdout.log').open('xb') as out,(attempt/'stderr.log').open('xb') as err:
                        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        handle=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=cmd)
                        write(attempt/'launch.json',handle);update(state='preparing_failed_sources_in_private_compatible_environment',active_pid=p.pid,family=family)
                        code,usage=capped_wait(p,config,api,lambda x:update(x,state='preparing_failed_sources_in_private_compatible_environment'))
                    passed=code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed']
                    write(attempt/'guard_receipt.json',dict(passed=passed,exit_code=code,**usage,exact_child_handle=handle,proof_sha256=sha(proof) if proof.exists() else None))
                    if not passed:raise ValueError('Isolated diagnosed repair failed;preserved without retry')
                completed.append(family)
            except Exception as error:failed[family]=dict(type=type(error).__name__,message=str(error))
            update(state='scanning_next_repair_family',active_pid=None,family=None)
        update(state='all_diagnosed_repair_family_attempts_terminal',active_pid=None,family=None)
    finally:owner.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true');parser.add_argument('--probe-only',action='store_true');parser.add_argument('--family');args=parser.parse_args()
    if args.register:
        if (ROOT/CONFIG).exists():raise FileExistsError('Preserve frozen diagnosed repair')
        config=registration();verify(config)
        with (ROOT/CONFIG).open('x',encoding='utf-8',newline='\n') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    else:config=read(ROOT/CONFIG);verify(config)
    if args.probe_only:print(json.dumps(dict(passed=True,failed_only_cases=len(config['jobs']),new_scipy='1.10.1',base_environments_changed=False)))
    elif args.family:
        jobs=[j for j in config['jobs'] if j['family']==args.family]
        if not jobs:raise ValueError('Exact failed source family required')
        private_worker(config,args.family,jobs)
    else:controller(config,config['jobs'])


if __name__=='__main__':main()
