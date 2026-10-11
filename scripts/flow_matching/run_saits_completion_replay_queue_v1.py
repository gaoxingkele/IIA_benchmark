"""Automatically verify all210 original imputation jobs after complete native training."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.publish_saits_electricity20_seed26_v50 import exclusive
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def select_jobs(queue):
    if len(queue['jobs']) != 210:
        raise ValueError('Retain every210 original author/corrected SAITS/BRITS/base job')
    jobs = {}
    for item in queue['jobs']:
        path = ROOT / item['config']; native = read(path)
        if (sha(path) != item['config_sha256'] or item['runner'] != 'scripts/flow_matching/run_saits.py'
                or native.get('diagnostic') or native.get('limit_test_cases')
                or native['device'] != 'cuda' or native['bootstrap_group'] != 'month'
                or native['seed'] not in [26,27,28,29,30]
                or not re.fullmatch(r'[A-Za-z0-9_.]+', native['id'])
                or native['id'] in jobs):
            raise ValueError('Keep unique full original canonical configs, seeds, data and commands')
        jobs[native['id']] = dict(item=item, native=native)
    return jobs


def check_completed_result(native, result):
    if (result['status'] != 'completed' or result['config'] != native
            or result['config_sha256'] is None or result['dataset_sha256'] != native['dataset_sha256']
            or result['bootstrap_group'] != 'month' or result['parameters'] <= 0
            or result['test_windows'] <= 0 or result['target_count'] <= 0
            or result.get('benchmark_smoke') or result.get('diagnostic')):
        raise ValueError('A complete unchanged original native result is required before readonly replay')


def successful_producer(path):
    config = read(ROOT / path); receipt = read(ROOT / config['state_root'] / 'receipt.json')
    report = read(ROOT / config['artifact_directory'] / 'checkpoint_replay_audit.json')
    from scripts.flow_matching.verify_saits_completed_full_replay_batch_v1 import check_terminal
    check_terminal(report, receipt, config)
    if report['configuration_sha256'] != sha(ROOT / path):
        raise ValueError('Existing complete native report must bind its exact producer configuration')
    return config, report


def catalog(runtime, jobs):
    native_by_id = {}; verified = set()
    for path in runtime['known_completed_native_configurations']:
        config, report = successful_producer(path)
        for row in report['records']:
            if row['id'] not in jobs or row['id'] in native_by_id:
                raise ValueError('Unique existing original full-model proofs required')
            if sha(ROOT / row['config']) != row['config_sha256'] or sha(Path(row['result_path'])) != row['result_sha256']:
                raise ValueError('Existing native model/result content differs')
            native_by_id[row['id']] = path
    for path in runtime['known_completed_independent_configurations']:
        config = read(ROOT / path); producer, report = successful_producer(config['replay_configuration'])
        proof_path = ROOT / config['output_directory'] / 'independent_full_replay_batch_verification.json'
        review = read(proof_path); receipt = read(ROOT / config['state_root'] / 'receipt.json')
        if (not review['passed'] or not review['all_original_grid_coordinates_independently_checked']
                or review['native_replay_report_sha256'] != sha(ROOT / producer['artifact_directory'] / 'checkpoint_replay_audit.json')
                or receipt['exit_code'] != 0 or receipt['resource_guard_aborted']
                or not receipt['full_native_batch_independent_review_passed']
                or {r['id'] for r in review['records']} != {r['id'] for r in report['records']}):
            raise ValueError('Actual complete independent array proof and unchanged guard receipt required')
        verified.update(r['id'] for r in review['records'])
    return native_by_id, verified


def verify(runtime):
    for pin in runtime['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen automatic original completion binding changed')
    queue = read(ROOT / runtime['queue']); jobs = select_jobs(queue)
    template = read(ROOT / runtime['native_replay_template'])
    if (runtime['settings'] != template['settings'] or runtime['python'] != template['python']
            or runtime['resource_lock'] != template['resource_lock']
            or runtime['cpu_threads'] != template['cpu_threads']
            or template['metric_tolerance'] != dict(rtol=2e-6, atol=2e-6)):
        raise ValueError('The original full native environment, thread budget, gates and tolerance must remain')
    return jobs, template


def native_registration(runtime, template, entry):
    native = entry['native']; output = ROOT / native['output_root']; result = read(output / 'result.json')
    check_completed_result(native, result)
    if sha(ROOT / entry['item']['config']) != result['config_sha256']:
        raise ValueError('Actual completed original configuration hash differs')
    config = {k: copy.deepcopy(template[k]) for k in ('python', 'cpu_threads', 'metric_tolerance',
        'metric_tolerance_source', 'settings', 'original_scope', 'resource_lock')}
    identity = native['id']; config.update(full_model_count=1,
        completed_jobs=[dict(id=identity, config=entry['item']['config'],
            result_path=str(output / 'result.json'), result_sha256=sha(output / 'result.json'))],
        artifact_directory=str(Path(runtime['artifact_root']) / identity),
        state_root=runtime['child_state_root'] + '/' + identity + '/native',
        boundary='Actual completed original canonical model only. Keep all original architecture, early stopping, '
        'data, native full-test windows, coordinates and month groups. Full native reinference is not a new training seed. '
        'Original210job queue and full45/681/273scope retained; no publication, author/TAB certification or retry.')
    files = [entry['item']['config']] + [str(output / n) for n in
        ('result.json','resume.pt','best.pt','group_statistics.npz','training.log')]
    files += [str(ROOT / native['dataset_root'] / n) for n in ('datasets.h5','audit_arrays.npz','data_audit.json')]
    source = ROOT / result['patch_manifest']['working_path']
    files += [str(p) for p in source.rglob('*.py')]
    config['source_receipts'] = runtime['code_receipts'] + [dict(path=p,sha256=sha(ROOT / p)) for p in files]
    path = runtime['generated_config_root'] + '/' + identity + '.native.json'
    exclusive(ROOT / path, config)
    return path


def launch(runtime, path, module, directory):
    directory.mkdir(parents=True, exist_ok=False)
    command = [runtime['controller_python'], '-X','utf8','-u','-m',module,'--config',path]
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),creationflags=subprocess.CREATE_NO_WINDOW)
    handle = psutil.Process(child.pid)
    receipt = dict(pid=child.pid,create_time=handle.create_time(),actual_command=handle.cmdline())
    exclusive(directory/'launch.json',receipt)
    return child, receipt


def review_registration(runtime, producer_path, handle, identity):
    producer = read(ROOT / producer_path)
    config = dict(replay_configuration=producer_path, required_full_models=producer['full_model_count'],
        python=producer['python'],settings=producer['settings'],resource_lock=producer['resource_lock'],
        original_scope=producer['original_scope'],producer_controller_pid=handle['pid'],
        producer_controller_create_time=handle['create_time'],
        state_root=runtime['child_state_root']+'/'+identity+'/review',
        output_directory=runtime['review_output_root']+'/'+identity,
        source_receipts=runtime['code_receipts']+[dict(path=producer_path,sha256=sha(ROOT/producer_path))],
        boundary='Independent every-coordinate original HDF5/mask/unit/month/checkpoint/optimizer/RNG/early-stop review. '
        'Same native full-model count, existing tolerance and resource settings. No new model inference, '
        'training seed, numeric publication or automatic retry; every failure is preserved.')
    path = runtime['generated_config_root']+'/'+identity+'.review.json'; exclusive(ROOT/path,config)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args();runtime=read(ROOT/args.config)
    jobs,template=verify(runtime);native_by_id,verified=catalog(runtime,jobs)
    if args.probe_only:
        print(json.dumps(dict(original_jobs=len(jobs),existing_native_models=len(native_by_id),
            existing_independently_verified_models=len(verified),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules)));return
    api=resource_api();base=ROOT/runtime['state_root'];base.mkdir(parents=True,exist_ok=True)
    lane=api['exclusive_lock'](base/'queue.lock')
    if lane is None:raise RuntimeError('Original automatic completion queue already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        required_original_models=210,verified_original_ids=sorted(verified),failed_preserved={},
        all_original_experiments_complete=False)
    def update(value):
        state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    def wait_child(child,handle,phase):
        while child.poll() is None:
            try:
                p=psutil.Process(handle['pid'])
                if p.create_time()!=handle['create_time'] or p.cmdline()!=handle['actual_command']:
                    raise RuntimeError('Exact owned native/review child identity changed')
                t=p.cpu_times();update(dict(state='waiting_actual_'+phase,active_pid=p.pid,
                    active_create_time=p.create_time(),actual_command=p.cmdline(),actual_child_cpu_seconds=t.user+t.system))
            except psutil.NoSuchProcess:
                if child.poll() is None:
                    time.sleep(.1);continue
            time.sleep(5)
        update(dict(active_pid=None));return child.returncode
    try:
        while True:
            scheduled=False
            for identity,entry in jobs.items():
                if identity in verified or identity in state['failed_preserved']:continue
                existing = native_by_id.get(identity)
                output=ROOT/entry['native']['output_root']/'result.json'
                if existing is None and not output.exists():continue
                if existing is None:
                    try:result=read(output)
                    except json.JSONDecodeError:continue  # Native final result write can still be in progress.
                    if result.get('status')!='completed':continue
                    producer_path=native_registration(runtime,template,entry)
                    child,handle=launch(runtime,producer_path,'scripts.flow_matching.audit_saits_full_checkpoint_replay_v1',base/identity/'native_dispatch')
                    update(dict(active_job=identity));code=wait_child(child,handle,'full_native_replay')
                    producer=read(ROOT/producer_path);receipt=read(ROOT/producer['state_root']/'receipt.json') if (ROOT/producer['state_root']/'receipt.json').exists() else {}
                    if code!=0 or receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') or not receipt.get('full_native_model_replay_passed'):
                        state['failed_preserved'][identity]=dict(phase='native',exit_code=code,configuration=producer_path,receipt=receipt);update({});scheduled=True;continue
                else:
                    producer_path=existing;producer,report=successful_producer(existing)
                    prior=read(ROOT/producer['state_root']/'status.json')
                    handle=dict(pid=prior['pid'],create_time=prior['create_time'])
                review_id='existing_'+sha(ROOT/producer_path)[:16] if existing else identity
                path=review_registration(runtime,producer_path,handle,review_id)
                child,review_handle=launch(runtime,path,'scripts.flow_matching.verify_saits_completed_full_replay_batch_v1',base/review_id/'review_dispatch')
                update(dict(active_job=identity));code=wait_child(child,review_handle,'independent_full_array_review')
                config=read(ROOT/path);receipt_path=ROOT/config['state_root']/'receipt.json'
                receipt=read(receipt_path) if receipt_path.exists() else {}
                ids=[r['id'] for r in producer['completed_jobs']]
                if code==0 and receipt.get('exit_code')==0 and not receipt.get('resource_guard_aborted') and receipt.get('full_native_batch_independent_review_passed'):
                    proof=read(ROOT/config['output_directory']/'independent_full_replay_batch_verification.json')
                    if not proof['passed'] or {r['id'] for r in proof['records']}!=set(ids):raise ValueError('Every scheduled original model must be independently checked')
                    verified.update(ids);update(dict(verified_original_ids=sorted(verified)))
                else:
                    for id in ids:state['failed_preserved'][id]=dict(phase='independent',exit_code=code,configuration=path,receipt=receipt)
                    update({})
                scheduled=True
            if len(verified)+len(state['failed_preserved'])==210:
                update(dict(state='all_original_models_verified' if len(verified)==210 else 'all_jobs_terminal_with_failures',
                    full_registered_imputation_queue_verified=len(verified)==210,active_pid=None));return
            if not scheduled:
                from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
                handle=process_identity(runtime['training_controller']['pid'],runtime['training_controller']['create_time'],
                    'run_saits_guarded_original_queue_v1')
                update(dict(state='waiting_actual_original_training_completion',actual_training_controller=handle,
                    pending_original_models=210-len(verified)-len(state['failed_preserved']),active_pid=None))
                time.sleep(10)
    finally:lane.close()


if __name__=='__main__':main()
