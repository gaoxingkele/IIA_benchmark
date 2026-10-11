"""Continue full checkpoint replay and independent array review for all285 CFM jobs."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


TOLERANCE = dict(rtol=1e-7, atol=1e-8)
EXECUTION_FIELDS = {'id', 'output_directory', 'artifact_directory', 'recovery_original_job_id'}


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_binding(original, job):
    canonical = job.get('recovery_original_job_id', job['id'])
    if (canonical != original['id'] or job['seed'] not in [1103,1104,1105,1106,1107]
            or {k:v for k,v in original.items() if k not in EXECUTION_FIELDS}
            != {k:v for k,v in job.items() if k not in EXECUTION_FIELDS}):
        raise ValueError('Recovery must preserve every original scientific job field')


def check_native_terminal(config, report, receipt):
    expected = config['full_results']
    if (receipt.get('exit_code') != 0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('full_replay_passed') is not True
            or report.get('full_checkpoint_contents_and_re_inference_passed') is not True
            or report.get('prediction_tolerance') != TOLERANCE
            or report.get('cuda_context_initialized') is not False
            or report.get('additional_training_seed_slots') != 0
            or report.get('cpu_threads') != config['cpu_threads']
            or report.get('full_job_count') != len(expected)
            or len(report.get('records', [])) != len(expected)):
        raise ValueError('Successful actual full native replay terminal receipt required')
    rows = {r['original_canonical_id']:r for r in report['records']}
    if len(rows) != len(expected) or set(rows) != {r['original_canonical_id'] for r in expected}:
        raise ValueError('Every original canonical replay must be covered exactly once')
    for item in expected:
        row = rows[item['original_canonical_id']]
        if (row['result_path'] != item['result_path'] or row['result_sha256'] != item['result_sha256']
                or row['execution_id'] != item['job']['id']):
            raise ValueError('Native proof must bind the selected immutable execution')


def verify_record(config, canonical, replay):
    """Same all-coordinate checks as the frozen full replay union, for one new model."""
    import numpy as np
    from scripts.flow_matching.audit_cfm_full_arrays_v1 import audit_arrays, prediction_metrics
    result = read(ROOT / replay['result_path'])
    job = result['job']
    if (config['prediction_tolerance'] != TOLERANCE
            or result['job'] != canonical['job'] or result['status'] != 'completed'
            or replay['original_canonical_id'] != canonical['original_canonical_id']
            or replay['result_path'] != canonical['result_path']
            or replay['result_sha256'] != canonical['result_sha256']
            or sha(ROOT / replay['result_path']) != replay['result_sha256']
            or sha(replay['checkpoint_path']) != replay['checkpoint_sha256']
            or sha(replay['replay_arrays']) != replay['replay_arrays_sha256']
            or sha(ROOT / job['data_path']) != job['data_sha256']
            or replay['original_full_data_sha256'] != job['data_sha256']
            or sha(ROOT / job['model_config']) != job['model_config_sha256']
            or replay['epochs'] != job['epochs']
            or replay['optimizer_updates'] != job['expected_optimizer_updates']
            or replay['execution_id'] != job['id'] or replay['seed'] != job['seed']
            or replay['checkpoint_contents_and_full_budget_verified'] is not True
            or replay['original_full_grid_replayed'] is not True):
        raise ValueError('Original checkpoint/data/result/budget binding differs')
    artifacts = {Path(a['path']).name:a for a in result['artifacts']}
    for artifact in artifacts.values():
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Preserved training artifact changed')
    if (Path(artifacts['model.pt']['path']).resolve() != Path(replay['checkpoint_path']).resolve()
            or artifacts['model.pt']['sha256'] != replay['checkpoint_sha256']):
        raise ValueError('Replay checkpoint must match actual completed training')
    with np.load(ROOT / job['data_path'], allow_pickle=False) as data, \
            np.load(artifacts['predictions.npz']['path'], allow_pickle=False) as original, \
            np.load(replay['replay_arrays'], allow_pickle=False) as reproduced:
        audit_arrays(job, result, original, data)
        for field in ('truth', 'times', 'test_ids'):
            if not np.array_equal(reproduced[field], original[field]):
                raise ValueError('Full replay test identity differs: '+field)
        shape = (job['test_trajectories'], job['test_points_per_trajectory'], job['observed_dimensions'])
        if tuple(replay['full_replay_shape']) != shape or reproduced['predictions'].shape != shape:
            raise ValueError('Every original trajectory/grid/dimension coordinate required')
        if not np.isfinite(reproduced['predictions']).all() or not np.allclose(
                reproduced['predictions'], original['predictions'], **TOLERANCE):
            raise ValueError('Full replay predictions differ under original frozen tolerance')
        if replay['bitwise_identical'] != bool(np.array_equal(reproduced['predictions'], original['predictions'])):
            raise ValueError('Bitwise-equivalence evidence differs')
        difference = float(np.max(np.abs(reproduced['predictions'].astype(np.float64)
                                             - original['predictions'].astype(np.float64))))
        if difference != replay['max_absolute_difference']:
            raise ValueError('Actual maximum prediction difference differs')
        metrics, _ = prediction_metrics(reproduced['predictions'], reproduced['truth'])
        for key,value in metrics.items():
            recorded = replay['replayed_metrics'][key]
            if ((value is None) != (recorded is None)
                    or value is not None and not np.allclose(value, recorded, rtol=1e-12, atol=1e-12)):
                raise ValueError('Independent full replay metric differs')
    return dict(replay, all_original_grid_coordinates_independently_checked=True,
        source_full_model_replay_and_current_artifacts_verified=True)


def verify(runtime):
    for pin in runtime['source_receipts']:
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen automatic full replay source changed: '+pin['path'])
    original = read(ROOT / runtime['original_queue'])
    jobs = {j['id']:j for j in original['jobs']}
    routes = runtime['routes']
    if (len(jobs) != 285 or len(routes) != 285 or len({r['original_canonical_id'] for r in routes}) != 285
            or {r['original_canonical_id'] for r in routes} != set(jobs)):
        raise ValueError('Keep every285 original canonical experiment, without duplicate seeds')
    template = read(ROOT / runtime['native_replay_template'])
    if (template['prediction_tolerance'] != TOLERANCE or runtime['prediction_tolerance'] != TOLERANCE
            or any(runtime[k] != template[k] for k in ['python','cpu_threads','settings','resource_lock','environment_lock'])
            or runtime['original_scope'] != dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)):
        raise ValueError('Original full native environment, solver tolerance and all stronger resource limits required')
    for route in routes:
        check_binding(jobs[route['original_canonical_id']], route['job'])
        if route['result_path'] != (Path(route['job']['output_directory'])/'result.json').as_posix():
            raise ValueError('Result must use the explicit frozen selected execution directory')
    preserved = read(ROOT / runtime['preserved_verified_union'])
    records = {r['original_canonical_id']:r for r in preserved['records']}
    if (len(records) != 85 or len(preserved['records']) != 85
            or preserved['full_canonical_models_verified'] != 85
            or preserved['full_checkpoint_contents_and_re_inference_passed'] is not True
            or preserved['all_original_test_points_covered'] is not True
            or preserved['original_prediction_tolerance'] != TOLERANCE
            or preserved['additional_canonical_training_seed_slots'] != 0):
        raise ValueError('Existing85 actual full native and independent proofs required')
    for route in routes:
        record = records.get(route['original_canonical_id'])
        if record and (record['result_path'] != route['result_path']
                or record['result_sha256'] != sha(ROOT / route['result_path'])
                or record['execution_id'] != route['job']['id']
                or record['source_full_model_replay_and_current_artifacts_verified'] is not True):
            raise ValueError('Preserved85 verified executions may not be replaced or rerun')
    return routes, template, records


def native_registration(runtime, template, route):
    result = read(ROOT / route['result_path']); job = route['job']
    from scripts.flow_matching.run_cfm_ts_original_job import fingerprint
    if (result['status'] != 'completed' or result['job'] != job
            or result['job_sha256'] != fingerprint(job) or result['epochs_completed'] != job['epochs']
            or result['optimizer_updates'] != job['expected_optimizer_updates']
            or result['test_prediction_shape'] != [job['test_trajectories'],job['test_points_per_trajectory'],job['observed_dimensions']]
            or result['diagnostic_smoke'] is not False or result['author_equivalence_certified'] is not False):
        raise ValueError('Only an unchanged complete original training result may enter replay')
    config = copy.deepcopy(template)
    identity = route['original_canonical_id']
    config.update(full_job_count=1, full_results=[dict(route,result_sha256=sha(ROOT / route['result_path']))],
        state_root=runtime['child_state_root']+'/'+identity+'/native',
        output_directory=str(Path(runtime['artifact_root'])/identity/'native'),
        original_canonical_queue=runtime['original_queue'], original_canonical_slots=285,
        source_receipts=copy.deepcopy(runtime['source_receipts']),
        boundary='Complete frozen original canonical training only; full original grid/solver/environment, unchanged '
        'prediction tolerance and guard. One deterministic post-training replay, no retraining or new seed. '
        'Independent every-coordinate review is separately required; original-author equivalence remains unproved.')
    config['source_receipts'].append(dict(path=route['result_path'],sha256=sha(ROOT / route['result_path'])))
    return config


def review_worker(config, config_path):
    producer = read(ROOT / config['native_configuration'])
    report_path = ROOT / producer['output_directory'] / 'checkpoint_replay_audit.json'
    report = read(report_path); receipt = read(ROOT / producer['state_root'] / 'receipt.json')
    if report['configuration_sha256'] != sha(ROOT / config['native_configuration']):
        raise ValueError('Actual full native report must bind its immutable configuration')
    check_native_terminal(producer, report, receipt)
    records = [verify_record(producer, item, row) for item,row in zip(producer['full_results'], report['records'])]
    exclusive(ROOT / config['output_proof'], dict(configuration=config_path,
        configuration_sha256=sha(ROOT / config_path), captured_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, native_report_sha256=sha(report_path), native_receipt_sha256=sha(ROOT / producer['state_root'] / 'receipt.json'),
        all_original_grid_coordinates_independently_checked=True, records=records,
        additional_training_seed_slots=0, original_scope=config['original_scope'], all_original_experiments_complete=False,
        author_equivalence_certified=False))


def run_review(config, config_path):
    api = resource_api(); base = ROOT / config['state_root']; base.mkdir(parents=True,exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('Independent full-array review already has an owner')
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting')
    def update(values): state.update(values); write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'],config['settings'],update):
            command = [config['python'],'-X','utf8','-u','-m',__name__ if __name__!='__main__'
                else 'scripts.flow_matching.run_cfm_completion_replay_queue_v1','--config',config_path,'--review-worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1'),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle = psutil.Process(child.pid)
                update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage = capped_wait(child,config,api,update)
        passed = bool(code==0 and not usage['resource_guard_aborted'] and (ROOT / config['output_proof']).is_file())
        write(base/'receipt.json',dict(exit_code=code,independent_full_array_review_passed=passed,**usage))
        update(dict(active_pid=None,state='completed' if passed else 'failed_preserved'))
        if not passed: raise RuntimeError('Independent complete-grid review failed; all inputs preserved')
    finally: lane.close()


def controller(runtime, path):
    routes,template,verified = verify(runtime)
    api = resource_api(); base = ROOT / runtime['state_root']; base.mkdir(parents=True,exist_ok=True)
    lane = api['exclusive_lock'](base/'controller.lock')
    if lane is None: raise RuntimeError('Automatic full285 replay queue already has an owner')
    state = dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration_sha256=sha(ROOT/path),
        required_original_models=285,verified_original_ids=sorted(verified),failed_preserved={},active_pid=None,
        additional_training_seed_slots=0,all_original_experiments_complete=False)
    def update(values):
        state.update(values);state['captured_utc']=datetime.now(timezone.utc).isoformat();write(base/'status.json',state)
    def launch(config_path,module,phase,identity):
        logs=base/'logs'/identity;logs.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-X','utf8','-u','-m',module,'--config',config_path]
        if phase=='independent':command.append('--review-controller')
        with (logs/(phase+'.stdout.log')).open('xb') as out,(logs/(phase+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        handle=psutil.Process(child.pid);creation=handle.create_time();actual=handle.cmdline()
        update(dict(active_pid=child.pid,active_create_time=creation,actual_command=actual,active_job=identity))
        while child.poll() is None:
            process=psutil.Process(child.pid)
            if process.create_time()!=creation or process.cmdline()!=actual:raise ValueError('Child identity changed')
            update(dict(state='waiting_actual_full_'+phase,actual_child_cpu_seconds=sum(process.cpu_times()[:2])))
            time.sleep(5)
        code=child.wait();update(dict(active_pid=None));return code
    try:
        while True:
            dispatched=False
            for route in routes:
                identity=route['original_canonical_id']
                if identity in verified or identity in state['failed_preserved'] or not (ROOT/route['result_path']).is_file():continue
                config=native_registration(runtime,template,route)
                child_path=runtime['child_config_root']+'/'+identity+'.native.json'
                exclusive(ROOT/child_path,config)
                code=launch(child_path,'scripts.flow_matching.audit_cfm_checkpoint_replay_v1','native',identity)
                receipt_path=ROOT/config['state_root']/'receipt.json'
                receipt=read(receipt_path) if receipt_path.exists() else {}
                report_path=ROOT/config['output_directory']/'checkpoint_replay_audit.json'
                if code!=0 or not report_path.exists():
                    state['failed_preserved'][identity]=dict(phase='native',exit_code=code,configuration=child_path,receipt=receipt)
                    update({});dispatched=True;continue
                check_native_terminal(config,read(report_path),receipt)
                review={k:copy.deepcopy(runtime[k]) for k in ['python','settings','resource_lock','original_scope']}
                review.update(native_configuration=child_path,source_receipts=copy.deepcopy(runtime['source_receipts']),
                    state_root=runtime['child_state_root']+'/'+identity+'/independent',
                    output_proof=str(Path(runtime['artifact_root'])/identity/'independent_full_array_review.json'))
                review_path=runtime['child_config_root']+'/'+identity+'.review.json';exclusive(ROOT/review_path,review)
                code=launch(review_path,'scripts.flow_matching.run_cfm_completion_replay_queue_v1','independent',identity)
                review_receipt_path=ROOT/review['state_root']/'receipt.json'
                review_receipt=read(review_receipt_path) if review_receipt_path.exists() else {}
                if code==0 and review_receipt.get('exit_code')==0 and review_receipt.get('resource_guard_aborted') is False and review_receipt.get('independent_full_array_review_passed'):
                    proof=read(ROOT/review['output_proof'])
                    if not proof['passed'] or len(proof['records'])!=1 or proof['records'][0]['original_canonical_id']!=identity:
                        raise ValueError('Complete independent proof of the exact canonical execution required')
                    verified[identity]=proof['records'][0]
                    update(dict(verified_original_ids=sorted(verified)))
                    write(base/'verified_full_model_records.json',dict(records=list(verified.values()),
                        required_original_models=285,full_canonical_models_verified=len(verified),
                        all_original_experiments_complete=False,author_equivalence_certified=False))
                else:
                    state['failed_preserved'][identity]=dict(phase='independent',exit_code=code,configuration=review_path,receipt=review_receipt);update({})
                dispatched=True
            if len(verified)+len(state['failed_preserved'])==285:
                update(dict(state='all_original_models_verified' if len(verified)==285 else 'all_jobs_terminal_with_failures',
                    full_registered_CFM_queue_verified=len(verified)==285,active_pid=None));return
            if not dispatched:
                update(dict(state='waiting_actual_original_training_completion',active_pid=None,
                    pending_original_models=285-len(verified)-len(state['failed_preserved'])));time.sleep(10)
    finally:lane.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--review-controller',action='store_true')
    parser.add_argument('--review-worker',action='store_true');args=parser.parse_args();config=read(ROOT/args.config)
    if args.review_controller or args.review_worker:
        for pin in config['source_receipts']:
            if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Independent review source changed')
        (review_worker if args.review_worker else run_review)(config,args.config);return
    if args.probe_only:
        routes,_,records=verify(config)
        print(json.dumps(dict(registered_original_models=len(routes),preserved_verified_models=len(records),
            completed_original_results_present=sum((ROOT/r['result_path']).is_file() for r in routes),
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
            benchmark_performance=False)));return
    controller(config,args.config)


if __name__=='__main__':main()
