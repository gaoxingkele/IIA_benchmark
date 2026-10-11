"""Independently verify every coordinate of a registered complete SAITS/BRITS batch."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import ROOT,read,sha
from scripts.flow_matching.register_grasp_tau0_seed1104_replay_v1 import exclusive_json
from scripts.flow_matching.verify_brits_electricity_full_replay_v3 import check_arrays


def check_terminal(replay,receipt,producer):
    wanted=[r['id'] for r in producer['completed_jobs']]
    actual=[r['id'] for r in replay['records']]
    if (len(wanted)!=producer['full_model_count'] or len(wanted)!=len(set(wanted))
            or len(actual)!=len(set(actual)) or set(actual)!=set(wanted)
            or replay['full_models_replayed']!=producer['full_model_count']
            or len(actual)!=producer['full_model_count']
            or not replay['full_checkpoint_contents_and_re_inference_passed']
            or replay['cuda_context_initialized'] or replay['additional_training_seed_slots']!=0
            or receipt['exit_code']!=0 or receipt['resource_guard_aborted']
            or not receipt['full_native_model_replay_passed']):
        raise ValueError('Exactly all registered full native CPU model replays must finish without abort')


def verify_record_identity(record,native,result,registered):
    if (native['id']!=registered['id'] or record['id']!=registered['id']
            or record['config']!=registered['config'] or result['config']!=native
            or record['seed']!=native['seed'] or native['bootstrap_group']!='month'
            or native.get('diagnostic') or native.get('limit_test_cases')
            or result['status']!='completed' or record['model_parameters']!=result['parameters']
            or record['held_out_coordinates']!=result['target_count']
            or record['original_full_dataset_sha256']!=native['dataset_sha256']
            or record['validation_best']!=result['validation_best']):
        raise ValueError('Every full original method/data/seed/result/selected-checkpoint identity must match')


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen full batch review source changed')
    producer=read(ROOT/config['replay_configuration'])
    from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import verify_registration as check_producer
    check_producer(producer)
    if (producer['settings']!=config['settings'] or producer['resource_lock']!=config['resource_lock']
            or producer['python']!=config['python'] or config['required_full_models']!=producer['full_model_count']):
        raise ValueError('Retain original full native replay model count, environment and resource gates')
    return producer


def verify(config,path,producer):
    import h5py
    import numpy as np
    from scripts.flow_matching.validate_saits_full_result_v1 import validate
    report_path=ROOT/producer['artifact_directory']/'checkpoint_replay_audit.json'
    receipt_path=ROOT/producer['state_root']/'receipt.json'
    replay,receipt=read(report_path),read(receipt_path)
    check_terminal(replay,receipt,producer)
    if (replay['configuration_sha256']!=sha(ROOT/config['replay_configuration'])
            or replay['metric_tolerance']!=producer['metric_tolerance']):
        raise ValueError('Keep the frozen complete producer configuration and existing tolerances')
    by_id={item['id']:item for item in replay['records']};records=[]
    for registered in producer['completed_jobs']:
        record=by_id[registered['id']]
        native_path=ROOT/registered['config'];native=read(native_path)
        result_path=ROOT/registered['result_path'];result=read(result_path)
        verify_record_identity(record,native,result,registered)
        output=ROOT/native['output_root'];data=ROOT/native['dataset_root']
        if result_path.resolve()!=(output/'result.json').resolve():raise ValueError('Keep each actual original result path')
        arrays_path=Path(record['replay_arrays'])
        hashes=dict(config=sha(native_path),result=sha(result_path),checkpoint=sha(output/'best.pt'),
            arrays=sha(arrays_path),dataset=sha(data/'datasets.h5'),resume=sha(output/'resume.pt'),
            original_group_statistics=sha(output/'group_statistics.npz'),data_audit=sha(data/'data_audit.json'))
        for field in ['config','result','checkpoint','arrays','dataset']:
            key='original_full_dataset_sha256' if field=='dataset' else ('replay_arrays_sha256' if field=='arrays' else field+'_sha256')
            if hashes[field]!=record[key]:raise ValueError('Full native replay content hash differs: '+field)
        if hashes['result']!=registered['result_sha256']:raise ValueError('The frozen completed original result changed')
        completion=validate(native_path)
        if completion!=record['full_training_completion']:raise ValueError('Original full optimizer/RNG/early-stop completion changed')
        with h5py.File(data/'datasets.h5','r') as hf:
            truth,mask=hf['test']['X'][:],hf['test']['indicating_mask'][:]
        with np.load(data/'audit_arrays.npz',allow_pickle=False) as identities:
            ids,groups=identities['test_ids'],identities['test_groups']
        with np.load(arrays_path,allow_pickle=False) as arrays:
            actual,grouped=check_arrays(arrays,truth,mask,ids,groups,result['target_count'],producer['metric_tolerance'])
        for key,value in actual.items():
            for measured in [result['metrics'],record['replayed_metrics'],record['independently_recomputed_metrics']]:
                if not np.isclose(value,measured[key],**producer['metric_tolerance']):
                    raise ValueError('Independently recomputed full native metric differs: '+key)
        with np.load(output/'group_statistics.npz',allow_pickle=False) as original:
            if (not np.array_equal(original['group_ids'],np.unique(groups))
                    or not np.allclose(grouped,original['statistics'],**producer['metric_tolerance'])):
                raise ValueError('Every original grouped error statistic must agree')
        if (not completion['full_training_finished'] or completion['benchmark_smoke']
                or completion['full_test_units']!=truth.shape[0] or list(truth.shape)!=record['full_test_shape']
                or completion['held_out_coordinates']!=result['target_count']):
            raise ValueError('The complete original training/test-grid budget is required')
        records.append(dict(id=native['id'],method=native['method'],dataset=native['dataset'],
            missing_ratio=native['missing_ratio'],seed=native['seed'],hashes=hashes,
            full_test_shape=list(truth.shape),full_test_grid_cells=int(truth.size),
            held_out_coordinates=result['target_count'],full_truth_masks_units_groups_equal=True,
            full_finite_predictions=True,full_group_statistics_independently_recomputed=True,
            months=len(grouped),independently_recomputed_metrics=actual,original_metrics=result['metrics'],
            full_training_completion=completion,actual_full_checkpoint_reinference_completed=True))
    verify_registration(config)
    target=ROOT/config['output_directory'];target.mkdir(exist_ok=False,parents=True)
    exclusive_json(target/'independent_full_replay_batch_verification.json',dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=path,
        configuration_sha256=sha(ROOT/path),native_replay_report_sha256=sha(report_path),
        native_resource_receipt=receipt,full_native_models_verified=len(records),records=records,
        metric_tolerance=producer['metric_tolerance'],all_original_grid_coordinates_independently_checked=True,
        additional_model_inference_executed_by_this_review=False,additional_training_seed_slots=0,
        original_scope=config['original_scope'],author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False))
    exclusive_json(target/'native_checkpoint_replay_audit.json',replay)
    print(json.dumps(dict(full_native_models_independently_verified=len(records),all_original_coordinates_checked=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);producer=verify_registration(config)
    if args.probe_only:
        import sys
        print(dict(required_full_models=producer['full_model_count'],numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules));return
    if args.worker:verify(config,args.config,producer);return
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait,write
    from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(exist_ok=True,parents=True)
    lane=api['exclusive_lock'](base/'controller.lock')
    if lane is None:raise RuntimeError('This original full batch review controller is already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting')
    def update(values):state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        receipt_path=ROOT/producer['state_root']/'receipt.json'
        while not receipt_path.exists():
            handle=process_identity(config['producer_controller_pid'],config['producer_controller_create_time'],'audit_saits_full_checkpoint_replay_v1')
            if not handle['live'] or not handle['create_time_verified']:
                update(dict(state='missing_or_unverified_original_full_replay',actual_producer=handle));return
            update(dict(state='waiting_actual_full11_native_replay_receipt',actual_producer=handle));time.sleep(10)
        check_terminal(read(ROOT/producer['artifact_directory']/'checkpoint_replay_audit.json'),read(receipt_path),producer)
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.verify_saits_completed_full_replay_batch_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(producer['cpu_threads']),MKL_NUM_THREADS=str(producer['cpu_threads'])),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                handle=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=handle.create_time(),actual_command=handle.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        proof=ROOT/config['output_directory']/'independent_full_replay_batch_verification.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed'])
        write(base/'receipt.json',dict(exit_code=code,full_native_batch_independent_review_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,full_native_batch_independent_review_passed=passed))
    finally:lane.close()


if __name__=='__main__':main()
