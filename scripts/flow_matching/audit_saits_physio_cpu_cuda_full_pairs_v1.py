"""Compare all ten saved full CPU/CUDA predictions; preserve original strict failures."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from scripts.flow_matching.audit_saits_full_checkpoint_replay_v1 import ROOT,read,sha,resource_api,capped_wait,write

IDS=[f'saits_paper_{method}_physio_seed{seed}' for seed in range(26,31) for method in ['saits','brits']]
SCOPE=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273)
KEYS={'predictions','truth','mask','test_indices','unit_ids','group_ids','group_statistics'}


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen complete CPU/CUDA comparison source changed')
    guard=read(ROOT/config['readonly_full_array_guard_template'])
    if (any(config[k]!=guard[k] for k in ['python','settings','resource_lock'])
            or config['original_scope']!=SCOPE or config['required_full_model_pairs']!=10
            or config['full_test_shape']!=[2398,48,37] or config['required_held_out_coordinates_each']!=82605
            or config['additional_training_seed_slots']!=0 or [r['id'] for r in config['pairs']]!=IDS):
        raise ValueError('All ten full patient-model pairs and unchanged existing full-array guards required')
    cpu=read(ROOT/config['cpu_report']);cuda=read(ROOT/config['cuda_report'])
    if (cpu['full_models_replayed']!=10 or cuda['full_models_replayed']!=10
            or not cpu['all_registered_native_models_reinferred'] or not cuda['all_registered_native_models_reinferred']
            or cpu['cuda_context_initialized'] is not False or cuda['cuda_context_initialized'] is not True
            or cuda['replay_inference_device']!='cuda'
            or config['original_metric_tolerance']!=cpu['metric_tolerance']
            or config['original_metric_tolerance']!=cuda['metric_tolerance']
            or cpu['full_checkpoint_contents_and_re_inference_passed'] is not False
            or cuda['full_checkpoint_contents_and_re_inference_passed'] is not False):
        raise ValueError('Both actual complete original failed strict replays must remain explicit')
    for side in ['cpu','cuda']:
        receipt=read(ROOT/config[side+'_receipt'])
        if receipt['exit_code']!=0 or receipt['resource_guard_aborted'] or receipt['full_native_model_replay_passed']:
            raise ValueError('Actual complete numerical execution without resource abort and preserved strict failure required')
    if [r['id'] for r in cpu['records']]!=IDS or [r['id'] for r in cuda['records']]!=IDS:
        raise ValueError('Every full CPU/CUDA record required in the original order')
    for pair,a,b in zip(config['pairs'],cpu['records'],cuda['records']):
        for field in ['id','seed','config','config_sha256','result_sha256','checkpoint_sha256',
                'full_test_shape','full_native_batch_size','full_test_batches','held_out_coordinates','original_full_dataset_sha256']:
            if a[field]!=b[field]:raise ValueError('The two sides must share the entire original model/data/batch identity')
        if (a['full_test_shape']!=config['full_test_shape'] or a['full_native_batch_size']!=128
                or a['full_test_batches']!=19 or a['held_out_coordinates']!=82605
                or sha(ROOT/pair['cpu_arrays'])!=a['replay_arrays_sha256']
                or sha(ROOT/pair['cuda_arrays'])!=b['replay_arrays_sha256']
                or sha(ROOT/a['config'])!=a['config_sha256']
                or sha(ROOT/a['result_path'])!=a['result_sha256']
                or sha(ROOT/a['checkpoint_path'])!=a['checkpoint_sha256']
                or sha(ROOT/pair['original_statistics'])!=a['full_training_completion']['statistics_sha256']):
            raise ValueError('Every actual full recorded prediction artifact required')
    return cpu,cuda


def group_statistics(prediction,truth,mask,unit_ids,np):
    difference=(prediction.astype(np.float64)-truth.astype(np.float64))*mask
    stats=np.stack([np.abs(difference).sum(axis=(1,2)),np.square(difference).sum(axis=(1,2)),
        mask.sum(axis=(1,2),dtype=np.float64),np.abs(truth.astype(np.float64)*mask).sum(axis=(1,2))],axis=1)
    ids,inverse=np.unique(unit_ids,return_inverse=True)
    grouped=np.zeros((len(ids),4),dtype=np.float64);np.add.at(grouped,inverse,stats)
    return ids,grouped


def compare_pair(cpu,cuda,original,tolerance,expected_shape,expected_held_out):
    import numpy as np
    if set(cpu)!=KEYS or set(cuda)!=KEYS:raise ValueError('All seven complete native prediction/identity/statistic arrays required')
    for side in [cpu,cuda]:
        if (list(side['predictions'].shape)!=list(expected_shape) or side['predictions'].shape!=side['truth'].shape
                or side['mask'].shape!=side['truth'].shape or not np.isfinite(side['predictions']).all()
                or not np.isfinite(side['truth']).all() or not np.isin(side['mask'],[0,1]).all()
                or int(side['mask'].sum())!=expected_held_out
                or not np.array_equal(np.sort(side['test_indices']),np.arange(expected_shape[0]))
                or side['unit_ids'].shape!=(expected_shape[0],) or len(np.unique(side['unit_ids']))!=expected_shape[0]):
            raise ValueError('Every complete original finite patient grid, held-out mask and identity required')
    for key in ['truth','mask','test_indices','unit_ids','group_ids']:
        if not np.array_equal(cpu[key],cuda[key]):raise ValueError('Paired original data/mask/identity/order differs: '+key)
    if (not np.array_equal(original['group_ids'],cpu['group_ids'])
            or not np.array_equal(np.sort(original['unit_ids']),np.sort(cpu['unit_ids']))
            or original['statistics'].shape!=(expected_shape[0],4) or not np.isfinite(original['statistics']).all()):
        raise ValueError('Complete original patient statistics and identities required')
    recomputed={}
    for name,side in [('cpu',cpu),('cuda',cuda)]:
        ids,stats=group_statistics(side['predictions'],side['truth'],side['mask'],side['unit_ids'],np)
        if not np.array_equal(ids,side['group_ids']) or not np.allclose(stats,side['group_statistics'],**tolerance):
            raise ValueError('Independent full prediction-derived patient statistics differ from their own replay archive')
        recomputed[name]=stats
    delta=cuda['predictions'].astype(np.float64)-cpu['predictions'].astype(np.float64)
    held=cpu['mask'].astype(bool);absolute=np.abs(delta)
    original_values=original['statistics'];reviews={}
    for name,side in [('cpu',cpu),('cuda',cuda)]:
        values=side['group_statistics'];fail=~np.isclose(values,original_values,**tolerance)
        reviews[name]=dict(strict_original_statistics_match=bool(not fail.any()),failed_cells=int(fail.sum()),
            failed_patients=int(fail.any(axis=1).sum()),failures=[dict(patient_id=str(cpu['group_ids'][r]),column=int(c),
                replay=float(values[r,c]),original=float(original_values[r,c]),
                difference=float(values[r,c]-original_values[r,c]),
                original_allowed_absolute_difference=float(tolerance['atol']+tolerance['rtol']*abs(original_values[r,c])))
                for r,c in np.argwhere(fail)])
    totals={name:values.sum(axis=0) for name,values in recomputed.items()}
    metrics={name:dict(mae=float(v[0]/v[2]),rmse=float(np.sqrt(v[1]/v[2])),mre=float(v[0]/v[3])) for name,v in totals.items()}
    different=cpu['predictions']!=cuda['predictions']
    full_summary=dict(full_shape=list(cpu['predictions'].shape),full_prediction_coordinates=int(delta.size),
        held_out_coordinates=int(held.sum()),full_prediction_coordinates_different=int(different.sum()),
        held_out_prediction_coordinates_different=int(different[held].sum()),
        prediction_arrays_bitwise_equal=bool(cpu['predictions'].dtype==cuda['predictions'].dtype
            and cpu['predictions'].tobytes()==cuda['predictions'].tobytes()),
        maximum_prediction_difference_all_coordinates=float(absolute.max()),
        maximum_prediction_difference_held_out=float(absolute[held].max()),
        mean_absolute_prediction_difference_held_out=float(absolute[held].mean()),
        paired_prediction_rmse_held_out=float(np.sqrt(np.square(delta[held]).mean())),
        all_original_data_mask_indices_and_patient_identities_exactly_equal=True,
        own_full_patient_statistics_independently_recomputed=True,
        cpu_cuda_independently_recomputed_metrics=metrics,original_strict_statistics_reviews=reviews,
        original_tolerance_unchanged=tolerance,original_training_predictions_persisted=False,
        original_numerical_mismatch_cause_proven=False,full_native_reproduction_certified=False)
    artifacts=dict(group_ids=cpu['group_ids'],unit_ids=cpu['unit_ids'],original_statistics=original_values,
        cpu_statistics=cpu['group_statistics'],cuda_statistics=cuda['group_statistics'],
        cpu_independent_statistics=recomputed['cpu'],cuda_independent_statistics=recomputed['cuda'],
        cuda_minus_cpu_statistics=cuda['group_statistics']-cpu['group_statistics'],
        different_prediction_count_per_unit=different.sum(axis=(1,2)),
        different_held_out_prediction_count_per_unit=(different&held).sum(axis=(1,2)),
        maximum_absolute_prediction_difference_per_unit=absolute.max(axis=(1,2)))
    return full_summary,artifacts


def worker(config,path):
    import numpy as np
    cpu,cuda=verify(config);target=ROOT/config['artifact_directory'];target.mkdir(parents=True,exist_ok=False)
    records=[]
    for pair,a,b in zip(config['pairs'],cpu['records'],cuda['records']):
        with np.load(ROOT/pair['cpu_arrays'],allow_pickle=False) as x,np.load(ROOT/pair['cuda_arrays'],allow_pickle=False) as y,np.load(ROOT/pair['original_statistics'],allow_pickle=False) as z:
            summary,arrays=compare_pair({k:x[k] for k in x.files},{k:y[k] for k in y.files},{k:z[k] for k in z.files},
                config['original_metric_tolerance'],config['full_test_shape'],config['required_held_out_coordinates_each'])
        for name,source in [('cpu',a),('cuda',b)]:
            if summary['original_strict_statistics_reviews'][name]['strict_original_statistics_match']!=source['original_patient_statistics_match']:
                raise ValueError('Independent original strict-failure classification must agree with actual full replay')
        artifact=target/(pair['id']+'_complete_paired_patient_statistics.npz')
        with artifact.open('xb') as stream:np.savez_compressed(stream,**arrays)
        records.append(dict(id=pair['id'],cpu_original_checkpoint_sha256=a['checkpoint_sha256'],
            cuda_original_checkpoint_sha256=b['checkpoint_sha256'],cpu_arrays_sha256=sha(ROOT/pair['cpu_arrays']),
            cuda_arrays_sha256=sha(ROOT/pair['cuda_arrays']),original_statistics_sha256=sha(ROOT/pair['original_statistics']),
            complete_pair_statistics_artifact=str(artifact),complete_pair_statistics_sha256=sha(artifact),**summary))
        del arrays
    verify(config)
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=path,configuration_sha256=sha(ROOT/path),
        full_native_prediction_pairs_independently_compared=10,all_complete_pair_data_and_checkpoint_identities_verified=True,
        all_original_strict_failure_classifications_preserved=True,records=records,numpy_comparison_version=np.__version__,
        numerical_model_inference_executed=False,numerical_optimizer_updates=0,additional_training_seed_slots=0,
        original_numerical_mismatch_cause_proven=False,full_native_reproduction_certified=False,
        author_equivalence_certified=False,strict_TAB_result=False,all_original_experiments_complete=False,
        boundary='Full saved CPU/CUDA arrays of all ten original checkpoints are independently compared on every coordinate. Data, masks, indices and patient identities must agree exactly, and each side own patient statistics are recomputed. Original strict mismatches remain failures. The original training-time prediction tensors and backend flags were not persisted, so paired differences do not establish the causal explanation of the original discrepancy. No new model inference or training, paper or TAB claim.')
    write(target/'complete_cpu_cuda_prediction_pair_comparison.json',report)
    print(json.dumps(dict(full_prediction_pairs_compared=10,full_native_reproduction_certified=False,
        original_numerical_mismatch_cause_proven=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        print(json.dumps(dict(full_model_pairs=10,full_test_shape=config['full_test_shape'],
            torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,numerical_model_inference_executed=False)));return
    if args.worker:return worker(config,args.config)
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=False)
    api=resource_api();owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Complete comparison already has an actual owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting')
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_saits_physio_cpu_cuda_full_pairs_v1','--config',args.config,'--worker']
            with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            p=psutil.Process(child.pid);update(dict(state='comparing_all_ten_complete_saved_prediction_pairs',active_pid=p.pid,
                active_create_time=p.create_time(),actual_command=p.cmdline()))
            code,usage=capped_wait(child,config,api,update)
        target=ROOT/config['artifact_directory']/'complete_cpu_cuda_prediction_pair_comparison.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and target.exists()
            and read(target)['full_native_prediction_pairs_independently_compared']==10)
        write(base/'receipt.json',dict(exit_code=code,complete_saved_prediction_pair_comparison_passed=passed,
            full_native_reproduction_certified=False,**usage))
        update(dict(state='complete_saved_array_comparison' if passed else 'failed_preserved',active_pid=None))
        if not passed:raise RuntimeError('Comparison failed; preserve the failed attempt and all originals')
    finally:owner.close()


if __name__=='__main__':main()
