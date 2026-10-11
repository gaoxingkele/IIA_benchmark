"""Replay every checkpoint/generator/full MMD for all five original500epoch jobs.

Original native training and its successful guard receipt are prerequisites.
This independent read-only native replay adds no training or seed and retains
the exact original CPU environment, full evaluator arrays and bitwise gates.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, write, sha


def exclusive(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as output:
        json.dump(value,output,ensure_ascii=False,indent=2,allow_nan=False)


def check_routes(queue, config):
    jobs={job['id']:job for job in queue['jobs']}
    routes=config['routes']
    if (len(jobs)!=5 or len(routes)!=5 or len({r['canonical_job']['id'] for r in routes})!=5
            or {r['canonical_job']['id'] for r in routes}!=set(jobs)
            or {j['seed'] for j in queue['jobs']}!=set(range(1103,1108))
            or any(config[key]!=queue[key] for key in ('python','settings','resource_lock'))
            or config['required_epochs_per_job']!=500 or config['required_generated_points_per_epoch']!=4984
            or config['required_generator_calls_per_epoch']!=178 or config['full_fit_points']!=996
            or config['full_MMD_points']!=3988 or config['equivalence_gate']!='bitwise'
            or config['cpu_threads']!=2):
        raise ValueError('All five original500epoch/4984generator/996fit/3988MMD obligations and unchanged guards required')
    for route in routes:
        canonical=route['canonical_job'];job=route['execution_job']
        if canonical!=jobs[canonical['id']] or job['seed']!=canonical['seed']:
            raise ValueError('Canonical original full seed identity must remain unchanged')
        if route['kind']=='original':
            if job!=canonical or route.get('recovery_configuration') is not None:
                raise ValueError('Original seed must use its exact registered native result')
        elif route['kind']=='resume':
            recovery=read(ROOT/route['recovery_configuration'])
            if recovery['canonical_job']!=canonical or recovery['recovery_job']!=job:
                raise ValueError('Recovery must use the exact immutable canonical/actual execution mapping')
        else:raise ValueError('Only exact original or proven same-seed resume routes allowed')
        if route['result_path']!=(Path(job['output_directory'])/'result.json').as_posix():
            raise ValueError('Only the explicitly frozen actual execution result path allowed')
    return routes


def verify(config):
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen full500 replay source changed: '+pin['path'])
    queue=read(ROOT/config['original_queue'])
    from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_registration
    verify_registration(queue)
    if config['original_scope']!=dict(papers=45,method_baseline_ablation_records=681,original_data_task_records=273):
        raise ValueError('Overall original research scope remains required')
    return queue,check_routes(queue,config)


def complete_metadata(route, result, receipt):
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get(route['successful_native_receipt_field']) is not True
            or result.get('job')!=route['execution_job'] or result.get('full_training_completed') is not True
            or result.get('full_original_evaluators_completed') is not True
            or result.get('independent_training_seed_slots')!=1
            or result.get('author_equivalence_certified') is not False or result.get('strict_TAB_result') is not False
            or len(result.get('epochs',[]))!=500 or len(result.get('evaluators',[]))!=500
            or len(result.get('checkpoints',[]))!=500
            or any(result.get('optimizer_updates',{}).get(key)!=642000 for key in ('D','G'))):
        raise ValueError('Actual successful guarded complete500epoch native result required')
    if route['kind']=='resume':
        recovery=read(ROOT/route['recovery_configuration'])
        if (result.get('original_canonical_job')!=route['canonical_job']
                or result.get('full_original500_epoch_budget_completed_with_verified_resume') is not True
                or result.get('full_original500_epoch_loop_completed') is not False
                or result.get('verified_prior_full_epochs')!=recovery['verified_completed_epochs']
                or result.get('resume_segment_full_epochs')!=recovery['remaining_full_epochs']):
            raise ValueError('Exact proven prior plus remaining full native epochs required')
    elif result.get('full_original500_epoch_loop_completed') is not True:
        raise ValueError('Original full500epoch source loop must actually complete')
    sigma=0
    for epoch,(loss,evaluation,checkpoint) in enumerate(zip(result['epochs'],result['evaluators'],result['checkpoints'])):
        sigma+=evaluation['sigma_optimizer_updates']
        if (loss['epoch']!=epoch or evaluation['epoch']!=epoch or checkpoint['epoch']!=epoch
                or evaluation['complete_generated_points']!=4984 or evaluation['full_fit_points']!=996
                or evaluation['full_test_points']!=3988 or not 1<=evaluation['sigma_optimizer_updates']<=2000
                or (evaluation['sigma_optimizer_updates']<2000 and evaluation['stopping_delta']>.001)
                or any(checkpoint['optimizer_updates'][key]!=(epoch+1)*1284 for key in ('D','G'))
                or checkpoint['optimizer_updates']['sigma']!=sigma
                or not all(math.isfinite(evaluation[key]) for key in ('native_mmd2','native_ratio'))):
            raise ValueError('Every complete original evaluator/checkpoint/update/stopping ledger required')
    if result['optimizer_updates']['sigma']!=sigma:
        raise ValueError('All500 sigma update counts must sum exactly')


def checkpoint_directory(record):
    candidates=[Path(item['path']).parent for item in record['artifacts'] if Path(item['path']).name=='model.meta']
    if len(candidates)!=1:raise ValueError('Exactly one complete native checkpoint meta graph required per epoch')
    directory=candidates[0]
    if any(Path(item['path']).parent!=directory for item in record['artifacts']):
        raise ValueError('Checkpoint artifacts must share their original complete native directory')
    names={Path(item['path']).name for item in record['artifacts']}
    if not {'model.meta','model.index','numpy_random_state.npz','python_random_state.json'}.issubset(names):
        raise ValueError('Every full native graph/weight/index/Python/NumPy state required')
    if not any(name.startswith('model.data-') for name in names):raise ValueError('Actual full native model/optimizer tensor data required')
    return directory


def native_result(route, queue):
    if route['kind']=='original':
        from scripts.flow_matching.run_rcgan_native_original_queue_v2 import verify_result
        return verify_result(queue,route['execution_job'])
    recovery=read(ROOT/route['recovery_configuration'])
    if route['canonical_job']['seed']==1103:
        from scripts.flow_matching.rcgan_native_training_resume_runtime_v1 import verify_result
    elif route['canonical_job']['seed']==1104:
        from scripts.flow_matching.rcgan_native_training_resume_runtime_v2 import verify_result
    else:raise ValueError('Only the two explicit independently verified full resume mappings permitted')
    return verify_result(recovery,queue)


def bitwise_float32(left,right,np):
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise ValueError('Full native MMD values must remain finite')
    if np.asarray(left,dtype=np.float32).tobytes()!=np.asarray(right,dtype=np.float32).tobytes():
        raise ValueError('Complete original native MMD/ratio must replay bitwise; no relaxed tolerance')


def check_report(route, binding, report, receipt):
    if (receipt.get('exit_code')!=0 or receipt.get('resource_guard_aborted') is not False
            or receipt.get('complete_full500_native_replay_passed') is not True
            or report.get('passed') is not True or report.get('canonical_job')!=route['canonical_job']
            or report.get('execution_job')!=route['execution_job']
            or report.get('binding_sha256')!=binding['binding_sha256']
            or report.get('result_sha256')!=binding['result_sha256']
            or report.get('complete_replayed_epochs')!=500 or len(report.get('epochs',[]))!=500
            or report.get('full_generator_sample_count')!=2492000
            or report.get('full_native_generator_calls')!=89000
            or report.get('complete_original_MMD_calls')!=500
            or report.get('additional_training_seed_slots')!=0
            or report.get('training_executed') is not False):
        raise ValueError('Actual guarded complete500 checkpoint/generator/MMD replay required')
    for epoch,row in enumerate(report['epochs']):
        if (row['epoch']!=epoch or row['complete_generated_points']!=4984 or row['full_MMD_points']!=3988
                or any(row.get(key) is not True for key in ('all_native_model_and_optimizer_variables_restored_bitwise',
                    'all4984_generated_samples_replayed_bitwise','complete_original_MMD_and_ratio_replayed_bitwise',
                    'all_real_evaluator_images_belong_to_original_validation_group'))):
            raise ValueError('Every original full checkpoint/generator/metric/group must verify')


def worker(config, route, binding_path):
    queue,_=verify(config)
    binding=read(ROOT/binding_path)
    if (binding['runtime_configuration_sha256']!=sha(ROOT/config['runtime_configuration'])
            or binding['route']!=route or sha(ROOT/route['result_path'])!=binding['result_sha256']
            or sha(ROOT/route['native_receipt'])!=binding['native_receipt_sha256']):
        raise ValueError('Frozen exact actual native completion binding required')
    result=native_result(route,queue)
    complete_metadata(route,result,read(ROOT/route['native_receipt']))
    import numpy as np
    import tensorflow as tf
    import pkg_resources
    if (Path(sys.executable).resolve()!=Path(queue['python']).resolve() or sys.version!=queue['frozen_python_version']
            or tf.__version__!='1.15.0' or np.__version__!='1.18.5'):
        raise ValueError('Exact original native Python/TF/NumPy numerical environment required')
    package_root=(Path(sys.prefix)/'Lib/site-packages').resolve()
    if {p.key:p.version for p in pkg_resources.working_set if Path(p.location).resolve()==package_root}!=queue['frozen_native_packages']:
        raise ValueError('Complete original frozen native package environment required')
    for pin in queue['native_interpreter_receipts']:
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Original native interpreter changed')
    registry=read(ROOT/queue['source_registry']);asset=next(a for a in registry['assets'] if a['id']=='rcgan')
    from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
    source_before=verify_source(asset);sys.path.insert(0,asset['original_directory'])
    import mmd
    if Path(mmd.__file__).resolve()!=Path(asset['original_directory'],'mmd.py').resolve():
        raise ValueError('Unmodified actual original native MMD source required')
    data=read(ROOT/queue['dataset_configuration'])
    if sha(data['derived_training_array'])!=queue['derived_training_sha256']:raise ValueError('Full original raw-derived MNIST table changed')
    table=np.load(data['derived_training_array'],mmap_mode='r',allow_pickle=False)
    loader=result['full_native_loader_audit']
    if (table.shape!=(60000,785) or not loader['grouped_disjoint_image_indices']
            or not loader['every_original_split_pixel_and_label_bitwise_verified']
            or sha(loader['full_split_indices_path'])!=loader['full_split_indices_sha256']):
        raise ValueError('Full original60000 source images and disjoint original splits required')
    permutation=np.random.RandomState(route['canonical_job']['seed']).permutation(60000)
    with np.load(loader['full_split_indices_path'],allow_pickle=False) as splits:
        for tag,left,right in [('train',0,36000),('vali',36000,48000),('test',48000,60000)]:
            if not np.array_equal(splits[tag],permutation[left:right]):raise ValueError('Every original grouped image split identity must match canonical seed')
        validation_ids=splits['vali'].copy()
    validation_pixels={hashlib.sha256(np.float32(table[index,1:]).tobytes()).digest() for index in validation_ids}
    output=ROOT/config['output_root']/route['canonical_job']['id']
    if output.exists():raise FileExistsError('Preserve every complete or failed full500 replay')
    output.mkdir(parents=True)
    first=checkpoint_directory(result['checkpoints'][0]);graph=tf.Graph()
    with graph.as_default():
        saver=tf.train.import_meta_graph(str(first/'model.meta'),clear_devices=True)
        variables=tf.global_variables();trainable=tf.trainable_variables()
        parameters=sum(int(np.prod(v.shape.as_list())) for v in trainable if v.name.startswith(('generator/','discriminator/')))
        if parameters!=96929 or {v.name for v in trainable if not v.name.startswith(('generator/','discriminator/'))}!={'sigma:0'}:
            raise ValueError('All original96929 GAN parameters and native sigma required')
        candidates=[tensor for op in graph.get_operations() for tensor in op.outputs
            if op.name.startswith('generator/') and op.type=='Reshape' and tensor.shape.as_list()==[28,28,28]]
        noises=[tensor for op in graph.get_operations() if op.type=='Placeholder' for tensor in op.outputs if tensor.shape.as_list()==[28,28,5]]
        if len(candidates)!=1 or len(noises)!=1:raise ValueError('Unique complete original generator/noise tensors required')
        a=tf.placeholder(tf.float32,[3988,28,28],name='independent_full_MMD_real')
        b=tf.placeholder(tf.float32,[3988,28,28],name='independent_full_MMD_generated')
        sigma=graph.get_tensor_by_name('sigma:0')
        values=mmd.mix_rbf_mmd2_and_ratio(a,b,sigmas=sigma,biased=False)
    schema={v.op.name:v.shape.as_list() for v in variables}
    if not any('/Adam' in name for name in schema) or not any('/RMSProp' in name for name in schema):
        raise ValueError('All full original generator/critic/sigma optimizer states required')
    rows=[]
    session_config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)
    with tf.Session(graph=graph,config=session_config) as session:
        for epoch,(evaluation,checkpoint) in enumerate(zip(result['evaluators'],result['checkpoints'])):
            directory=checkpoint_directory(checkpoint);checkpoint_path=str(directory/'model')
            reader=tf.train.load_checkpoint(checkpoint_path)
            if reader.get_variable_to_shape_map()!=schema:raise ValueError('Every complete native checkpoint must retain all model/optimizer variables')
            saved={name:reader.get_tensor(name) for name in schema}
            if not all(np.isfinite(value).all() for value in saved.values()):raise ValueError('Every checkpoint value must remain finite')
            saver.restore(session,checkpoint_path)
            restored=session.run(variables)
            if any(not np.array_equal(value,saved[var.op.name]) for var,value in zip(variables,restored)):
                raise ValueError('Every complete native model and optimizer variable must restore bitwise')
            with np.load(evaluation['array_path'],allow_pickle=False) as archive:
                real,generated,noise,saved_sigma=[archive[key] for key in ('real','generated','noise','sigma')]
            if (real.shape!=(4984,28,28) or generated.shape!=real.shape or noise.shape!=(4984,28,5)
                    or not all(np.isfinite(v).all() for v in (real,generated,noise,saved_sigma))
                    or not np.array_equal(session.run(sigma),saved_sigma)):
                raise ValueError('All full original real/generated/noise arrays and optimized sigma required')
            if any(hashlib.sha256(np.float32(image).tobytes()).digest() not in validation_pixels for image in real):
                raise ValueError('Every original evaluator real image must belong to its original validation group')
            replay=np.empty_like(generated)
            for start in range(0,4984,28):
                replay[start:start+28]=session.run(candidates[0],feed_dict={noises[0]:noise[start:start+28]})
            if not np.array_equal(replay,generated):raise ValueError('Every one of4984 original generated samples must replay bitwise')
            calculated=session.run(values,feed_dict={a:real[996:],b:replay[996:]})
            bitwise_float32(calculated[0],evaluation['native_mmd2'],np)
            bitwise_float32(calculated[1],evaluation['native_ratio'],np)
            rows.append(dict(epoch=epoch,checkpoint_directory=str(directory),evaluation_array_sha256=evaluation['array_sha256'],
                complete_generated_points=4984,full_MMD_points=3988,original_generator_calls=178,
                full_native_variable_count=len(variables),all_native_model_and_optimizer_variables_restored_bitwise=True,
                all4984_generated_samples_replayed_bitwise=True,complete_original_MMD_and_ratio_replayed_bitwise=True,
                all_real_evaluator_images_belong_to_original_validation_group=True,
                replayed_generated_sha256=hashlib.sha256(replay.tobytes()).hexdigest(),
                replayed_MMD2=float(calculated[0]),replayed_ratio=float(calculated[1])))
            write(output/'progress.json',dict(complete_replayed_epochs=len(rows),required_epochs=500,
                required_original_canonical_jobs=5,training_executed=False,all_original_experiments_complete=False))
            del saved,restored,replay,real,generated,noise,saved_sigma,reader
            gc.collect()
    if source_before!=verify_source(asset):raise ValueError('Original native source changed during full replay')
    if sha(ROOT/route['result_path'])!=binding['result_sha256'] or sha(ROOT/route['native_receipt'])!=binding['native_receipt_sha256']:
        raise ValueError('Original actual completed result/guard receipt changed during full replay')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        canonical_job=route['canonical_job'],execution_job=route['execution_job'],binding_sha256=sha(ROOT/binding_path),
        result_sha256=binding['result_sha256'],complete_replayed_epochs=len(rows),epochs=rows,
        full_generator_sample_count=2492000,full_native_generator_calls=89000,complete_original_MMD_calls=500,
        complete_original_source_splits_independently_verified=True,full_native_GAN_parameters=96929,
        original_evaluator_real_image_source='original12000image_validation_group_sampled_with_replacement',
        optimized_sigma_replayed=True,sigma_optimizer_refitted=False,training_executed=False,
        additional_training_seed_slots=0,author_equivalence_certified=False,strict_TAB_result=False,
        all_original_experiments_complete=False)
    write(output/'complete_full500_native_replay.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--canonical-id');parser.add_argument('--binding')
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    config=read(ROOT/args.config);queue,routes=verify(config)
    if args.probe_only:
        print(json.dumps(dict(required_original_jobs=5,full_epochs_per_job=500,total_checkpoints_required=2500,
            total_generator_samples_required=12460000,total_native_generator_calls_required=445000,
            full_MMD_replays_required=2500,numpy_imported='numpy' in sys.modules,tensorflow_imported='tensorflow' in sys.modules,
            all_original_experiments_complete=False)));return
    if args.worker:
        if not args.binding or not args.canonical_id:raise ValueError('Exact frozen completed case binding required')
        worker(config,next(r for r in routes if r['canonical_job']['id']==args.canonical_id),args.binding);return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full original five-seed completion/replay already has an owner')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,
        required_original_models=5,required_epochs_each=500,verified_original_ids=[],pending_original_models=5,
        failed={},configuration_sha256=sha(ROOT/args.config),all_original_experiments_complete=False)
    def update(values):
        state.update(values,captured_utc=datetime.now(timezone.utc).isoformat());write(base/'status.json',state)
    try:
        while len(state['verified_original_ids'])<5:
            for route in routes:
                identity=route['canonical_job']['id']
                if identity in state['verified_original_ids'] or identity in state['failed']:continue
                if not (ROOT/route['result_path']).exists() or not (ROOT/route['native_receipt']).exists():continue
                try:
                    complete_metadata(route,read(ROOT/route['result_path']),read(ROOT/route['native_receipt']))
                    binding=dict(runtime_configuration_sha256=sha(ROOT/args.config),route=route,
                        result_sha256=sha(ROOT/route['result_path']),native_receipt_sha256=sha(ROOT/route['native_receipt']))
                    binding_path=base/'bindings'/f'{identity}.json';exclusive(binding_path,binding)
                    command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v1',
                        '--config',args.config,'--worker','--canonical-id',identity,'--binding',str(binding_path)]
                    with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
                        logs=base/'logs';logs.mkdir(exist_ok=True)
                        with (logs/(identity+'.stdout.log')).open('xb') as out,(logs/(identity+'.stderr.log')).open('xb') as err:
                            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                                    PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            update(dict(state='running_full500_original_native_replay',active_pid=child.pid,
                                active_create_time=psutil.Process(child.pid).create_time(),active_job=identity,actual_command=command))
                            code,usage=capped_wait(child,config,api,update)
                    report_path=ROOT/config['output_root']/identity/'complete_full500_native_replay.json'
                    passed=code==0 and usage['resource_guard_aborted'] is False and report_path.exists()
                    receipt=dict(exit_code=code,complete_full500_native_replay_passed=passed,**usage)
                    write(base/'receipts'/f'{identity}.json',receipt)
                    if not passed:raise ValueError('Actual complete native full500 replay failed; preserve its full attempt')
                    binding['binding_sha256']=sha(binding_path)
                    check_report(route,binding,read(report_path),receipt)
                    state['verified_original_ids'].append(identity)
                except Exception as error:
                    state['failed'][identity]=str(error)
                    exclusive(base/'failures'/f'{identity}.json',dict(error=str(error),route=route,all_original_experiments_complete=False))
                update(dict(active_pid=None,active_job=None,pending_original_models=5-len(state['verified_original_ids'])))
            actual_owners=[]
            for expected in config['expected_original_native_owners']:
                try:
                    process=psutil.Process(expected['pid'])
                    if process.create_time()==expected['create_time'] and process.cmdline()==expected['command']:
                        actual_owners.append(dict(expected,live_now=True))
                except psutil.NoSuchProcess:pass
            update(dict(state='waiting_actual_complete_original_full500_native_results',actual_verified_live_original_owners=actual_owners))
            if len(state['verified_original_ids'])<5:time.sleep(config['poll_seconds'])
        update(dict(state='all_five_complete_full500_native_results_replayed',all_five_original_RCGAN_full500_replays_complete=True))
    finally:owner.close()


if __name__=='__main__':main()
