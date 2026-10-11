"""Verify interrupted full native GAN checkpoints, losses and next update state.

Requires the independently matched238-epoch random/order chain. This restores
every real checkpoint, reproduces native loss observations (never elapsed time),
replays all latest generator evaluator samples and checks deterministic original
D/G transitions from restored optimizer state. It is recovery evidence, not a
completed500-epoch result or a new training seed.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,write,sha
from scripts.flow_matching.audit_rcgan_random_order_chain_v1 import verify_registration as verify_chain_registration


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen complete checkpoint recovery evidence changed')
    chain_config=read(ROOT/config['random_chain_configuration'])
    queue,job,original,evaluations=verify_chain_registration(chain_config)
    chain=read(ROOT/chain_config['output_directory']/'random_order_chain_audit.json')
    terminal=read(ROOT/chain_config['state_root']/'receipt.json')
    if (not chain['passed'] or not chain['complete_full_rng_and_cumulative_data_order_verified']
            or chain['original_completed_epochs']!=238 or chain['required_original_epochs']!=500
            or terminal['exit_code']!=0 or terminal['resource_guard_aborted']
            or not terminal['random_order_chain_passed'] or chain['job']!=job
            or sha(chain['derived_order_artifact']['path'])!=chain['derived_order_artifact']['sha256']):
        raise ValueError('Actual full238-epoch bitwise random/order audit must complete first')
    return queue,job,original,evaluations,chain


def reachable_ops(roots):
    pending=list(roots);seen=set()
    while pending:
        op=pending.pop()
        if op in seen:continue
        seen.add(op)
        pending.extend(t.op for t in op.inputs)
        pending.extend(op.control_inputs)
    return seen


def assert_no_random_transition(roots):
    operations=reachable_ops(roots)
    unexpected=[op.name for op in operations if 'Random' in op.type or op.type in ('Multinomial','TruncatedNormal')]
    if unexpected:
        raise ValueError('Training-reachable stateful random operations need separate exact-state recovery: '+str(unexpected))
    return len(operations)


def compare_loss(recomputed,original,np):
    # The original trace formats np.float32 with its shortest round-trip string.
    if np.float32(recomputed)!=np.float32(original):
        raise ValueError('Every persisted original full training loss must reproduce bitwise as float32')


def worker(config,config_path):
    queue,job,original,evaluations,chain=verify_registration(config)
    import numpy as np
    import tensorflow as tf
    if (Path(sys.executable).resolve()!=Path(queue['python']).resolve()
            or tf.__version__!='1.15.0' or np.__version__!='1.18.5'
            or sys.version!=queue['frozen_python_version']):
        raise ValueError('Exact native Python/TF/NumPy environment required')
    import pkg_resources
    packages_root=(Path(sys.prefix)/'Lib/site-packages').resolve()
    if {p.key:p.version for p in pkg_resources.working_set if Path(p.location).resolve()==packages_root}!=queue['frozen_native_packages']:
        raise ValueError('Complete original frozen package environment required')
    for item in queue['native_interpreter_receipts']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Native interpreter changed')
    registry=read(ROOT/queue['source_registry'])
    asset=next(a for a in registry['assets'] if a['id']=='rcgan')
    from scripts.flow_matching.audit_adopted_gan_native_v1 import verify_source
    source_before=verify_source(asset)
    sys.path.insert(0,asset['original_directory'])
    import model
    provided=read(Path(asset['original_directory'])/queue['source_settings'])
    settings=read(original/'experiments/settings'/(job['id']+'.txt'))
    if (settings['identifier']!=job['id'] or settings['dp']
            or any(settings[k]!=value for k,value in provided.items() if k!='identifier')):
        raise ValueError('Actual parsed author defaults must preserve every supplied original setting')
    # Obtain exact GAN fetch names and schemas from unchanged original functions.
    source_graph=tf.Graph()
    with source_graph.as_default():
        tf.set_random_seed(job['seed'])
        Z,X,CG,CD,CS=model.create_placeholders(28,28,5,28,0)
        generator_settings={k:settings[k] for k in ['hidden_units_g','seq_length','batch_size','num_generated_features','cond_dim','learn_scale']}
        discriminator_settings={k:settings[k] for k in ['hidden_units_d','seq_length','cond_dim','batch_size','batch_mean']}
        D_loss,G_loss=model.GAN_loss(Z,X,generator_settings,discriminator_settings,settings['kappa'],False,CG,CD,CS,wrong_labels=settings['wrong_labels'])
        D_solver,G_solver,_=model.GAN_solvers(D_loss,G_loss,settings['learning_rate'],28,total_examples=36000,
            l2norm_bound=settings['l2norm_bound'],batches_per_lot=settings['batches_per_lot'],sigma=settings['dp_sigma'],dp=settings['dp'])
        GAN_sample=model.generator(Z,**generator_settings,reuse=True,c=CG)
        gan_schema={v.name:v.shape.as_list() for v in tf.global_variables()}
        gan_trainable={v.name:v.shape.as_list() for v in tf.trainable_variables()}
    names=dict(Z=Z.name,X=X.name,D_loss=D_loss.name,G_loss=G_loss.name,D_solver=D_solver.name,
               G_solver=G_solver.name,G_sample=GAN_sample.name)
    target=ROOT/config['output_directory']
    if target.exists():raise FileExistsError('Preserve every checkpoint-state audit attempt')
    target.mkdir(parents=True)
    latest=original/'native_checkpoints/epoch_237/model'
    graph=tf.Graph()
    with graph.as_default():
        saver=tf.train.import_meta_graph(str(latest)+'.meta',clear_devices=True)
    variables=graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    trainable=graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    schema={v.name:v.shape.as_list() for v in variables}
    expected_trainable={**gan_trainable,'sigma:0':[2]}
    if {v.name:v.shape.as_list() for v in trainable}!=expected_trainable or any(schema.get(k)!=v for k,v in gan_schema.items()):
        raise ValueError('Every full original trainable and GAN optimizer variable schema must match')
    parameters=sum(int(np.prod(shape)) for shape in gan_trainable.values())
    if parameters!=96929 or not any('/Adam' in k for k in schema) or not any('/RMSProp' in k for k in schema):
        raise ValueError('Complete96929-parameter GAN with original Adam/RMSProp slots required')
    tensors={k:graph.get_tensor_by_name(names[k]) for k in ['Z','X','D_loss','G_loss','G_sample']}
    solvers=[graph.get_operation_by_name(names[k]) for k in ['D_solver','G_solver']]
    reachable_count=assert_no_random_transition(solvers)
    # Native control-flow functions must also contain no untracked random nodes.
    for function in graph.as_graph_def().library.function:
        if any('Random' in n.op or n.op in ('Multinomial','TruncatedNormal') for n in function.node_def):
            raise ValueError('A native graph function contains an untracked random operation')
    dataset=read(ROOT/queue['dataset_configuration'])
    if sha(dataset['derived_training_array'])!=queue['derived_training_sha256']:raise ValueError('Original data changed')
    table=np.load(dataset['derived_training_array'],mmap_mode='r',allow_pickle=False)
    with np.load(chain['derived_order_artifact']['path'],allow_pickle=False) as arrays:
        loss_noise=arrays['original_epoch_loss_noise']
        loss_ids=arrays['original_epoch_loss_batch_image_ids']
        final_ids=arrays['cumulative_train_image_ids']
    if loss_noise.shape!=(238,28,28,5) or loss_ids.shape!=(238,28) or final_ids.shape!=(36000,):
        raise ValueError('All238 native full last-batch noise/image identities required')
    trace=original/'experiments/traces'/(job['id']+'.trace.txt')
    lines=trace.read_text(encoding='utf-8').splitlines()
    if lines[0]!='epoch time D_loss G_loss mmd2 that pdf real_pdf':raise ValueError('Original trace schema changed')
    persisted={int(fields[0]):fields for fields in [line.split() for line in lines[1:]]}
    if list(persisted)!=list(range(231)):raise ValueError('The exact original persisted0..230 trace gap must be preserved')
    session_config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)
    records=[]
    with tf.Session(graph=graph,config=session_config) as session:
        for epoch in range(238):
            directory=original/'native_checkpoints'/('epoch_%03d'%epoch)
            checkpoint=str(directory/'model')
            reader=tf.train.load_checkpoint(checkpoint)
            if {k+':0':v for k,v in reader.get_variable_to_shape_map().items()}!=schema:
                raise ValueError('Every original checkpoint must retain all model and optimizer variables')
            saved={v.name:reader.get_tensor(v.op.name) for v in variables}
            if not all(np.isfinite(value).all() for value in saved.values()):raise ValueError('Nonfinite checkpoint variables')
            saver.restore(session,checkpoint)
            restored=session.run(variables)
            if any(not np.array_equal(value,saved[v.name]) for v,value in zip(variables,restored)):
                raise ValueError('Every restored model/optimizer variable must match the checkpoint bitwise')
            dump=original/'experiments/parameters'/(job['id']+'_'+str(epoch)+'.npy')
            dump_verified=False
            if dump.exists():
                parameters_dump=np.load(str(dump),allow_pickle=True).item()
                if set(parameters_dump)!=set(expected_trainable) or any(not np.array_equal(value,saved[name]) for name,value in parameters_dump.items()):
                    raise ValueError('Every available original parameter dump must match its actual full checkpoint')
                dump_verified=True
            values=session.run([tensors['D_loss'],tensors['G_loss']],feed_dict={
                tensors['X']:np.float32(table[loss_ids[epoch],1:].reshape(28,28,28)),tensors['Z']:loss_noise[epoch]})
            losses=[float(np.mean(value)) for value in values]
            if not np.isfinite(losses).all():raise ValueError('Reproduced complete native loss is nonfinite')
            if epoch in persisted:
                for value,text in zip(losses,persisted[epoch][2:4]):compare_loss(value,float(text),np)
            artifacts=[dict(path=str(p),sha256=sha(p)) for p in sorted(directory.iterdir()) if p.is_file()]
            records.append(dict(epoch=epoch,D_loss=losses[0],G_loss=losses[1],
                loss_provenance='native_checkpoint_full_last_batch_recalculation',
                original_persisted_trace_loss_bitwise_verified=epoch in persisted,
                original_elapsed_time=float(persisted[epoch][1]) if epoch in persisted else None,
                missing_original_elapsed_time_preserved=epoch not in persisted,
                original_parameter_dump_bitwise_verified=dump_verified,checkpoint_artifacts=artifacts,
                full_model_and_optimizer_restore_bitwise_verified=True))
            write(target/'progress.json',dict(full_checkpoints_verified=len(records),required=238,benchmark_performance=False))
        with np.load(evaluations[-1]['array_path'],allow_pickle=False) as evaluation:
            generated=evaluation['generated'];noise=evaluation['noise'];sigma=evaluation['sigma']
        replay=np.empty_like(generated)
        if not np.array_equal(session.run(graph.get_tensor_by_name('sigma:0')),sigma):
            raise ValueError('The final full evaluator sigma must match its checkpoint')
        for start in range(0,4984,28):
            replay[start:start+28]=session.run(tensors['G_sample'],feed_dict={tensors['Z']:noise[start:start+28]})
        if not np.array_equal(replay,generated):raise ValueError('All4984 latest generator samples must replay bitwise')
        with np.load(str(latest.parent/'numpy_random_state.npz'),allow_pickle=False) as state:
            np.random.set_state((str(state['algorithm'].item()),state['keys'],int(state['position']),int(state['has_gauss']),float(state['cached_gaussian'])))
        z_d=model.sample_Z(28,28,5,False);z_g=model.sample_Z(28,28,5,False)
        x=np.float32(table[final_ids[:28],1:].reshape(28,28,28))
        transitioned=[]
        for replay_id in range(2):
            saver.restore(session,str(latest))
            session.run(solvers[0],feed_dict={tensors['X']:x,tensors['Z']:z_d})
            session.run(solvers[1],feed_dict={tensors['Z']:z_g})
            transitioned.append(session.run(variables))
        if any(not np.isfinite(a).all() or not np.array_equal(a,b) for a,b in zip(*transitioned)):
            raise ValueError('The same full restored native optimizer transition must reproduce bitwise')
        if not any(not np.array_equal(a,saved[v.name]) for v,a in zip(variables,transitioned[0]) if v.name in gan_trainable):
            raise ValueError('The recovery transition proof must actually update the original GAN')
    if source_before!=verify_source(asset):raise ValueError('Original native source changed')
    verify_registration(config)
    write(target/'checkpoint_resume_state_audit.json',dict(passed=True,configuration=config_path,
        configuration_sha256=sha(ROOT/config_path),captured_utc=datetime.now(timezone.utc).isoformat(),job=job,
        actual_full_GAN_parameters=96929,all238_full_model_and_optimizer_checkpoints_restored_bitwise=True,
        all231_persisted_native_trace_losses_reproduced_bitwise_float32=True,
        seven_missing_loss_observations_recalculated_from_actual_checkpoint_and_verified_noise=True,
        seven_missing_original_elapsed_times_preserved=True,epochs=records,
        all4984_latest_generator_samples_replayed_bitwise=True,latest_full_generator_forward_calls=178,
        original_next_D_G_update_repeated_bitwise_in_all_model_and_optimizer_variables=True,
        training_reachable_operations_without_random_nodes=reachable_count,
        audit_only_optimizer_transition_updates=dict(D=2,G=2),audit_transition_weights_not_saved_for_training=True,
        random_chain_proof_sha256=sha(ROOT/read(ROOT/config['random_chain_configuration'])['output_directory']/'random_order_chain_audit.json'),
        native_checkpoint_variable_count=len(variables),full500_epoch_result_verified=False,
        benchmark_performance=False,additional_training_seed_slots=0,author_equivalence_certified=False,
        original_scope=config['original_scope'],source_receipts=config['source_receipts'],all_original_experiments_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify_registration(config)
    if args.probe_only:
        print(dict(native_checkpoints=238,numpy_imported='numpy' in sys.modules,tensorflow_imported='tensorflow' in sys.modules));return
    if args.worker:worker(config,args.config);return
    import psutil
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('This complete checkpoint audit already has a live controller')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),state='starting',active_pid=None)
    def update(values):state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[config['python'],'-X','utf8','-u','-m','scripts.flow_matching.audit_rcgan_checkpoint_resume_state_v1','--config',args.config,'--worker']
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                             CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                process=psutil.Process(child.pid);update(dict(active_pid=child.pid,active_create_time=process.create_time(),actual_command=process.cmdline()))
                code,usage=capped_wait(child,config,api,update)
        proof=ROOT/config['output_directory']/'checkpoint_resume_state_audit.json'
        passed=bool(code==0 and not usage['resource_guard_aborted'] and proof.exists() and read(proof)['passed'])
        write(base/'receipt.json',dict(exit_code=code,checkpoint_resume_state_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None,checkpoint_resume_state_passed=passed))
        if not passed:raise RuntimeError('Preserve failed complete native checkpoint-state audit')
    finally:owner.close()


if __name__=='__main__':main()
