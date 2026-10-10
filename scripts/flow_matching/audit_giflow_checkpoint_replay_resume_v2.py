"""Resume a terminal GiFlow replay without discarding verified whole test batches."""
import argparse
import ast
import copy
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def native_test_function(source, prefix_metrics=None):
    """Extract original model.eval, metric lists and whole native no_grad test block."""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_experiment')
    assignments = {n.targets[0].id: n for n in function.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    start = function.body.index(assignments['losses_mae'])
    block = function.body[start:start+4]
    if ([n.targets[0].id for n in block[:3]] != ['losses_mae','losses_mse','losses_mape']
            or not isinstance(block[3], ast.With)):
        raise ValueError('Frozen original test-loop boundary changed')
    steps = [n.value.value for n in ast.walk(block[3]) if isinstance(n, ast.Assign)
             and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='num_steps'
             and isinstance(n.value,ast.Constant)]
    if steps != [20]:
        raise ValueError('All20 original Euler steps required')
    mode = function.body[start-2]
    if ast.unparse(mode) != 'model.eval()' or ast.unparse(function.body[start-1]) != 'ema.apply_shadow()':
        raise ValueError('Actual saved test weights must replace only original EMA application')
    # Saved evaluated_model.pt already contains the actually applied shadow weights.
    # The native restore occurs only after all test predictions and changes no metric.
    args = ['model','test_loader','node_embed','edge_index','H_spatial','H_temporal','args','device','ema']
    body=copy.deepcopy([mode]+block)
    if prefix_metrics is not None:
        # Exact original unweighted mean of every full native batch. Only complete
        # previously executed batches populate the original metric lists.
        for node,key in zip(body[1:4], ('mae','mse','mape')):
            if not isinstance(node.value, ast.List) or node.value.elts:
                raise ValueError('Expected original empty native metric lists')
            node.value.elts=[ast.Constant(value=float(v)) for v in prefix_metrics[key]]
    generated = ast.FunctionDef(name='replay_native_test', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=a) for a in args], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=body, decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=[generated],type_ignores=[]))


def compare_vectors(actual, saved, tolerance):
    import numpy as np
    if actual.shape != saved.shape or not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError('Complete finite trained-model prediction vectors required')
    difference = np.abs(actual.astype(np.float64)-saved.astype(np.float64))
    if not np.allclose(actual,saved,**tolerance):
        raise ValueError('Full GiFlow model prediction differs; max_abs='+str(float(difference.max())))
    return dict(max_absolute_difference=float(difference.max()),
        max_relative_difference=float((difference/np.maximum(np.abs(saved),1e-12)).max()),
        bitwise_identical=bool(np.array_equal(actual,saved)))


def verify(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen whole GiFlow model replay input changed: '+item['path'])


def verified_prefix(config, full_loader, native, spatial, temporal, originals, result):
    """Audit preserved predictions against the freshly loaded full native data."""
    import numpy as np
    import torch
    prior=config['preserved_prefix']
    progress=read(ROOT/prior['progress_path'])
    count=prior['batch_count']
    if (progress['completed_batches']!=count or progress['required_batches']!=result['full_test_batches']
            or progress['forward_calls']!=20*count):
        raise ValueError('Frozen previous original inference budget changed')
    iterator=iter(full_loader)
    metrics=dict(mae=[],mse=[],mape=[])
    records=[]
    for index,registered in enumerate(prior['batches']):
        if index!=registered['batch'] or sha(registered['path'])!=registered['sha256']:
            raise ValueError('Only intact contiguous complete native batch prefix can resume')
        batch=next(iterator).to(torch.device('cpu'))
        values=native.preprocess_fm_test(batch,spatial,temporal)
        truth=values[2][values[4]].detach().numpy()
        positions=np.flatnonzero(values[4].numpy());shape=np.asarray(values[4].shape)
        with np.load(registered['path'],allow_pickle=False) as cache, np.load(originals[index]['path'],allow_pickle=False) as original:
            for key,value in [('target',truth),('masked_flat_positions',positions),('batch_shape',shape)]:
                if not np.array_equal(value,cache[key]) or not np.array_equal(value,original[key]):
                    raise ValueError('Cached whole batch identity differs from actual original native data: '+key)
            prediction=cache['prediction'].copy()
            agreement=compare_vectors(prediction,original['prediction'],config['prediction_tolerance'])
        # Keep the original float32 tensor reduction, including nonzero MAPE mask.
        pred=torch.from_numpy(prediction);actual=torch.from_numpy(truth)
        delta=pred-actual;nonzero=torch.abs(actual)>1e-6
        metrics['mae'].append(float(torch.mean(torch.abs(delta)).item()))
        metrics['mse'].append(float(torch.mean(delta**2).item()))
        metrics['mape'].append(float((torch.mean(torch.abs(delta[nonzero]/actual[nonzero]))*100).item()))
        records.append(dict(batch=index,windows=int(shape[0]),targets=len(truth),path=registered['path'],
            sha256=registered['sha256'],original_sha256=originals[index]['sha256'],agreement=agreement,
            preserved_completed_batch=True))
    if len(records)!=count or 20*sum(x['windows'] for x in records)!=progress['window_evaluations']:
        raise ValueError('Preserved full native model-forward window budget changed')
    return iterator,metrics,records


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    verify(config)
    queue=read(ROOT/config['queue'])
    from scripts.flow_matching.giflow_native_protocol import verify as verify_queue, path_only_loader, configured_loader
    from scripts.flow_matching.summarize_giflow_canonical_v1 import observed_result
    from scripts.flow_matching.run_giflow_native_job import audit_data
    verify_queue(queue)
    for line in (ROOT/queue['environment_lock']).read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        package,expected=line.strip().split('==',1)
        if importlib.metadata.version(package)!=expected:
            raise ValueError('Locked original GiFlow environment required: '+package)
    import numpy as np
    import torch
    from torch_geometric.utils import get_laplacian,to_scipy_sparse_matrix
    job=next(j for j in queue['jobs'] if j['id']==config['job_id'])
    result=observed_result(job,config['queue'])
    if result is None:raise ValueError('Actual full completed original training required')
    target,artifacts=ROOT/config['output_directory'],Path(config['artifact_root'])
    if target.exists() or artifacts.exists():raise FileExistsError('Preserve all prior replay attempts')
    target.mkdir(parents=True);artifacts.mkdir(parents=True)
    source=ROOT/job['author_source'];sys.path.insert(0,str(source))
    native=importlib.import_module('main');data=importlib.import_module('data')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    data_root=ROOT/read(ROOT/queue['data_config'])['output_root']
    loader=path_only_loader(data,data_root)[0] if job['track']=='released_mirror' else configured_loader(data,data_root)
    args=argparse.Namespace(**job['arguments'])
    # Only device-bound native plain tensors move to CPU; architecture unchanged.
    args.device,args.cuda='cpu',False
    keys=('dataset_name','missing_rate','missing_type','window','stride','adj_threshold','val_len','test_len','seed','batch_size')
    dm,edges,weights,nodes,time_dim=loader(**{k:getattr(args,k) for k in keys})
    binding=audit_data(dm,edges,weights,nodes,artifacts)
    original_output=ROOT/job['output_directory']
    with np.load(original_output/'split_mask_graph_scaler.npz',allow_pickle=False) as old, np.load(artifacts/'split_mask_graph_scaler.npz',allow_pickle=False) as new:
        if set(old.files)!=set(new.files) or any(not np.array_equal(old[k],new[k]) for k in old.files):
            raise ValueError('Full native masks, graph, scaler or grouped timestamp splits changed')
    for key,value in result['data_binding'].items():
        if key!='audit_sha256' and binding[key]!=value:raise ValueError('Original full data binding differs: '+key)
    args.time_dim,args.n_nodes=time_dim,nodes
    spatial=temporal=None
    if args.diffusion_type in ('graph','graph_time'):
        le,lw=get_laplacian(edges,normalization='sym')
        lap=to_scipy_sparse_matrix(le,lw,num_nodes=nodes).astype('float64')
        spatial=native.operator_generation(lap,args.k_eig,args.tau_s)
    if args.diffusion_type in ('time','graph_time'):
        temporal=native.operator_generation(native.generate_time_laplacian(args.window),args.k_eig,args.tau_t)
    np.savez_compressed(artifacts/'native_operators.npz',spatial=spatial.numpy() if spatial is not None else np.empty(0),
                        temporal=temporal.numpy() if temporal is not None else np.empty(0))
    model=native.models.FlowMatching(args)
    state=torch.load(original_output/'evaluated_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    if any(not torch.equal(value,state[key]) for key,value in model.state_dict().items()):
        raise ValueError('Every actual tested state tensor must be restored exactly')
    if sum(p.numel() for p in model.parameters())!=result['parameters']:
        raise ValueError('Original full architecture parameter count differs')
    original_batches=[item for item in result['artifacts'] if 'targets' in item]
    full_loader=dm.test_dataloader(batch_size=args.batch_size,shuffle=False)
    with torch.no_grad():
        remaining_loader,prefix_metrics,preserved=verified_prefix(config,full_loader,native,spatial,temporal,original_batches,result)
    prefix_calls=20*len(preserved);prefix_windows=20*sum(b['windows'] for b in preserved)
    context=dict(batches=preserved,current=None,forward_calls=prefix_calls,window_evaluations=prefix_windows,forward_rng=None)
    write(target/'progress.json',dict(phase='verified_preserved_complete_batch_prefix',completed_batches=len(preserved),
        required_batches=result['full_test_batches'],forward_calls=prefix_calls,window_evaluations=prefix_windows))
    def before(module,inputs,kwargs):
        if module.training:raise ValueError('Native trained-model evaluation mode required')
        context['forward_rng']=torch.random.get_rng_state().clone()
        context['forward_calls']+=1;context['window_evaluations']+=len(kwargs['x'])
    def after(module,inputs,output):
        if not torch.equal(context['forward_rng'],torch.random.get_rng_state()):
            raise ValueError('Original deterministic test inference unexpectedly consumes RNG')
    hooks=[model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(after)]
    original_preprocess=native.preprocess_fm_test
    patched=set()
    def preprocess(batch,*a):
        values=original_preprocess(batch,*a)
        context['current']=dict(target=values[2],mask=values[4],recorded=False)
        cls=type(batch.transform['y'])
        if cls not in patched:
            original_inverse=cls.inverse_transform
            def inverse(scaler,prediction,*rest,**kw):
                unscaled=original_inverse(scaler,prediction,*rest,**kw)
                current=context['current']
                if current is not None and not current['recorded']:
                    current['recorded']=True;index=len(context['batches']);mask=current['mask']
                    prediction=unscaled[mask].detach().numpy();truth=current['target'][mask].detach().numpy()
                    positions=np.flatnonzero(mask.numpy());shape=np.asarray(mask.shape)
                    path=artifacts/f'batch_{index:05d}.npz'
                    # Save the complete replay before checking numerical agreement.
                    np.savez_compressed(path,prediction=prediction,target=truth,masked_flat_positions=positions,batch_shape=shape)
                    original=original_batches[index]
                    with np.load(original['path'],allow_pickle=False) as saved:
                        for name,value in [('target',truth),('masked_flat_positions',positions),('batch_shape',shape)]:
                            if not np.array_equal(saved[name],value):raise ValueError('Full original test target identity/geometry differs: '+name)
                        agreement=compare_vectors(prediction,saved['prediction'],config['prediction_tolerance'])
                    context['batches'].append(dict(batch=index,windows=int(shape[0]),targets=len(truth),
                        path=str(path),sha256=sha(path),original_sha256=original['sha256'],agreement=agreement))
                    write(target/'progress.json',dict(phase='full_native_model_reinference',completed_batches=index+1,
                        required_batches=result['full_test_batches'],forward_calls=context['forward_calls'],
                        window_evaluations=context['window_evaluations']))
                return unscaled
            cls.inverse_transform=inverse;patched.add(cls)
        return values
    namespace=dict(vars(native));namespace['preprocess_fm_test']=preprocess
    exec(compile(native_test_function((source/'main.py').read_text(encoding='utf-8'),prefix_metrics),str(source/'main.py'),'exec'),namespace)
    started=time.perf_counter()
    # Original EMA application is represented by its actually saved tested weights.
    # Restoration after the complete loop is immaterial to scores and remains local.
    values=tuple(float(v) for v in namespace['replay_native_test'](model,
        remaining_loader,torch.arange(nodes),edges,spatial,temporal,args,
        torch.device('cpu'),SimpleNamespace(restore=lambda:None)))
    for hook in hooks:hook.remove()
    metrics=dict(zip(('native_mean_batch_mae','native_mean_batch_mse','native_mean_batch_mape_percent'),values))
    metrics['sqrt_native_mean_batch_mse']=math.sqrt(metrics['native_mean_batch_mse'])
    expected_windows=result['data_binding']['windows']['test']
    if (len(context['batches'])!=result['full_test_batches'] or sum(b['windows'] for b in context['batches'])!=expected_windows
            or context['forward_calls']!=20*result['full_test_batches'] or context['window_evaluations']!=20*expected_windows
            or sum(b['targets'] for b in context['batches'])!=result['test_targets_with_window_repetitions']):
        raise ValueError('Every full original test window/target and all20 Euler steps required')
    for key,value in metrics.items():
        if not math.isclose(value,result['metrics'][key],**config['metric_tolerance']):
            raise ValueError('Original whole-test replay metric differs: '+key)
    if any(not torch.equal(value,state[key]) for key,value in model.state_dict().items()) or torch.cuda.is_initialized():
        raise ValueError('Read-only full CPU replay must not mutate actual weights or initialize CUDA')
    verify(config);verify_queue(queue)
    proof=dict(job_id=job['id'],seed=job['seed'],full_checkpoint_contents_and_reinference_passed=True,
        full_data_masks_graph_scaler_and_splits_bitwise=True,actual_tested_weights_restored_bitwise=True,
        original_batch_size=args.batch_size,full_test_windows=expected_windows,full_test_batches=len(context['batches']),
        full_masked_test_targets=sum(b['targets'] for b in context['batches']),original_Euler_steps=20,
        observed_forward_calls=context['forward_calls'],observed_window_evaluations=context['window_evaluations'],
        preserved_completed_native_batches=len(preserved),newly_replayed_native_batches=len(context['batches'])-len(preserved),
        this_attempt_forward_calls=context['forward_calls']-prefix_calls,
        this_attempt_window_evaluations=context['window_evaluations']-prefix_windows,
        original_metrics=result['metrics'],replayed_metrics=metrics,full_batch_receipts=context['batches'],
        elapsed_seconds=time.perf_counter()-started,deterministic_model_forwards_consume_no_RNG=True,
        original_training_seed_slots_unchanged=True,not_a_new_training_seed=True,cuda_context_initialized=False,
        author_equivalence_certified=False,all_original_experiments_complete=False,
        boundary=config['boundary'])
    write(target/'checkpoint_replay_audit.json',proof)
    write(target/'validation.json',dict(passed=True,source_receipts=config['source_receipts'],
        outputs={'checkpoint_replay_audit.json':sha(target/'checkpoint_replay_audit.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.worker:worker(config,args.config);return
    api=resource_api();base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    owner=api['exclusive_lock'](base/'controller.lock')
    if owner is None:raise RuntimeError('Full GiFlow checkpoint replay controller already live')
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),active_pid=None,state='starting')
    def update(values):
        if values.get('state')=='replaying_full_checkpoints':values=dict(values,state='replaying_full_native_GiFlow_checkpoint')
        state.update(values);write(base/'status.json',state)
    try:
        with api['resource_slot'](ROOT/config['resource_lock'],config['settings'],update):
            command=[str(ROOT/config['python']),'-X','utf8','-u','-m','scripts.flow_matching.audit_giflow_checkpoint_replay_resume_v2',
                     '--config',args.config,'--worker']
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                     CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads']))
            with (base/'worker.stdout.log').open('xb') as out,(base/'worker.stderr.log').open('xb') as err:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                update(dict(active_pid=child.pid,active_create_time=psutil.Process(child.pid).create_time(),actual_command=command))
                code,usage=capped_wait(child,config,api,update)
        path=ROOT/config['output_directory']/'checkpoint_replay_audit.json'
        passed=code==0 and path.exists() and read(path)['full_checkpoint_contents_and_reinference_passed']
        write(base/'receipt.json',dict(exit_code=code,full_model_replay_passed=passed,**usage))
        update(dict(state='completed' if passed else 'failed_preserved',active_pid=None))
        if not passed:raise RuntimeError('Full original checkpoint replay failed; preserve all partial evidence')
    finally:owner.close()


if __name__=='__main__':main()
