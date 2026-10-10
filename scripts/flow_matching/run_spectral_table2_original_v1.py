"""Run full released native Table2 pipelines; diagnostics are never scores."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT,sha
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify(queue):
    for source in queue['source_receipts']:
        if sha(ROOT/source['path'])!=source['sha256']:
            raise ValueError('Frozen full native source changed: '+source['path'])
    if len(queue['jobs'])!=30 or len({j['id'] for j in queue['jobs']})!=30:
        raise ValueError('All six original five-seed pipeline cohorts required')
    protocol=read(ROOT/queue['protocol'])
    for data in protocol['data'].values():
        if sha(ROOT/data['path'])!=data['sha256']:raise ValueError('Original raw data changed')
    return protocol


def verify_environment(queue):
    lock=read(ROOT/queue['environment_lock'])
    if Path(sys.prefix).resolve()!=(ROOT/queue['python']).parent.parent.resolve():
        raise ValueError('Isolated native Python differs')
    for package,expected in lock['packages'].items():
        if importlib.metadata.version(package)!=expected:raise ValueError('Frozen package changed: '+package)
    if sha(ROOT/lock['inherited_environment_lock'])!=lock['inherited_environment_sha256']:
        raise ValueError('Inherited environment lock changed')
    if sha(lock['overlay_pth'])!=lock['overlay_pth_sha256']:raise ValueError('Overlay site path changed')


def workspace(queue,destination):
    protocol=verify(queue);source=ROOT/protocol['author_source']
    if destination.exists():raise FileExistsError('Preserve existing author attempt workspace')
    destination.mkdir(parents=True)
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.sh') and '__pycache__' not in path.parts:
            target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    for item in protocol['data'].values():
        target=destination/item['workspace_path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/item['path'],target)
        if sha(target)!=item['sha256']:raise ValueError('Private data mirror differs')
    return destination


def arguments_for(stage,seed):
    arguments=list(stage['original_arguments'])
    if '--gpu' in arguments:arguments[arguments.index('--gpu')+1]='0'
    if stage['stage']!='evaluation':arguments[arguments.index('--seed')+1]=str(seed)
    return arguments


def periodic_counts(stage):
    if not stage['optimizer_updates']:return Counter()
    points=[stage['warmup_updates']]+[stage['warmup_updates']+i for i in
        range(stage['periodic_eval_interval'],stage['main_optimizer_updates']+1,stage['periodic_eval_interval'])]
    return Counter({point:stage['discriminator_repeats'] for point in points})


def stage_contract(stage,job,private,index):
    entry=private/stage['entry']
    return dict(stage=stage['stage'],entry=str(entry),entry_sha256=sha(entry),workspace=str(private),
        arguments=arguments_for(stage,job['seed']),optimizer_updates=stage['optimizer_updates'],
        model_keys=stage['model_keys'],receipt=str(private.parent/f'stage_{index}_{stage["stage"]}.json'),
        final_checkpoint=str(private.parent/f'stage_{index}_full_final.pth'))


def complete(job):
    result_path=ROOT/job['output_directory']/'result.json'
    if not result_path.exists():return False
    result=read(result_path)
    if (result['status']!='completed' or result['job_sha256']!=fingerprint(job)
            or result['benchmark_smoke'] or result['evaluated_reference_shape']!=job['data']['shape']):return False
    import torch
    for index,stage in enumerate(job['stages']):
        record=result['stages'][index]
        if sha(record['receipt'])!=record['receipt_sha256']:return False
        receipt=read(record['receipt'])
        if receipt['status']!='completed' or receipt['optimizer_updates']!=stage['optimizer_updates']:return False
        if stage['optimizer_updates']:
            counts=Counter(r['generator_optimizer_updates'] for r in receipt['metrics'] if r['metric']=='discriminative')
            if counts!=periodic_counts(stage):return False
            checkpoint=record['final_checkpoint']
            if sha(checkpoint)!=receipt['final_checkpoint_sha256']:return False
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['optimizer_updates']!=stage['optimizer_updates'] or set(saved['models'])!=set(stage['model_keys']):return False
            steps=[float(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s]
            if not steps or any(x!=stage['optimizer_updates'] for x in steps):return False
            if saved['final_iteration']!=stage['main_optimizer_updates']:return False
            if sha(record['selected_checkpoint'])!=record['selected_checkpoint_sha256']:return False
        if stage['stage']!='evaluation':
            loaders=receipt['data_loaders']
            if not loaders or any(l['num_workers']!=8 or l['drop_last'] or l['sampler']!='RandomSampler' for l in loaders):return False
    for role in ('reference','generated'):
        item=result['arrays'][role]
        if sha(item['path'])!=item['sha256']:return False
    import numpy as np
    reference=np.load(result['arrays']['reference']['path'],mmap_mode='r',allow_pickle=False)
    generated=np.load(result['arrays']['generated']['path'],mmap_mode='r',allow_pickle=False)
    if list(reference.shape)!=job['data']['shape'] or generated.shape!=reference.shape:return False
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():return False
    if sha(ROOT/job['data']['path'])!=job['data']['sha256']:return False
    expected=original_array(job['dataset'],Path(result['workspace']))
    import hashlib
    def row_hashes(array):
        return Counter(hashlib.sha256(np.asarray(row,dtype=np.float32).tobytes()).digest() for row in array)
    if row_hashes(reference)!=row_hashes(expected):return False
    if set(result['metrics'])!={'context_fid','discriminative','predictive'}:return False
    evaluation=read(result['stages'][-1]['receipt'])
    for metric,values in result['metrics'].items():
        original=[r['value'] for r in evaluation['metrics'] if r['metric']==metric]
        if values['raw_repeats']!=original or len(original)!=5 or not all(math.isfinite(x) for x in original):return False
        if values['mean']!=statistics.mean(original):return False
    return True


def original_array(dataset,private):
    import numpy as np
    old_directory=Path.cwd();os.chdir(private);sys.path.insert(0,str(private))
    try:
        from dataset.dataset_VQ import TSDataset
        if dataset=='stock':
            data=TSDataset(dataset,window_size=24,dataset_type='train')
            return np.stack([data[i] for i in range(len(data))]).astype(np.float32)
        filename='sine_ground_truth_24_train.npy' if dataset=='sine' else 'mujoco_norm_truth_24_train.npy'
        return np.load(private/'dataset'/filename,allow_pickle=False).astype(np.float32)
    finally:os.chdir(old_directory)


def run_job(queue,job):
    verify_environment(queue);protocol=verify(queue)
    output=ROOT/job['output_directory'];artifact=Path(job['artifact_directory'])
    if output.exists() or artifact.exists():raise FileExistsError('Preserve previous incomplete/full original attempt')
    output.mkdir(parents=True);artifact.mkdir(parents=True)
    private=workspace(queue,artifact/'author_workspace')
    records=[];begin=time.perf_counter()
    environment=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',
        PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2',CUDA_VISIBLE_DEVICES='0')
    for index,stage in enumerate(job['stages']):
        contract=stage_contract(stage,job,private,index)
        contract_path=artifact/f'stage_{index}.contract.json';write(contract_path,contract)
        write(output/'progress.json',dict(stage=stage['stage'],stage_index=index,required_updates=stage['optimizer_updates']))
        command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.spectral_table2_bootstrap_v1',
            '--contract',str(contract_path)]
        with (output/f'stage_{index}.stdout.log').open('xb') as out,(output/f'stage_{index}.stderr.log').open('xb') as err:
            code=subprocess.run(command,cwd=ROOT,env=environment,stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
        if code!=0:raise RuntimeError('Full original stage failed: '+stage['stage'])
        receipt=read(contract['receipt'])
        record=dict(stage=stage['stage'],receipt=contract['receipt'],receipt_sha256=sha(contract['receipt']),
            final_checkpoint=contract['final_checkpoint'] if stage['optimizer_updates'] else None)
        if stage['optimizer_updates']:
            selected=private/stage['selected_checkpoint']
            record.update(selected_checkpoint=str(selected),selected_checkpoint_sha256=sha(selected))
        records.append(record)
    sampling=job['stages'][-2]
    from scripts.flow_matching.register_spectral_table2_original_v1 import option
    generated_dir=private/option(sampling['original_arguments'],'--out-dir')/option(sampling['original_arguments'],'--exp-name')
    names=('true_data.npy','fake_data.npy') if job['method']=='spectral_flow' else ('labels_ds.npy','pres_ds.npy')
    import numpy as np
    arrays={role:dict(path=str(generated_dir/name),sha256=sha(generated_dir/name)) for role,name in zip(('reference','generated'),names)}
    evaluation=read(records[-1]['receipt']);metrics={}
    for metric in protocol['metrics']:
        values=[r['value'] for r in evaluation['metrics'] if r['metric']==metric]
        metrics[metric]=dict(raw_repeats=values,mean=statistics.mean(values),population_std=statistics.pstdev(values))
    result=dict(id=job['id'],job_sha256=fingerprint(job),status='completed',workspace=str(private),
        method=job['method'],dataset=job['dataset'],seed=job['seed'],stages=records,metrics=metrics,arrays=arrays,
        evaluated_reference_shape=list(np.load(arrays['reference']['path'],mmap_mode='r').shape),
        seconds=time.perf_counter()-begin,completed_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_smoke=False,author_equivalence_certified=False,source_rng_boundary=protocol['source_rng_boundary'])
    write(output/'result.json',result)
    if not complete(job):raise ValueError('Full original result/checkpoint/data audit failed')


def preflight_cpu(queue,destination):
    verify_environment(queue);protocol=verify(queue)
    private=workspace(queue,destination.parent/'cpu_workspace')
    import numpy as np
    data={name:original_array(name,private) for name in protocol['data']}
    for name,array in data.items():
        if list(array.shape)!=protocol['data'][name]['shape'] or not np.isfinite(array).all():
            raise ValueError('Full original numeric data audit failed')
    sys.path.insert(0,str(private))
    from metrics.discriminative_metrics import discriminative_score_metrics
    # A full 2000-step original evaluator on identical real arrays is an API
    # diagnostic only, with no generated model or benchmark interpretation.
    start=time.perf_counter();diagnostic=float(discriminative_score_metrics(data['sine'],data['sine']))
    if not math.isfinite(diagnostic):raise ValueError('Original TF evaluator failed')
    write(destination,dict(passed=True,diagnostic_only=True,queue_sha256=fingerprint(queue),
        complete_raw_numeric_audit=True,data_shapes={n:list(a.shape) for n,a in data.items()},
        original_tf_discriminator_updates=2000,identical_real_array_diagnostic=diagnostic,
        seconds=time.perf_counter()-start,benchmark_result=False))


def preflight_gpu(queue,destination):
    verify_environment(queue);protocol=verify(queue)
    private=workspace(queue,destination.parent/'gpu_workspace')
    import numpy as np
    import torch
    torch.cuda.set_device(0);sys.path.insert(0,str(private))
    from models.spectral_flow.model import SpectralFlow
    import models.sdformer.vqvae as vqvae
    import models.sdformer.transformer_ar as transformer
    from scripts.flow_matching.register_spectral_table2_original_v1 import source_pipelines
    import options.option_spectral as spectral_options
    import options.option_vq as vq_options
    import options.option_transformer as ar_options
    old_argv=sys.argv;proof=[];loader_proof=[]
    from dataset.dataset_VQ import DATALoader
    for dataset in protocol['data']:
        loader=DATALoader(dataset,512,window_size=24,dataset_type='train')
        iterator=iter(loader);workers=list(iterator._workers)
        batches=[batch.numpy() for batch in iterator]
        actual=np.concatenate(batches,axis=0)
        if list(actual.shape)!=protocol['data'][dataset]['shape'] or not np.isfinite(actual).all():
            raise ValueError('Full original eight-worker loader failed')
        if len(workers)!=8 or any(worker.is_alive() or worker.exitcode!=0 for worker in workers):
            raise ValueError('Original eight-worker spawn did not terminate naturally')
        loader_proof.append(dict(dataset=dataset,num_workers=loader.num_workers,rows=len(actual),
            full_shape=list(actual.shape),worker_pids=[w.pid for w in workers],
            worker_exit_codes=[w.exitcode for w in workers],drop_last=loader.drop_last,
            sampler=type(loader.sampler).__name__))
        del iterator,loader,batches,actual,workers
    for (method,dataset),stages in source_pipelines(private).items():
        data=original_array(dataset,private)
        if method=='spectral_flow':
            sys.argv=[str(private/'train.py'),*arguments_for(stages[0],42)]
            args=spectral_options.get_args_parser();torch.manual_seed(args.seed)
            model=SpectralFlow(args).cuda();batch=torch.from_numpy(data[:args.batch_size]).float().cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(.5,.9))
            optimizer.zero_grad();loss=model(batch);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();torch.cuda.synchronize()
            proof.append(dict(method=method,dataset=dataset,batch_shape=list(batch.shape),loss=float(loss),
                parameters=sum(p.numel() for p in model.parameters()),backward_and_optimizer_step=True,
                peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            del model,optimizer,loss,batch
        else:
            sys.argv=[str(private/'train_vq.py'),*arguments_for(stages[0],42)]
            args=vq_options.get_args_parser();torch.manual_seed(args.seed)
            net=vqvae.VQVAE(args,args.nb_code,args.code_dim,args.down_t,args.stride_t,args.width,
                args.depth,args.dilation_growth_rate,args.vq_act,args.vq_norm).cuda()
            batch=torch.from_numpy(data[:args.batch_size]).float().cuda()
            optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr*2/(args.warm_up_iter+1),
                weight_decay=args.weight_decay,betas=(.9,.99))
            optimizer.zero_grad();prediction,commit,perplexity=net(batch)
            loss=args.recon*torch.nn.functional.mse_loss(prediction,batch)+args.commit*commit
            loss.backward();optimizer.step();torch.cuda.synchronize()
            proof.append(dict(method='sdformer_vq',dataset=dataset,batch_shape=list(batch.shape),loss=float(loss),
                parameters=sum(p.numel() for p in net.parameters()),backward_and_optimizer_step=True))
            del batch,optimizer,prediction,commit,loss
            sys.argv=[str(private/'train_sdformer_ar.py'),*arguments_for(stages[1],42)]
            args=ar_options.get_args_parser();torch.manual_seed(args.seed)
            net.eval();batch=torch.from_numpy(data[:args.batch_size]).float().cuda()
            with torch.no_grad():tokens=net.encode(batch)
            model=transformer.TSG_Transformer(num_vq=args.nb_code,embed_dim=args.embed_dim_gpt,block_size=args.block_size,
                num_layers=args.num_layers,n_head=args.n_head_gpt,drop_out_rate=args.drop_out_rate,fc_rate=args.ff_rate).cuda()
            input_index=tokens.clone();input_index[:,1:]=input_index[:,:-1];input_index[:,0]=args.nb_code
            mask=torch.bernoulli(args.pkeep*torch.ones(input_index.shape,device=input_index.device)).round().long()
            input_index=mask*input_index+(1-mask)*torch.randint_like(input_index,args.nb_code)
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(.5,.9))
            optimizer.zero_grad();prediction=model(input_index)
            loss=torch.nn.functional.cross_entropy(prediction.reshape(-1,prediction.shape[-1]),tokens.reshape(-1))
            loss.backward();optimizer.step();torch.cuda.synchronize()
            proof.append(dict(method=method,dataset=dataset,batch_shape=list(batch.shape),token_shape=list(tokens.shape),
                loss=float(loss),parameters=sum(p.numel() for p in model.parameters()),backward_and_optimizer_step=True,
                vq_initialization_boundary='Random initial VQ for interface diagnostic only; formal AR uses full-trained best VQ checkpoint'))
            del net,model,batch,tokens,input_index,mask,optimizer,prediction,loss
        import gc
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    sys.argv=old_argv
    if len(proof)!=9 or not all(math.isfinite(r['loss']) for r in proof):raise ValueError('Full original native interfaces failed')
    write(destination,dict(passed=True,diagnostic_only=True,queue_sha256=fingerprint(queue),native_full_shape_checks=proof,
        full_original_eight_worker_loaders=loader_proof,
        original_num_workers=8,original_sampling_method_retained=True,benchmark_result=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--job-id')
    parser.add_argument('--verify-result',action='store_true')
    parser.add_argument('--preflight',choices=['cpu','gpu']);parser.add_argument('--output')
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    if args.preflight:
        destination=ROOT/args.output
        if destination.exists():raise FileExistsError('Preserve completed preflight proof')
        (preflight_cpu if args.preflight=='cpu' else preflight_gpu)(queue,destination);return 0
    job=next(j for j in queue['jobs'] if j['id']==args.job_id)
    if args.verify_result:
        valid=complete(job);print(json.dumps(dict(id=job['id'],complete=valid)));return 0 if valid else 1
    run_job(queue,job);return 0


if __name__=='__main__':sys.exit(main())
