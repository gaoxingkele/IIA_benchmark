"""Execute full author Table1 generation training and its full four evaluators."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.flow_matching.prepare_spectral_original_data import ROOT, read, sha, write


def fingerprint(job):
    import hashlib
    return hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify(queue):
    for receipt in queue['source_receipts']:
        if sha(ROOT/receipt['path'])!=receipt['sha256']:
            raise ValueError('Frozen source changed: '+receipt['path'])
    if len(queue['jobs'])!=70 or len({j['id'] for j in queue['jobs']})!=70:
        raise ValueError('All fourteen paired five-seed generation cohorts required')
    return read(ROOT/queue['protocol'])


def complete(job):
    path=ROOT/job['output_directory']/'result.json'
    if not path.exists():return False
    try:
        result=read(path)
        if (result['job_sha256']!=fingerprint(job) or result['status']!='completed'
                or result['optimizer_updates']!=job['training_updates']
                or result['evaluated_reference_samples']!=job['data_shape'][0]
                or result['evaluation_repeats']!=5 or result['benchmark_smoke']):return False
        if set(result['metrics'])!={'context_fid','correlational','discriminative','predictive'}:return False
        for values in result['metrics'].values():
            if len(values['raw_repeats'])!=5 or not all(math.isfinite(v) for v in values['raw_repeats']):return False
            if not math.isclose(sum(values['raw_repeats'])/5,values['mean'],rel_tol=1e-12,abs_tol=1e-12):return False
        if {a['role'] for a in result['artifacts']}!={'checkpoint','generated','reference'}:return False
        for artifact in result['artifacts']:
            if sha(ROOT/artifact['path'])!=artifact['sha256']:return False
        import torch
        checkpoint=torch.load(ROOT/next(a['path'] for a in result['artifacts'] if a['role']=='checkpoint'),map_location='cpu',weights_only=True)
        if checkpoint['step']!=job['training_updates'] or not checkpoint['model'] or not checkpoint['ema']:return False
        steps=[float(state['step']) for state in checkpoint['opt']['state'].values() if 'step' in state]
        if not steps or any(value!=job['training_updates'] for value in steps):return False
        import numpy as np
        generated=np.load(ROOT/next(a['path'] for a in result['artifacts'] if a['role']=='generated'),mmap_mode='r',allow_pickle=False)
        reference=np.load(ROOT/next(a['path'] for a in result['artifacts'] if a['role']=='reference'),mmap_mode='r',allow_pickle=False)
        count=job['generation_batch_size']*(job['data_shape'][0]//job['generation_batch_size']+1)
        if list(reference.shape)!=job['data_shape'] or list(generated.shape)!=[count,*job['data_shape'][1:]]:return False
        if not np.isfinite(reference).all() or not np.isfinite(generated).all():return False
        if sha(ROOT/job['reference_path'])!=job['reference_sha256']:return False
        if not np.array_equal(reference,np.load(ROOT/job['reference_path'],mmap_mode='r',allow_pickle=False)):return False
        return True
    except (KeyError,ValueError,OSError,TypeError,RuntimeError,EOFError):return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--job-id',required=True)
    parser.add_argument('--verify-result',action='store_true')
    args=parser.parse_args();queue=read(ROOT/args.queue);policy=verify(queue)
    job=next(j for j in queue['jobs'] if j['id']==args.job_id)
    if args.verify_result:
        valid=complete(job);print(json.dumps({'id':job['id'],'complete':valid}));return 0 if valid else 1
    if complete(job):return
    output=ROOT/job['output_directory'];artifact=Path(job['artifact_directory'])
    if output.exists() or artifact.exists():raise FileExistsError('Preserve partial original author attempt')
    for path,expected in [(job['frozen_samples'],job['data_sha256']),(job['reference_path'],job['reference_sha256'])]:
        if sha(ROOT/path)!=expected:raise ValueError('Paired source data changed')
    lock=read(ROOT/queue['environment_lock'])
    for package,expected in lock['packages'].items():
        if importlib.metadata.version(package)!=expected:raise ValueError('Environment version changed: '+package)
    artifact.mkdir(parents=True);output.mkdir(parents=True)
    workspace=artifact/'author_workspace'
    shutil.copytree(ROOT/policy['author_source'],workspace,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(workspace));sys.path.insert(0,str(ROOT/'src'))
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    from engine.solver import Trainer
    from engine.logger import Logger
    from utils.io_utils import seed_everything,instantiate_from_config
    from models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
    from iia_benchmark.models.fm_spectral_author import FrozenAuthorGenerationDataset
    torch.set_num_threads(2);torch.cuda.set_device(policy['resource_policy']['gpu_index'])
    seed_everything(job['seed'])
    config=yaml.safe_load((ROOT/job['author_yaml']).read_text(encoding='utf-8'))
    source_steps=config['solver']['max_epochs']
    if source_steps!=job['training_updates'] or config['solver']['gradient_accumulate_every']!=job['gradient_accumulate_every']:
        raise ValueError('Full author training budget differs')
    arguments=SimpleNamespace(config=str(ROOT/job['author_yaml']),gpu=policy['resource_policy']['gpu_index'],
        name=job['id'],output=str(artifact),tensorboard=False,seed=job['seed'],milestone=10,
        test_batch_size=job['generation_batch_size'],opts=None,save_dir=str(artifact/'author_output'))
    logger=Logger(arguments);logger.save_config(config)
    model=instantiate_from_config(config['model']).cuda()
    parameter_count=sum(p.numel() for p in model.parameters())
    if job.get('expected_parameters') is not None and parameter_count!=job['expected_parameters']:
        raise ValueError('Published Table9 parameter count differs')
    data=FrozenAuthorGenerationDataset(ROOT/job['frozen_samples'])
    loader=DataLoader(data,batch_size=config['dataloader']['batch_size'],shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    trainer=Trainer(config,arguments,model,{'dataloader':loader,'dataset':data},logger)
    optimizer_step=trainer.opt.step;observed_updates=0
    def update(*a,**kw):
        nonlocal observed_updates
        result=optimizer_step(*a,**kw);observed_updates+=1
        if observed_updates%100==0:
            write(output/'progress.json',{'phase':'training','optimizer_updates':observed_updates,'required_updates':source_steps})
        return result
    trainer.opt.step=update
    start=time.perf_counter();trainer.train();training_seconds=time.perf_counter()-start
    if observed_updates!=source_steps or trainer.step!=source_steps:raise ValueError('Full author training was not completed')
    trainer.load(10)
    checkpoint=Path(arguments.save_dir)/'checkpoints/checkpoint-10.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['step']!=source_steps:raise ValueError('Complete checkpoint step mismatch')
    if job['generation_field_chunk_size'] is not None:
        raise ValueError('Chunked generation is not author-equivalent and is excluded')
    write(output/'progress.json',{'phase':'sampling','optimizer_updates':observed_updates})
    start=time.perf_counter();samples=trainer.sample(num=len(data),size_every=job['generation_batch_size'],shape=[data.window,data.var_num])
    samples=unnormalize_to_zero_to_one(samples);sampling_seconds=time.perf_counter()-start
    generated=artifact/'fake_data.npy';np.save(generated,samples,allow_pickle=False)
    reference=artifact/'reference_data.npy';shutil.copy2(ROOT/job['reference_path'],reference)
    # Release trained networks before allocating the full original evaluators.
    del saved,trainer,model,loader,data,optimizer_step
    import gc
    gc.collect();torch.cuda.empty_cache()
    import tensorflow as tf
    from utils.context_fid import context_fid
    from utils.cross_correlation import CrossCorrelLoss
    from utils.discriminative_metric import discriminative_score_metrics
    from utils.predictive_metric import predictive_score_metrics
    from scipy.stats import t
    seed_everything(policy['evaluation_seed'])
    tf1=discriminative_score_metrics.__globals__['tf1']
    original_reset=tf1.reset_default_graph;graph_seed={'value':0}
    def seeded_reset():
        original_reset();tf1.set_random_seed(graph_seed['value'])
    tf1.reset_default_graph=seeded_reset
    predictive_score_metrics.__globals__['tf1'].reset_default_graph=seeded_reset
    truth=np.load(reference,allow_pickle=False);fake=samples[:len(truth)]
    raw={key:[] for key in ['context_fid','correlational','discriminative','predictive']}
    start=time.perf_counter()
    for metric in raw:
        for repeat in range(policy['evaluation_repeats']):
            graph_seed['value']=policy['evaluation_seed']+repeat
            write(output/'progress.json',{'phase':'evaluation','metric':metric,'repeat':repeat,'completed_metrics':raw})
            if metric=='context_fid':value=context_fid(truth[:],fake[:])
            elif metric=='correlational':
                size=int(len(truth)/policy['evaluation_repeats'])
                real_ids=np.random.randint(0,len(truth),size);fake_ids=np.random.randint(0,len(fake),size)
                value=CrossCorrelLoss(torch.from_numpy(truth[real_ids]),name='CrossCorrelLoss').compute(torch.from_numpy(fake[fake_ids])).item()
            elif metric=='discriminative':value=discriminative_score_metrics(truth[:],fake[:])[0]
            else:value=predictive_score_metrics(truth[:],fake[:])
            raw[metric].append(float(value))
    evaluation_seconds=time.perf_counter()-start
    metrics={}
    for key,values in raw.items():
        mean=float(np.mean(values));std=float(np.std(values,ddof=1));se=std/len(values)**.5;margin=float(t.ppf(.975,len(values)-1))*se
        metrics[key]={'raw_repeats':values,'mean':mean,'std':std,'se':se,'ci95_low':mean-margin,'ci95_high':mean+margin,'author_display_halfwidth':margin,'uncertainty_unit':'five full evaluator runs for one trained model'}
    result={'status':'completed','job':job,'job_sha256':fingerprint(job),'optimizer_updates':observed_updates,
        'gradient_accumulate_every':job['gradient_accumulate_every'],'training_minibatches':observed_updates*job['gradient_accumulate_every'],
        'evaluated_reference_samples':len(truth),'generated_samples':len(samples),'evaluation_repeats':policy['evaluation_repeats'],
        'parameter_count':parameter_count,'training_seconds':training_seconds,'sampling_seconds':sampling_seconds,'evaluation_seconds':evaluation_seconds,
        'data_audit':{'frozen_samples':job['frozen_samples'],'frozen_samples_sha256':job['data_sha256'],
            'reference_path':job['reference_path'],'reference_sha256':job['reference_sha256'],'shape':job['data_shape'],
            'actual_training_loader':'FrozenAuthorGenerationDataset over exact cached author-loader arrays',
            'window_order':'full original stride-one or simulated-trajectory order','data_track':job['data_track']},
        'metrics':metrics,'benchmark_smoke':False,'author_equivalence_certified':False,'explicit_boundaries':policy['explicit_boundaries'],
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'artifacts':[{'role':role,'path':str(path),'sha256':sha(path)} for role,path in [('checkpoint',checkpoint),('generated',generated),('reference',reference)]]}
    write(output/'result.json',result)
    if not complete(job):raise ValueError('Full original author execution audit failed')
    print(json.dumps({'id':job['id'],'status':'completed','metrics':metrics}))


if __name__=='__main__':sys.exit(main())
