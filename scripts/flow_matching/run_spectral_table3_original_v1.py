"""Full released Table3 data audit, capacity checks, training and ten-repeat evaluation."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(queue):
    for entry in queue['source_receipts']:
        if sha(ROOT/entry['path'])!=entry['sha256']:
            raise ValueError('Full original Table3 source/input differs: '+entry['path'])
    for job in queue['jobs']:
        config=read(ROOT/job['model_config'])
        if sha(ROOT/job['model_config'])!=job['model_config_sha256'] or config['diagnostic']:
            raise ValueError('Full original model configuration differs')
    if len(queue['jobs'])!=4:
        raise ValueError('Both full methods on both original datasets are required')


def parse_tsf(path):
    """Independently read the full released numeric rows, without author imports."""
    rows=[]
    for line in Path(path).read_text(encoding='cp1252').splitlines():
        text=line.strip()
        if not text or text.startswith(('#','@')):continue
        values=[float(v) for v in text.rsplit(':',1)[-1].split(',')]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite original TSF row')
        rows.append(values)
    if not rows or len({len(row) for row in rows})!=1:
        raise ValueError('Full equal-length original trajectories required')
    return rows


def data_audit(queue,output):
    import numpy as np
    records={}
    base=Path(output).parent
    base.mkdir(parents=True,exist_ok=True)
    for dataset,spec in queue['datasets'].items():
        rows=parse_tsf(ROOT/spec['raw_path'])
        raw=np.asarray(rows,dtype=np.float64)[:,:,None]
        if list(raw.shape)!=spec['shape']:raise ValueError('Full released original trajectory dimensions differ')
        # Match the author's float32 mean/std tensors and float32 trajectory tensors.
        mean=raw.mean(axis=1,keepdims=True).astype(np.float32)
        std=raw.std(axis=1,keepdims=True).astype(np.float32);std[std==0]=1
        normalized=(raw.astype(np.float32)-mean)/std
        if not np.isfinite(normalized).all():raise ValueError('Original normalization is nonfinite')
        arrays=base/(dataset+'_full_normalized.npz')
        if arrays.exists():raise FileExistsError('Preserve prior full data audit arrays')
        np.savez_compressed(arrays,raw=raw,normalized=normalized)
        splits=author_data_splits(queue,dataset,normalized) if 'source_root' in queue else None
        records[dataset]=dict(raw_shape=list(raw.shape),raw_sha256=sha(ROOT/spec['raw_path']),
            arrays=str(arrays.resolve()),arrays_sha256=sha(arrays),
            per_full_trajectory_normalization=True,random_series_split=[int(len(raw)*.8),len(raw)-int(len(raw)*.8)],
            paper_stated_length=spec['paper_stated_length'],released_length_differs_from_paper=spec['shape'][1]!=spec['paper_stated_length'],
            original_author_splits=splits,
            boundary='Full released trajectories are retained. No crop to the paper length, forecasting split, or training-only scaler substituted.')
    proof=dict(passed=True,diagnostic_only=True,kind='full_independent_raw_data_audit',
        queue_sha256=sha(ROOT/queue['queue_path']),datasets=records,formal_benchmark_result=False)
    write(output,proof);return proof


def author_data_splits(queue,dataset,normalized):
    """Check the real original CPU data function for both released seed policies."""
    # A data-only subprocess must not allocate a competing CUDA model/context.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    sys.path.insert(0,str((ROOT/queue['source_root']).resolve()))
    import torch
    import numpy as np
    from types import SimpleNamespace
    from utils.utils_data import gen_dataloader
    old=Path.cwd();os.chdir(ROOT/queue['source_root']);records={}
    try:
        for phase,seed in [('training',42),('evaluation',0)]:
            torch.manual_seed(seed)
            generator=torch.Generator().manual_seed(seed)
            order=torch.randperm(len(normalized),generator=generator).numpy();boundary=int(len(normalized)*.8)
            args=SimpleNamespace(dataset=dataset,batch_size=32,num_workers=4,device='cpu')
            train,test=gen_dataloader(args)
            for loader,ids in zip((train,test),(order[:boundary],order[boundary:])):
                if not np.allclose(loader.dataset.tensors[0].numpy(),normalized[ids],rtol=2e-6,atol=2e-6):
                    raise ValueError('Real original full author preprocessing/split differs')
            records[phase]=dict(seed=seed,train_ids=order[:boundary].tolist(),test_ids=order[boundary:].tolist(),
                seq_len=args.seq_len,full_rows_verified=len(normalized))
        records['training_rows_in_standalone_evaluation']=len(set(records['training']['train_ids'])&set(records['evaluation']['test_ids']))
        return records
    finally:os.chdir(old)


def prepare_workspace(job):
    artifact=Path(job['artifact_directory'])
    if artifact.exists():raise FileExistsError('Prior full author workspace preserved')
    workspace=artifact/'workspace';(workspace/'data/long_range').mkdir(parents=True)
    config=read(ROOT/job['model_config'])
    source=ROOT/config['raw_path'];target=workspace/'data/long_range'/source.name
    shutil.copy2(source,target)
    if sha(target)!=sha(source):raise ValueError('Workspace original raw dataset differs')
    return artifact,workspace,config


def original_optimizer(entry,model,args,torch):
    """Compile only the complete original optimizer assignments, without substitutions."""
    tree=ast.parse(Path(entry).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    names={'muon_params','adamw_params','param_groups','optimizer'}
    selected=[n for n in main.body if isinstance(n,ast.Assign)
              and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
    if not selected or selected[-1].targets[0].id!='optimizer':raise ValueError('Original full optimizer assignments missing')
    namespace=dict(model=model,args=args,torch=torch)
    if 'muon_params' in {n.targets[0].id for n in selected}:
        from models.spectral_flow.muon import SingleDeviceMuonWithAuxAdam
        namespace['SingleDeviceMuonWithAuxAdam']=SingleDeviceMuonWithAuxAdam
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(entry),'exec'),namespace)
    return namespace['optimizer']


def contract(queue,job,stage,artifact,workspace,config):
    entry=ROOT/config[stage+'_entry']
    ds=queue['datasets'][config['dataset']]
    raw=read(ROOT/queue['cpu_data_audit'])['datasets'][config['dataset']]
    return dict(method=config['method'],stage=stage,source_root=str((ROOT/queue['source_root']).resolve()),
        entry=str(entry.resolve()),entry_sha256=sha(entry),workspace=str(workspace.resolve()),
        arguments=['--config',str((ROOT/config['author_config']).resolve()),'--seed',str(config['seed']),
                   '--log_dir',str((artifact/'native_checkpoints').resolve())],
        normalized_arrays=raw['arrays'],epochs=config['author_hyperparameters']['epochs'],
        batch_size=config['author_hyperparameters']['batch_size'],num_workers=4,
        batches_per_epoch=math.ceil(int(ds['shape'][0]*.8)/config['author_hyperparameters']['batch_size']),
        periodic_evaluations=10,test_shape=[ds['shape'][0]-int(ds['shape'][0]*.8),*ds['shape'][1:]],
        receipt=str((artifact/(stage+'_receipt.json')).resolve()),
        final_checkpoint=str((artifact/'full_final_training.pt').resolve()),array_root=str((artifact/stage/'arrays').resolve()))


def preflight(queue,job,output):
    from scripts.flow_matching.spectral_table3_bootstrap_v1 import configure_modules
    artifact,workspace,config=prepare_workspace(dict(job,artifact_directory=job['preflight_artifact_directory']))
    c=contract(queue,job,'training',artifact,workspace,config)
    torch,args=configure_modules(c)
    if not torch.cuda.is_available():raise ValueError('Full original GPU model required')
    from utils.utils_data import gen_dataloader
    import numpy as np
    raw=read(ROOT/queue['cpu_data_audit'])['datasets'][config['dataset']]
    with np.load(raw['arrays'],allow_pickle=False) as payload:normalized=payload['normalized']
    generator=torch.Generator().set_state(torch.get_rng_state())
    order=torch.randperm(len(normalized),generator=generator).numpy();boundary=int(len(normalized)*.8)
    train,test=gen_dataloader(args)
    for loader,ids in zip((train,test),(order[:boundary],order[boundary:])):
        if not np.allclose(loader.dataset.tensors[0].numpy(),normalized[ids],rtol=2e-6,atol=2e-6):
            raise ValueError('Full original normalization/split differs from independent data audit')
    iterator=iter(train);workers=iterator._workers;first=next(iterator)[0]
    worker_pids=[p.pid for p in workers];rows=len(first)
    for data in iterator:rows+=len(data[0])
    if (rows!=len(train.dataset) or len(workers)!=4 or len(set(worker_pids))!=4
            or any(p.exitcode!=0 for p in workers) or len(first)!=c['batch_size']):
        raise ValueError('Full four-worker original loader did not complete naturally')
    model_class=(__import__('models.spectral_flow.model',fromlist=['SpectralFlow']).SpectralFlow
                 if config['method']=='spectral_flow' else
                 __import__('models.imagentime.model',fromlist=['ImagenTime']).ImagenTime)
    model=model_class(args=args,device=args.device).to(args.device)
    if config['method']=='imagentime' and args.use_stft:model.init_stft_embedder(train)
    model.epoch=0
    optimizer=original_optimizer(ROOT/config['training_entry'],model,args,torch)
    optimizer.zero_grad();x=model.ts_to_img(first.to(args.device));loss=model.loss_fn(x)
    loss=loss[0] if isinstance(loss,tuple) else loss
    if not torch.isfinite(loss).all():raise ValueError('Nonfinite full original loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();model.on_train_batch_end()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite original optimizer update')
    model.eval();reference=test.dataset.tensors[0].numpy()
    with torch.no_grad(),model.ema_scope():
        if config['method']=='spectral_flow':sample=model.sampling(sampling_number=len(reference))
        else:
            from models.imagentime.sampler import DiffusionProcess
            sample=DiffusionProcess(args,model.net,(args.input_channels,args.img_resolution,args.img_resolution)).sampling(sampling_number=len(reference))
        generated=model.img_to_ts(sample).detach().cpu().numpy()
    if generated.shape!=reference.shape or not np.isfinite(generated).all():raise ValueError('Full native generation shape differs')
    from metrics import evaluate_model_uncond
    scores=evaluate_model_uncond(reference,generated,args)
    if not all(math.isfinite(float(v)) for v in scores.values()):raise ValueError('Nonfinite full native evaluator')
    # Capacity measurements after one update are deliberately not benchmark scores.
    proof=dict(passed=True,diagnostic_only=True,formal_benchmark_result=False,job_id=job['id'],
        queue_sha256=sha(ROOT/queue['queue_path']),method=config['method'],dataset=config['dataset'],
        full_original_config_sha256=sha(ROOT/job['model_config']),full_model_parameters=sum(p.numel() for p in model.parameters()),
        training_batch_shape=list(first.shape),test_shape=list(reference.shape),
        optimizer=type(optimizer).__name__,native_full_generation_and_ten_repeat_evaluator_passed=True,
        full_loader_rows=rows,num_workers=4,worker_pids=worker_pids,worker_exitcodes=[p.exitcode for p in workers],
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),benchmark_scores_omitted=True)
    write(output,proof)


def run_job(queue,job):
    artifact,workspace,config=prepare_workspace(job)
    output=ROOT/job['output_directory'];output.mkdir(parents=True,exist_ok=False)
    for stage in ('training','evaluation'):
        c=contract(queue,job,stage,artifact,workspace,config)
        if stage=='evaluation':
            saved=read(artifact/'training_receipt.json')
            if saved['status']!='completed' or sha(saved['selected_checkpoint'])!=saved['selected_checkpoint_sha256']:
                raise ValueError('Full original marginal-selected checkpoint required before evaluation')
        path=artifact/(stage+'_contract.json');write(path,c)
        command=[str(ROOT/queue['python']),'-X','utf8','-u','-m','scripts.flow_matching.spectral_table3_bootstrap_v1','--contract',str(path)]
        with (output/(stage+'.stdout.log')).open('xb') as out,(output/(stage+'.stderr.log')).open('xb') as err:
            subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    training=read(artifact/'training_receipt.json');evaluation=read(artifact/'evaluation_receipt.json')
    if training['status']!='completed' or evaluation['status']!='completed':raise ValueError('Full original stages incomplete')
    result=dict(status='completed',id=job['id'],queue_sha256=sha(ROOT/queue['queue_path']),
        model_config_sha256=sha(ROOT/job['model_config']),training_receipt_sha256=sha(artifact/'training_receipt.json'),
        evaluation_receipt_sha256=sha(artifact/'evaluation_receipt.json'),metrics=evaluation['evaluations'][0]['scores'],
        training=training,evaluation=evaluation,author_equivalence_certified=False,
        boundary='Released author protocol: normalized full trajectories; train seed42 and standalone evaluation split seed0; test-set marginal checkpoint selection retained. Metric STD is ten evaluator fits, not ten generator seeds. Full trained-model re-evaluation/claim comparison remains required before formal publication.')
    write(output/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True)
    parser.add_argument('--job-id');parser.add_argument('--audit-data');parser.add_argument('--preflight-output')
    args=parser.parse_args();queue=read(ROOT/args.queue);verify(queue)
    if args.audit_data:data_audit(queue,ROOT/args.audit_data)
    else:
        job=next(j for j in queue['jobs'] if j['id']==args.job_id)
        if args.preflight_output:preflight(queue,job,ROOT/args.preflight_output)
        else:run_job(queue,job)
