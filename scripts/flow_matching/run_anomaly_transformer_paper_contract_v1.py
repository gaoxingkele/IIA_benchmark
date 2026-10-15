"""Full-width reconstruction of the frozen primary-paper numerical contract.

Separate from both pinned source-runtime tracks. Explicit implementation choices
remain disclosed wherever the paper does not define the executable procedure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.flow_matching import run_anomaly_transformer_author_full_v1 as author

ROOT, read, sha, write = author.ROOT, author.read, author.sha, author.write
CONTRACT='configs/reproducibility/fm_anomaly_transformer_primary_paper_protocol.2026-10-15_v1.json'
STREAMS=['all.reconstruction','all.association','all.addition','all.multiplication',
         'layer1.multiplication','layer2.multiplication','layer3.multiplication']


def verify(config, root=ROOT):
    contract=read(root/CONTRACT)
    if config['contract']!=CONTRACT or sha(root/CONTRACT)!=config['contract_sha256']:
        raise ValueError('Primary-paper contract changed')
    if config['model']!=dict(d_model=512,d_ff=512,n_heads=8,e_layers=3,dropout=0.0,window=100):
        raise ValueError('Complete full-width paper reconstruction required')
    expected_training=dict(batch_size=32,maximum_epochs=10,window_stride=100,learning_rate=.0001,
        discrepancy_weight=3,patience=3,kl_log_epsilon=.0001,cpu_threads=2,
        optimizer='Adam',backward_strategy='two_detached_losses_one_accumulated_Adam_update',
        loss_reduction='per_window_Frobenius_squared_and_L1_discrepancy_then_batch_mean',
        learning_rate_schedule='inherited_author_halving_after_epoch_two')
    if config['training']!=expected_training:
        raise ValueError('Frozen primary-paper training budget or explicit numerical choice changed')
    if config['seeds']!=[1103,1104,1105,1106,1107] or len(config['jobs'])!=25:
        raise ValueError('All five real datasets and all five seed slots required')
    expected={(d,s) for d in ['SMD','MSL','SMAP','PSM','SWaT'] for s in config['seeds']}
    if {(j['dataset'],j['seed']) for j in config['jobs']}!=expected:
        raise ValueError('Incomplete main dataset/seed matrix')
    for job in config['jobs']:
        if job['id']!=f"anomaly_transformer__paper_contract_reconstruction__{job['dataset'].lower()}__seed{job['seed']}":
            raise ValueError('Paper reconstruction canonical identity changed')
    if config['score_streams']!=STREAMS or config['original_scope']!=contract['original_scope']:
        raise ValueError('Complete scoring and full project scope must remain')
    limits=config['settings']
    if (limits['minimum_available_memory_bytes']<32*1024**3 or
        limits['minimum_commit_headroom_bytes']<40*1024**3 or
        limits['emergency_commit_headroom_bytes']<16*1024**3):
        raise ValueError('Full resource protection required')
    for pin in config['source_receipts']:
        if sha(root/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen paper reconstruction input changed: '+pin['path'])
    return contract


def head_mean_discrepancy(priors,series,epsilon=.0001,detach=None):
    """Algorithm2: normalize each prior head, mean heads, symmetric KL, mean layers."""
    import torch
    if len(priors)!=len(series) or not priors:
        raise ValueError('Matching nonempty layer lists required')
    layers=[]
    for p,s in zip(priors,series):
        p=p/p.sum(dim=-1,keepdim=True)
        if detach=='prior':p=p.detach()
        elif detach=='series':s=s.detach()
        elif detach is not None:raise ValueError('Unknown stop-gradient phase')
        p,s=p.mean(dim=1),s.mean(dim=1)
        layer=(p*(torch.log(p+epsilon)-torch.log(s+epsilon))+
               s*(torch.log(s+epsilon)-torch.log(p+epsilon))).sum(dim=-1)
        layers.append(layer)
    values=torch.stack(layers,dim=0)
    return values.mean(dim=0),values


def minimax_losses(reconstruction,target,priors,series,weight=3,epsilon=.0001):
    """Equation5 reductions; both detached phases include full reconstruction."""
    rec=(reconstruction-target).square().sum(dim=(-2,-1)).mean()
    minimize,_=head_mean_discrepancy(priors,series,epsilon,detach='series')
    maximize,_=head_mean_discrepancy(priors,series,epsilon,detach='prior')
    return rec+weight*minimize.abs().sum(dim=-1).mean(),rec-weight*maximize.abs().sum(dim=-1).mean(),rec


def score_components(reconstruction,target,priors,series,epsilon=.0001):
    """Algorithm3 feature mean, no source temperature50 or layer-sum multiplier."""
    import torch
    discrepancy,layers=head_mean_discrepancy(priors,series,epsilon)
    rec=(reconstruction-target).square().mean(dim=-1)
    association=torch.softmax(-discrepancy,dim=-1)
    scores={'all.reconstruction':rec,'all.association':association,
            'all.addition':rec+association,'all.multiplication':rec*association}
    for i,layer in enumerate(layers):
        scores[f'layer{i+1}.multiplication']=rec*torch.softmax(-layer,dim=-1)
    return scores,rec,layers


def prepare_data(config,dataset,contract):
    """Build immutable derived views; never alter source arrays or CSVs."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    folder=Path(config['data_cache_root'])/dataset
    manifest=folder/'manifest.json'
    if manifest.exists():
        cached=read(manifest)
        if cached['contract_sha256']!=config['contract_sha256']:
            raise ValueError('Data cache belongs to another paper contract')
        for record in cached['artifacts']:
            if sha(folder/record['name'])!=record['sha256']:
                raise ValueError('Derived data cache changed')
        return folder,cached
    if folder.exists():raise ValueError('Preserve incomplete data preparation for review')
    staging=folder.with_name(folder.name+'.preparing')
    staging.mkdir(parents=True,exist_ok=False)
    ds=read(ROOT/f'configs/datasets/mtsad_{dataset.lower()}.json')
    source=(ROOT/config['data_root']/ds['directory']).resolve()
    inputs=[dict(path=str(source/name),sha256=sha(source/name)) for name in ds['files'].values()]
    if ds['loader']=='npy':
        train=np.load(source/ds['files']['train'],mmap_mode='r')
        test=np.load(source/ds['files']['test'],mmap_mode='r')
        labels=np.load(source/ds['files']['label'],mmap_mode='r').reshape(-1)
    elif ds['loader']=='psm_csv':
        train=pd.read_csv(source/ds['files']['train']).values[:,1:].astype(np.float64)
        test=pd.read_csv(source/ds['files']['test']).values[:,1:].astype(np.float64)
        # Inherit the explicit author PSM NaN-to-zero preprocessing only.
        train=np.where(np.isnan(train),0,train);test=np.where(np.isnan(test),0,test)
        labels=pd.read_csv(source/ds['files']['label']).values[:,1:].reshape(-1)
    elif ds['loader']=='swat_csv':
        train=pd.read_csv(source/ds['files']['train']).values[:,:-1].astype(np.float64)
        values=pd.read_csv(source/ds['files']['test']).values
        test=values[:,:-1].astype(np.float64);labels=values[:,-1].astype(np.float64)
    else:raise ValueError('Unsupported primary real-data loader')
    split=contract['reported_table13_splits'][dataset]
    if (list(train.shape)!=[split['train']+split['validation'],split['features']] or
        list(test.shape)!=[split['test'],split['features']] or len(labels)!=split['test']):
        raise ValueError('Complete Table13 row/feature counts required')
    if not np.isfinite(train).all() or not np.isfinite(test).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid original numerical values or anomaly labels')
    cutoff=split['train'];scaler=StandardScaler().fit(train[:cutoff])
    for name,values in [('train',train[:cutoff]),('validation',train[cutoff:]),('test',test)]:
        np.save(staging/(name+'.npy'),scaler.transform(values).astype(np.float32),allow_pickle=False)
    np.save(staging/'labels.npy',np.asarray(labels,dtype=np.int8),allow_pickle=False)
    write(staging/'scaler.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        fitted_rows=int(scaler.n_samples_seen_),fit_on_training_only=True))
    artifacts=[dict(name=p.name,sha256=sha(p)) for p in sorted(staging.iterdir()) if p.is_file()]
    cache=dict(dataset=dataset,contract_sha256=config['contract_sha256'],split=split,
        source_data=inputs,partition='contiguous_first_train_rows_then_independent_validation_tail',
        normalization='StandardScaler fitted only on the stated training subset',
        exact_historical_partition_identity_certified=False,artifacts=artifacts)
    write(staging/'manifest.json',cache);os.rename(staging,folder)
    return folder,cache


def evaluation(labels,scores,validation_scores,r_percent,fixed_delta,roc_ratios):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    from iia_benchmark.evaluation.mtsad_metrics import pointwise_metrics,point_adjust
    labels=np.asarray(labels);scores=np.asarray(scores);validation_scores=np.asarray(validation_scores)
    if labels.shape!=scores.shape or not validation_scores.size:
        raise ValueError('Full covered test/validation score arrays required')
    def at(threshold):
        predicted=scores>threshold
        return dict(threshold=float(threshold),no_PA=pointwise_metrics(labels,predicted),
                    paper_PA=pointwise_metrics(labels,point_adjust(labels,predicted)))
    ratio=at(np.percentile(validation_scores,100-r_percent))
    fixed=at(fixed_delta)
    points=[]
    for r in roc_ratios:
        record=at(np.percentile(validation_scores,100-r))
        for key in ['no_PA','paper_PA']:
            m=record[key];negative=int(np.sum(labels==0));positive=int(np.sum(labels==1))
            record[key]['false_positive_rate']=m['false_positives']/negative if negative else None
            record[key]['true_positive_rate']=m['true_positives']/positive if positive else None
        points.append(dict(r_percent=r,**record))
    both=len(np.unique(labels))==2
    return dict(validation_ratio=ratio,fixed_delta=fixed,
        continuous_unadjusted_score_AUROC=float(roc_auc_score(labels,scores)) if both else None,
        continuous_score_AP=float(average_precision_score(labels,scores)) if both else None,
        seven_validation_calibrated_ROC_points=points,
        seven_point_curve_integration_not_paper_certified=True,
        test_features_or_labels_used_for_calibration=False)


def run_worker(config,job,config_path):
    contract=verify(config)
    import random
    import numpy as np
    import torch
    from torch.utils.data import Dataset,DataLoader
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    torch.set_default_device('cpu')
    base=Path(config['artifact_root'])/job['id'];base.mkdir(parents=True,exist_ok=False)
    folder,audit=prepare_data(config,job['dataset'],contract)
    workspace=base/'source'
    shutil.copytree(ROOT/config['source_directory'],workspace,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    attention=workspace/'model/attn.py';original=sha(attention)
    attention.write_text(author.patched('model/attn.py',attention.read_text(encoding='utf-8')),encoding='utf-8')
    write(base/'source_patch.json',dict(path='model/attn.py',original_sha256=original,runtime_sha256=sha(attention),
                                      numerical_algorithm_changes='none; CPU device compatibility only'))
    sys.path.insert(0,str(workspace))
    from model.AnomalyTransformer import AnomalyTransformer
    model=AnomalyTransformer(win_size=100,enc_in=audit['split']['features'],c_out=audit['split']['features'],
        d_model=512,n_heads=8,e_layers=3,d_ff=512,dropout=0.0)
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('Actual paper reconstruction parameters must be on CPU')
    arrays={name:np.load(folder/(name+'.npy'),mmap_mode='r') for name in ['train','validation','test']}
    class Windows(Dataset):
        def __init__(self,data):self.data=data
        def __len__(self):return len(self.data)//100
        def __getitem__(self,index):return np.array(self.data[index*100:(index+1)*100],copy=True)
    loaders={k:DataLoader(Windows(v),batch_size=32,shuffle=k=='train',num_workers=0) for k,v in arrays.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    epochs=[];best=None;patience=3;updates=0;checkpoint=base/'model.pt'
    start=time.perf_counter()
    for epoch in range(1,11):
        model.train();sums=[0.,0.,0.];count=0;steps=0
        for batch in loaders['train']:
            optimizer.zero_grad();reconstruction,series,priors,_=model(batch)
            minimum,maximum,rec=minimax_losses(reconstruction,batch,priors,series)
            if not all(torch.isfinite(v) for v in [minimum,maximum,rec]):
                raise ValueError('Nonfinite mathematical training loss; preserve attempt without retry')
            minimum.backward(retain_graph=True);maximum.backward();optimizer.step()
            steps+=1;updates+=1;count+=len(batch)
            for i,v in enumerate([minimum,maximum,rec]):sums[i]+=float(v.detach())*len(batch)
        model.eval();validation_sums=[0.,0.];n=0
        with torch.no_grad():
            for batch in loaders['validation']:
                reconstruction,series,priors,_=model(batch)
                minimum,maximum,_=minimax_losses(reconstruction,batch,priors,series)
                if not torch.isfinite(minimum) or not torch.isfinite(maximum):raise ValueError('Nonfinite validation loss')
                n+=len(batch);validation_sums[0]+=float(minimum)*len(batch);validation_sums[1]+=float(maximum)*len(batch)
        validation=[v/n for v in validation_sums]
        improved=best is None or (validation[0]<=best[0] and validation[1]<=best[1])
        if improved:
            best=validation;patience=3;torch.save(model.state_dict(),checkpoint)
        else:patience-=1
        epochs.append(dict(epoch=epoch,steps=steps,train_windows=count,train_losses=[v/count for v in sums],
                           validation_windows=n,validation_losses=validation,checkpoint_saved=improved,
                           remaining_patience=patience,learning_rate=optimizer.param_groups[0]['lr']))
        write(base/'epochs.json',epochs)
        if patience==0:break
        for group in optimizer.param_groups:group['lr']=.0001*(.5**(epoch-1))
    training_seconds=time.perf_counter()-start
    if updates!=sum(e['steps'] for e in epochs) or any(e['steps']!=math.ceil(len(loaders['train'].dataset)/32) for e in epochs):
        raise ValueError('Original full non-overlapping training coverage missing')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    all_scores={};score_artifacts=[];start=time.perf_counter()
    for split,data in arrays.items():
        loader=DataLoader(Windows(data),batch_size=32,shuffle=False,num_workers=0)
        buffers={k:[] for k in STREAMS};rec_buffers=[];discrepancy_buffers=[]
        with torch.no_grad():
            for batch in loader:
                reconstruction,series,priors,_=model(batch)
                scores,rec,layers=score_components(reconstruction,batch,priors,series)
                for name,value in scores.items():buffers[name].append(value.numpy().reshape(-1))
                rec_buffers.append(rec.numpy().reshape(-1))
                discrepancy_buffers.append(layers.numpy().transpose(1,2,0).reshape(-1,3))
        full={k:np.concatenate(v) for k,v in buffers.items()}
        full['component.reconstruction']=np.concatenate(rec_buffers)
        full['component.layer_discrepancies']=np.concatenate(discrepancy_buffers,axis=0)
        if not all(np.isfinite(v).all() for v in full.values()):raise ValueError('Nonfinite full score components')
        expected=len(data)//100*100
        if any(v.shape[0]!=expected for v in full.values()):raise ValueError('Incomplete split score coverage')
        path=base/(split+'_scores.npz');np.savez(path,**full);score_artifacts.append(path);all_scores[split]=full
    labels=np.asarray(np.load(folder/'labels.npy',mmap_mode='r')[:len(all_scores['test']['all.multiplication'])])
    np.save(base/'covered_test_labels.npy',labels,allow_pickle=False)
    metrics={name:evaluation(labels,all_scores['test'][name],all_scores['validation'][name],
        contract['paper_evaluation']['r_percent'][job['dataset']],
        contract['paper_evaluation']['fixed_delta'][job['dataset']],
        contract['paper_evaluation']['ROC_predefined_r_percent']) for name in STREAMS}
    score_seconds=time.perf_counter()-start
    artifacts=[checkpoint,base/'epochs.json',base/'source_patch.json',base/'covered_test_labels.npy',*score_artifacts]
    write(base/'result.json',dict(status='completed',job=job,configuration_sha256=sha(config_path),
        parameter_count=sum(p.numel() for p in model.parameters()),epochs=epochs,optimizer_updates=updates,
        actual_backward_calls=updates*2,training_seconds=training_seconds,score_seconds=score_seconds,
        data_audit=audit,data_manifest_sha256=sha(folder/'manifest.json'),
        covered_points={k:len(all_scores[k]['all.multiplication']) for k in arrays},
        omitted_tail_points={k:len(arrays[k])%100 for k in arrays},metrics=metrics,
        checkpoint_selection_uses_test=False,threshold_calibration_uses_test=False,
        paper_algorithm_equivalence_certified=False,strict_TAB_result=False,
        independent_full_checkpoint_replay_complete=False,
        explicit_reconstruction_choices=config['explicit_reconstruction_choices'],
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in artifacts],boundary=config['boundary']))


def check_result(config,job):
    result=read(Path(config['artifact_root'])/job['id']/'result.json')
    if result['status']!='completed' or result['job']!=job:raise ValueError('Wrong completed experiment')
    for artifact in result['artifacts']:
        if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Completed artifact changed')
    return result


def controller(config,config_path):
    from scripts.flow_matching.heavy_resources import guarded_wait
    from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
    from scripts.flow_matching.run_tsad_queue import exclusive_lock
    import psutil
    verify(config);state=ROOT/config['state_root'];lock=exclusive_lock(state/'controller.lock')
    if lock is None:raise RuntimeError('Paper-contract controller already has a live owner')
    common=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),selected_main_jobs=25,
                original_scope=config['original_scope'],all_original_experiments_complete=False)
    def update(value):write(state/'status.json',dict(common,**value))
    for job in config['jobs']:
        base=Path(config['artifact_root'])/job['id'];receipt=state/'receipts'/(job['id']+'.json')
        if (base/'result.json').exists():check_result(config,job);continue
        if base.exists() or receipt.exists():
            update(dict(state='preserved_prior_attempt_requires_review',active_pid=None,job=job['id']));continue
        def waiting(value):update(dict(value,job=job['id'],active_pid=None))
        with resource_slot(ROOT/config['resource_lock'],config['settings'],waiting,poll_seconds=10):
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            with (state/(job['id']+'.log')).open('xb') as log:
                process=subprocess.Popen([config['python'],'-B','-X','utf8','-u','-m',
                    'scripts.flow_matching.run_anomaly_transformer_paper_contract_v1','--config',config_path,
                    '--worker','--job-id',job['id']],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                update(dict(state='running_full_paper_contract_reconstruction',job=job['id'],active_pid=process.pid,
                            active_create_time=psutil.Process(process.pid).create_time()))
                code,peaks=guarded_wait(process,config['settings'],poll_seconds=2)
            passed=code==0 and not peaks['resource_guard_aborted']
            if passed:check_result(config,job)
            write(receipt,dict(job=job['id'],passed=passed,exit_code=code,**peaks))
    completed=[]
    for job in config['jobs']:
        receipt=state/'receipts'/(job['id']+'.json')
        if receipt.exists() and read(receipt).get('passed'):check_result(config,job);completed.append(job['id'])
    update(dict(state='main_reconstruction_attempts_terminal_full_project_and_independent_reviews_remain',
                active_pid=None,completed_main_reconstructions=completed));lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True,main_dataset_seed_slots=25,full_score_streams=7,
                              torch_imported='torch' in sys.modules,all_original_experiments_complete=False)))
    elif args.worker:run_worker(config,next(j for j in config['jobs'] if j['id']==args.job_id),ROOT/args.config)
    else:controller(config,args.config)


if __name__=='__main__':main()
