"""Full Tiny/ICM source-loop forecast fine-tuning,after real PILE checkpoints.

The source five-epoch optimizer/AMP/scheduler/validation-selection body stays
intact except a read-only gradient audit. Local disk writers replace services
and redundant prediction/history lists;all actual windows and predictions stay.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional, Union
from scripts.flow_matching import moment_icm_svm_classification_v1 as pretrained
from scripts.flow_matching import prepare_moment_icm_forecast_evaluation_inputs_v1 as inputs
from scripts.flow_matching import moment_icm_training_artifacts_v1 as serialization

ROOT,read,sha,write=inputs.ROOT,inputs.read,inputs.sha,inputs.write
SOURCE='experiments/runs/mtsad_protocol_audit/sources/moment_research/original'
BRANCHES={'independent_head':dict(mixer='independent',train_beta=False),
          'icm_beta_frozen':dict(mixer='icm',train_beta=False),
          'icm_beta_finetuned':dict(mixer='icm',train_beta=True)}


def ready_data(dataset,horizon):
    c=read(ROOT/inputs.CONFIG);p=Path(c['artifact_root'])/inputs.REPORT;g=ROOT/c['state_root']/'guard_receipt.json'
    if not p.exists() or not g.exists():return None
    proof,guard=read(p),read(g)
    if (not proof['passed'] or proof['configuration_sha256']!=sha(ROOT/inputs.CONFIG)
            or proof['datasets']!=6 or proof['horizon_dataset_pairs']!=18 or proof['role_horizon_groups']!=54
            or not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(p)):
        raise ValueError('Complete actual6/18/54 input proof and successful original data guard required')
    row=next(r for r in proof['source_records'] if r['job']['dataset']==dataset)
    if sha(ROOT/row['job']['path'])!=row['job']['sha256']:raise ValueError('Original forecast CSV changed')
    roles={}
    for role,facts in row['roles'].items():
        spec=next(h for h in facts['windows_by_horizon'] if h['horizon']==horizon)
        roles[role]=dict(directory=str(Path(c['artifact_root'])/dataset/role),facts=facts,window=spec)
    return dict(dataset=dataset,horizon=horizon,roles=roles,input_configuration_sha256=sha(ROOT/inputs.CONFIG),
                proof_sha256=sha(p),guard_sha256=sha(g),raw=row['job'])


class WindowDataset:
    """Spawn-safe readonly source TC windows;no expanded-window tensor or RNG."""
    def __init__(self,binding,role):
        r=binding['roles'][role];self.directory=r['directory'];self.horizon=binding['horizon'];self.role=role
        self.n=r['window']['windows'];self.channels=r['facts']['channels'];self._matrix=None;self._mask=None
    def __len__(self):return self.n
    def __getstate__(self):return dict(self.__dict__,_matrix=None,_mask=None)
    def __getitem__(self,index):
        import numpy as np
        index=int(index)
        if not 0<=index<self.n:raise IndexError('Every original forecast index is bounded')
        if self._matrix is None:
            self._matrix=np.load(Path(self.directory)/'values_TC.npy',mmap_mode='r',allow_pickle=False)
            self._mask=np.load(Path(self.directory)/'input_mask_window.npy',mmap_mode='r',allow_pickle=False)
        return SimpleNamespace(timeseries=self._matrix[index:index+256].T.copy(),
            forecast=self._matrix[index+256:index+256+self.horizon].T.copy(),input_mask=self._mask.copy(),
            name=self.role,window_index=index)


def collate_forecast(examples):
    """Exact source stacking,with an extra readonly origin-index identity."""
    import numpy as np
    import torch
    return SimpleNamespace(timeseries=torch.stack([torch.from_numpy(e.timeseries) for e in examples]),
        input_mask=torch.stack([torch.from_numpy(e.input_mask) for e in examples]),
        forecast=torch.stack([torch.from_numpy(e.forecast) for e in examples]),
        name=np.asarray([e.name for e in examples]),window_indices=np.asarray([e.window_index for e in examples],dtype=np.int64))


class RecordedLoader:
    def __init__(self,loader,root):
        self.loader=loader;self.dataset=loader.dataset;self.root=Path(root)
        self.records=[];self.phase='default'
    def __len__(self):return len(self.loader)
    def __iter__(self):
        import numpy as np
        chunks=[];batches=0;last=0;phase='train' if self.dataset.role=='train' and self.phase=='default' else 'validation'
        for batch in self.loader:
            chunks.append(batch.window_indices.copy());batches+=1;last=len(batch.window_indices)
            yield batch
        order=np.concatenate(chunks)
        if batches!=len(self.loader) or not np.array_equal(np.sort(order),np.arange(len(self.dataset))):
            raise ValueError('Every original forecast origin exactly once,including partial batch required')
        path=self.root/(self.dataset.role+'_'+str(len(self.records)+1)+'.npy')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:np.save(stream,order,allow_pickle=False)
        self.records.append(dict(role=self.dataset.role,instances=len(order),batches=batches,last_batch_items=last,
            order_path=str(path),order_sha256=sha(path),every_origin_once=True,drop_last=False,phase=phase))


def make_loaders(binding,root,*,num_workers=5):
    import torch
    return {role:RecordedLoader(torch.utils.data.DataLoader(WindowDataset(binding,role),batch_size=64 if role=='train' else 256,
                shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=False,collate_fn=collate_forecast),root)
            for role in ['train','val','test']}


def make_forecaster(core,horizon,branch,source_root):
    import torch
    if branch not in BRANCHES or horizon not in [96,192,384]:raise ValueError('Exact three branches and reported horizons required')
    choice=BRANCHES[branch]
    if core.seq_len!=256 or core.patch_len!=8 or core.d_model!=256 or bool(core.icm_layers)!=(choice['mixer']=='icm'):
        raise ValueError('Exact actual pretrained Tiny backbone and branch required')
    if core.patch_embedding.add_positional_embedding:
        raise ValueError('Frozen local core and source forecast positional policy must remain False')
    if core.encoder.gradient_checkpointing:
        raise ValueError('Frozen embeddings with beta gradients require checkpointing disabled here')
    path=Path(source_root)/'moment/models/moment.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='ForecastingHead')
    ns=dict(torch=torch,nn=torch.nn);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    core.requires_grad_(False);core.task='forecast';core.head=ns['ForecastingHead'](8192,horizon,head_dropout=.1)
    for layer in core.icm_layers:layer.beta.requires_grad_(choice['train_beta'])
    betas=sum(layer.beta.numel() for layer in core.icm_layers)
    if betas!=(16 if choice['mixer']=='icm' else 0):raise ValueError('All16 original ICM beta scalars required')
    class Adapter(torch.nn.Module):
        def __init__(self):super().__init__();self.core=core
        def forward(self,x_enc,input_mask,mask=None):
            value=self.core(x_enc,input_mask=input_mask,reconstruction_mask=input_mask)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite forecast;preserve without retry')
            return SimpleNamespace(forecast=value)
    model=Adapter()
    expected=horizon*(8192+1)+(16 if choice['train_beta'] else 0)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=expected:
        raise ValueError('Only the full forecast head and declared beta scalars may train')
    return model


class LocalLogger:
    def __init__(self,root,name):self.root=Path(root);self.name=name;self.id=name;self.steps=0
    def log(self,value):
        if 'step_train_loss' in value:
            if not math.isfinite(float(value['step_train_loss'])):raise FloatingPointError('Finite original training loss required')
            self.steps+=1
        def convert(x):
            if hasattr(x,'item'):return x.item()
            raise TypeError(type(x).__name__)
        with (self.root/'source_log.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(value,default=convert,allow_nan=False)+'\n')


class DiskBuffer:
    def __init__(self,path,shape):
        import numpy as np
        self.path=Path(path);self.shape=shape;self.count=0
        self.array=np.lib.format.open_memmap(self.path,mode='w+',dtype=np.float32,shape=shape)
    def append(self,value):
        import numpy as np
        if value.dtype!=np.float32 or value.shape[1:]!=self.shape[1:] or not np.isfinite(value).all():
            raise ValueError('Every full float32 forecast/target coordinate required')
        end=self.count+len(value)
        if end>self.shape[0]:raise ValueError('No repeated or extra forecast origins')
        self.array[self.count:end]=value;self.count=end
    def finish(self):
        if self.count!=self.shape[0]:raise ValueError('All original forecast origins must be written')
        self.array.flush();return self.array


class HistoryDigest:
    def __init__(self):self.count=0;self.digest=hashlib.sha256()
    def append(self,value):self.count+=len(value);self.digest.update(value.tobytes())
    def finish(self):return None


def source_symbols(source_root):
    """Exact private numerical source methods,without global services/imports."""
    import dataclasses
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import torch
    body=read(ROOT/pretrained.pretrain.masks.TRAIN_BODY)
    _,ns=pretrained.pretrain.train.load_source_symbols(source_root,body['original_numerical_source_sha256'])
    source_root=Path(source_root)
    utils=ast.parse((source_root/'moment/utils/utils.py').read_text(encoding='utf-8'))
    reduce=next(n for n in utils.body if isinstance(n,ast.FunctionDef) and n.name=='_reduce')
    metrics=ast.parse((source_root/'moment/utils/forecasting_metrics.py').read_text(encoding='utf-8'))
    nodes=[n for n in metrics.body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.ClassDef) and n.name=='ForecastingMetrics')]
    module=ModuleType('_isolated_icm_forecast_metrics_v1');sys.modules[module.__name__]=module
    module.__dict__.update(np=np,npt=npt,Optional=Optional,Union=Union,dataclass=dataclasses.dataclass)
    exec(compile(ast.Module(body=[reduce,*nodes],type_ignores=[]),str(source_root/'moment/utils/forecasting_metrics.py'),'exec'),module.__dict__)
    ns.update(pd=pd,np=np,torch=torch,nn=torch.nn,os=os,tqdm=lambda iterable,**kw:iterable,
        get_forecasting_metrics=module.get_forecasting_metrics,
        MetricsStore=lambda **kw:SimpleNamespace(**kw),
        make_dir_if_not_exists=lambda path,**kw:Path(path).mkdir(parents=True,exist_ok=True),
        wandb=SimpleNamespace(Table=lambda **kw:kw))
    task=next(n for n in ast.parse((source_root/'moment/tasks/forecast_finetune.py').read_text(encoding='utf-8')).body
              if isinstance(n,ast.ClassDef) and n.name=='ForecastFinetuning')
    methods=[deepcopy(n) for n in task.body if isinstance(n,ast.FunctionDef) and n.name in
             ['train','validation','evaluate_model','evaluate_and_log','compute_forecasting_metrics']]
    base=next(n for n in ast.parse((source_root/'moment/tasks/base.py').read_text(encoding='utf-8')).body
              if isinstance(n,ast.ClassDef) and n.name=='Tasks')
    methods.append(deepcopy(next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_model')))
    if len(methods)!=5:raise ValueError('All original five full source forecasting methods required')
    training=next(n for n in methods if n.name=='train');changed=0
    for loop in [n for n in ast.walk(training) if isinstance(n,ast.For)]:
        new=[]
        for statement in loop.body:
            new.append(statement)
            if ast.unparse(statement)=='nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)':
                new.append(ast.parse('self.audit_gradients(loss)').body[0]);changed+=1
        loop.body=new
    if changed!=1:raise ValueError('Only one read-only post-clip audit insertion required')
    validation=next(n for n in methods if n.name=='validation');assign=validation.body[0]
    if ast.unparse(assign)!='trues, preds, histories, losses = ([], [], [], [])':
        # ast.unparse formats unpacked-list tuples differently across Python.
        if not isinstance(assign,ast.Assign) or not isinstance(assign.value,ast.Tuple) or len(assign.value.elts)!=4:
            raise ValueError('Exact original four validation containers required')
    assign.value=ast.parse('self.validation_buffers(data_loader, return_preds)',mode='eval').body
    return ns,methods


class SourceForecastTask:
    def __init__(self,model,loaders,args,output,source_root,job,device,pretrained_binding):
        import numpy as np
        import torch
        self.model=model;self.args=args;self.device=torch.device(device);self.job=job
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.logger=LocalLogger(self.output,job['id']);self.loaders=loaders
        self.train_dataloader=loaders['train'];self.val_dataloader=loaders['val'];self.test_dataloader=loaders['test']
        self.loaded_pretrained_binding=pretrained_binding;self.audited_updates=0;self.checkpoints=[];self.bundles=[]
        self.frozen={name:p.detach().cpu().clone() for name,p in model.named_parameters() if not p.requires_grad}
        ns,methods=source_symbols(source_root)
        original_concat=np.concatenate
        def concat(value,*args,**kwargs):
            return value.finish() if isinstance(value,(DiskBuffer,HistoryDigest)) else original_concat(value,*args,**kwargs)
        ns['np']=SimpleNamespace(**dict(vars(np),concatenate=concat))
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<pinned-full-source-forecast-task>','exec'),ns)
        for name in ['train','validation','evaluate_model','evaluate_and_log','compute_forecasting_metrics',
                     '_select_optimizer','_select_criterion','_init_lr_scheduler']:
            setattr(self,name,MethodType(ns[name],self))
        original_validation=self.validation
        def recorded_validation(task,loader,return_preds=False):
            previous=len(task.bundles);old_phase=loader.phase;loader.phase='validation'
            try:result=original_validation(loader,return_preds)
            finally:loader.phase=old_phase
            if return_preds:
                if len(task.bundles)!=previous+1:raise ValueError('One complete forecast bundle per source prediction call required')
                task.bundles[-1]['order']=dict(loader.records[-1])
            return result
        self.validation=MethodType(recorded_validation,self)
        self.criterion=self._select_criterion(loss_type='mse',reduction='mean')
    def setup_logger(self):return self.logger
    def _create_results_dir(self,**kw):
        path=self.output/'source_results';path.mkdir(exist_ok=False);return str(path)
    def load_pretrained_moment(self):
        if not self.loaded_pretrained_binding:raise ValueError('Actual strict loaded pretrained checkpoint required')
        return True
    def validation_buffers(self,loader,record):
        if not record:return [],[],[],[]
        directory=self.output/'predictions'/('call'+str(len(self.bundles)+1));directory.mkdir(parents=True,exist_ok=False)
        shape=(len(loader.dataset),loader.dataset.channels,self.args.forecast_horizon)
        truth=DiskBuffer(directory/'truth.npy',shape);pred=DiskBuffer(directory/'prediction.npy',shape);history=HistoryDigest()
        self.bundles.append(dict(directory=str(directory),loader=loader,truth=truth,prediction=pred,history=history))
        return truth,pred,history,[]
    def audit_gradients(self,loss):
        import torch
        if not torch.isfinite(loss).all():raise FloatingPointError('Finite complete forecast loss required')
        for p in self.model.parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise FloatingPointError('Every declared head/beta gradient must be finite')
            if not p.requires_grad and p.grad is not None:raise ValueError('Frozen backbone must not get gradients')
        self.audited_updates+=1
    def save_model_and_alert(self,opt_steps):
        self.checkpoints.append(self.save_state('best_'+str(self.logger.steps)))
    def save_state(self,name):
        import torch
        rng=serialization.capture_rng()
        active={k:p for k,p in self.model.named_parameters() if p.requires_grad}
        state,count=serialization.audit_optimizer(self.optimizer,self.logger.steps)
        if count!=len(active):raise ValueError('Every trainable head/beta optimizer state required')
        payload=dict(job=self.job,steps=self.logger.steps,model_state_dict=self.model.state_dict(),optimizer_state_dict=state,
            scaler_state_dict=self.scaler.state_dict(),scheduler_state_dict=self.lr_scheduler.state_dict(),rng=rng)
        path=self.output/(name+'.pt')
        with path.open('xb') as stream:torch.save(payload,stream)
        restored=torch.load(path,map_location='cpu',weights_only=False)
        if not serialization.exact(payload,restored) or not serialization.exact(rng,serialization.capture_rng()):
            raise ValueError('Full saved state must be exact and IO must not consume random draws')
        return dict(path=str(path),sha256=sha(path),optimizer_steps=self.logger.steps)
    def save_results(self,frame,path,steps):frame.to_csv(Path(path)/('metrics_'+str(steps)+'.csv'))
    def finalize(self):
        import torch
        expected=5*len(self.train_dataloader)
        trained_iterations=[r for r in self.train_dataloader.records if r['phase']=='train']
        if self.args.max_epoch!=5 or self.logger.steps!=expected or self.audited_updates!=expected or len(trained_iterations)!=5:
            raise ValueError('All five full epochs and every audited optimizer update required')
        for name,p in self.model.named_parameters():
            if name in self.frozen and not torch.equal(p.detach().cpu(),self.frozen[name]):raise ValueError('Frozen backbone/beta changed')
        if not self.checkpoints or len(self.bundles)!=len(self.checkpoints):raise ValueError('Every selected checkpoint needs full TEST prediction bundle')
        final=self.save_state('full_five_epoch_final')
        bundles=[]
        for b in self.bundles:
            truth,pred=b['truth'],b['prediction'];truth.finish();pred.finish()
            # Each extra compute_forecasting_metrics call creates one TEST
            # iteration;capture the correct order immediately in the wrapper.
            bundles.append(dict(directory=b['directory'],truth_sha256=sha(truth.path),prediction_sha256=sha(pred.path),
                shape=list(truth.shape),history_digest=b['history'].digest.hexdigest(),order=b['order']))
        return dict(passed=True,job=self.job,full_epochs=5,full_optimizer_updates=expected,final_checkpoint=final,
            selected_best_checkpoint=self.checkpoints[-1],selected_TEST_bundle=bundles[-1],all_best_checkpoints=self.checkpoints,
            all_prediction_bundles=bundles,loader_iterations={k:v.records for k,v in self.loaders.items()},
            frozen_parameters_exactly_unchanged=True,all_head_beta_gradients_audited=True,
            historical_paper_equivalence_certified=False,all_original_experiments_complete=False)


def verify_input_arrays(binding):
    for row in binding['roles'].values():
        for key,pin in row['facts']['files'].items():
            if sha(Path(row['directory'])/(key+'.npy'))!=pin['sha256']:
                raise ValueError('Complete immutable downstream input arrays changed')


def independent_metrics(prediction,truth,order,block=640,seed=10013):
    """All coordinates,then conditional circular origin-block uncertainty."""
    import numpy as np
    if prediction.shape!=truth.shape or prediction.ndim!=3 or len(order)!=len(truth):
        raise ValueError('All original origin/channel/horizon coordinates required')
    n=len(order)
    if n<=0:raise ValueError('At least one complete original forecast origin required')
    if not np.array_equal(np.sort(order),np.arange(n)):raise ValueError('All forecast origins exactly once required')
    mae=np.empty(n,dtype=np.float64);mse=np.empty(n,dtype=np.float64)
    for start in range(0,n,64):
        p=prediction[start:start+64].astype(np.float64);y=truth[start:start+64].astype(np.float64)
        if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Every forecast coordinate must be finite')
        difference=p-y;indices=order[start:start+len(p)]
        mae[indices]=np.abs(difference).mean((1,2));mse[indices]=np.square(difference).mean((1,2))
    rng=np.random.RandomState(seed);replicates=[];size=min(block,n);blocks=(n+size-1)//size
    for _ in range(200):
        selection=((rng.randint(0,n,size=blocks)[:,None]+np.arange(size))%n).reshape(-1)[:n]
        replicates.append((float(mae[selection].mean()),float(mse[selection].mean())))
    intervals=np.quantile(np.asarray(replicates),[.025,.975],axis=0)
    return dict(MAE=float(mae.mean()),MSE=float(mse.mean()),RMSE=float(np.sqrt(mse.mean())),
        coordinates=int(prediction.size),origins=n,conditional_block_bootstrap=dict(resamples=200,seed=seed,
            block_origins=size,MAE95=[float(x) for x in intervals[:,0]],MSE95=[float(x) for x in intervals[:,1]],
            training_seed_uncertainty=False,temporal_stationarity_certified=False))


def numerical_setup(job):
    body=read(ROOT/pretrained.pretrain.masks.TRAIN_BODY)
    if pretrained.pretrain.numerical_environment()!=body['verified_native_environment']:
        raise ValueError('Actual full native source training environment required')
    import numpy as np
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Source CUDA/bfloat16 AMP required;no CPU or precision fallback')
    torch.cuda.set_device(0);torch.set_num_threads(2)
    path=ROOT/SOURCE/'moment/utils/utils.py';node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name=='control_randomness')
    namespace=dict(random=random,os=os,np=np,torch=torch)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    namespace['control_randomness'](job['seed'])


def worker(config,job,output):
    trained=pretrained.trained_checkpoint(config,BRANCHES[job['branch']]['mixer'])
    binding=ready_data(job['dataset'],job['horizon'])
    if trained is None or binding is None:
        raise ValueError('Actual full39479update pretrained model and all18inputs required before NN imports')
    verify_input_arrays(binding);numerical_setup(job)
    import numpy as np
    import torch
    import yaml
    args=yaml.safe_load((ROOT/SOURCE/'configs/default.yaml').read_text(encoding='utf-8'))
    args.update(yaml.safe_load((ROOT/SOURCE/'configs/forecasting/linear_probing.yaml').read_text(encoding='utf-8')))
    if (args['max_epoch']!=5 or args['train_batch_size']!=64 or args['val_batch_size']!=256 or args['num_workers']!=5
            or not args['shuffle'] or not args['pin_memory'] or args['lr_scheduler_type']!='onecyclelr'
            or args['head_dropout']!=.1 or args['add_positional_embedding'] or not args['use_amp'] or args['torch_dtype']!='bfloat16'):
        raise ValueError('Exact registered source fine-tuning recipe required')
    args.update(seq_len=256,forecast_horizon=job['horizon'],dataset_names=binding['raw']['path'],
                checkpoint_path=str(Path(output)/'source_checkpoints'))
    core=pretrained.load_trained_core(config,trained)
    model=make_forecaster(core,job['horizon'],job['branch'],ROOT/SOURCE)
    loaders=make_loaders(binding,Path(output)/'orders')
    task=SourceForecastTask(model,loaders,SimpleNamespace(**args),output,ROOT/SOURCE,job,'cuda:0',trained)
    task.train();proof=task.finalize()
    selected=proof['selected_TEST_bundle'];directory=Path(selected['directory'])
    p=np.load(directory/'prediction.npy',mmap_mode='r',allow_pickle=False);y=np.load(directory/'truth.npy',mmap_mode='r',allow_pickle=False)
    order=np.load(selected['order']['order_path'],allow_pickle=False)
    metrics=independent_metrics(p,y,order,block=256+job['horizon'])
    source_metrics=task.compute_forecasting_metrics.__globals__['get_forecasting_metrics'](y,p)
    for name,value in [('MAE',source_metrics.mae),('MSE',source_metrics.mse),('RMSE',source_metrics.rmse)]:
        if not np.isclose(metrics[name],float(value),**config['metric_tolerance']):
            raise ValueError('All-coordinate independent float64 metrics must agree with source')
    write(Path(output)/'full_forecast_training.json',dict(**proof,configuration_sha256=sha(ROOT/config['_config_path']),
        actual_complete_pretrained_checkpoint=trained,data_binding=binding,metrics=metrics,
        source_point_metrics=dict(MAE=float(source_metrics.mae),MSE=float(source_metrics.mse),RMSE=float(source_metrics.rmse)),
        source_validation_is_unweighted_batch_average=True,source_all_roles_shuffle_retained=True,
        native_full_checkpoint_replay_complete=False,strict_leaderboard_result=False))


def replay_worker(config,job,output):
    trained=pretrained.trained_checkpoint(config,BRANCHES[job['branch']]['mixer']);binding=ready_data(job['dataset'],job['horizon'])
    if trained is None or binding is None:raise ValueError('Actual complete trained model and data required')
    verify_input_arrays(binding)
    output=Path(output);native_path=output/'full_forecast_training.json';proof=read(native_path)
    if (not proof['passed'] or proof['job']!=job or proof['configuration_sha256']!=sha(ROOT/config['_config_path'])
            or proof['actual_complete_pretrained_checkpoint']!=trained or proof['data_binding']!=binding
            or proof['full_epochs']!=5):raise ValueError('Actual complete five-epoch original result required')
    checkpoint=proof['selected_best_checkpoint'];bundle=proof['selected_TEST_bundle'];directory=Path(bundle['directory'])
    for p,h in [(checkpoint['path'],checkpoint['sha256']),(directory/'truth.npy',bundle['truth_sha256']),
                (directory/'prediction.npy',bundle['prediction_sha256']),(bundle['order']['order_path'],bundle['order']['order_sha256'])]:
        if sha(p)!=h:raise ValueError('Full selected checkpoint/prediction/order changed')
    numerical_setup(job)
    import numpy as np
    import torch
    model=make_forecaster(pretrained.load_trained_core(config,trained),job['horizon'],job['branch'],ROOT/SOURCE)
    payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
    if payload['job']!=job or payload['steps']!=checkpoint['optimizer_steps']:raise ValueError('Exact selected checkpoint identity required')
    model.load_state_dict(payload['model_state_dict'],strict=True);model.to('cuda:0');model.eval()
    order=np.load(bundle['order']['order_path'],allow_pickle=False)
    if not np.array_equal(np.sort(order),np.arange(len(order))):raise ValueError('Full saved origin order required')
    dataset=WindowDataset(binding,'test')
    loader=torch.utils.data.DataLoader(dataset,batch_size=256,sampler=order.tolist(),num_workers=5,pin_memory=True,
        drop_last=False,collate_fn=collate_forecast)
    target=np.load(directory/'truth.npy',mmap_mode='r',allow_pickle=False)
    original_prediction=np.load(directory/'prediction.npy',mmap_mode='r',allow_pickle=False)
    if (output/'full_native_replay_prediction.npy').exists():raise FileExistsError('Preserve prior full native forecast attempt')
    replay=np.lib.format.open_memmap(output/'full_native_replay_prediction.npy',mode='w+',dtype=np.float32,shape=target.shape)
    position=0;maximum=0.
    with torch.no_grad():
        for batch in loader:
            x=batch.timeseries.float().to('cuda:0');mask=batch.input_mask.long().to('cuda:0')
            y=batch.forecast.float().cpu().numpy()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):value=model(x_enc=x,input_mask=mask,mask=None).forecast
            predicted=value.detach().cpu().numpy();end=position+len(y)
            if (not np.array_equal(batch.window_indices,order[position:end]) or not np.array_equal(y,target[position:end])
                    or not np.allclose(predicted,original_prediction[position:end],**config['prediction_tolerance'])):
                raise ValueError('Full actual native forecast/targets/origin replay must agree')
            replay[position:end]=predicted;maximum=max(maximum,float(np.max(np.abs(predicted-original_prediction[position:end]))));position=end
    if position!=len(dataset):raise ValueError('Every source TEST origin required in fresh full model replay')
    replay.flush();metrics=independent_metrics(replay,target,order,block=256+job['horizon'])
    for key in ['MSE','MAE','RMSE']:
        if not np.isclose(metrics[key],proof['metrics'][key],**config['metric_tolerance']):raise ValueError('Independent replay metrics differ')
    write(output/'full_native_forecast_review.json',dict(passed=True,job=job,source_training_proof_sha256=sha(native_path),
        reloaded_entire_selected_model=True,full_TEST_origins=position,maximum_absolute_prediction_difference=maximum,
        prediction_tolerance=config['prediction_tolerance'],metrics=metrics,replay_prediction_sha256=sha(output/'full_native_replay_prediction.npy'),
        historical_paper_equivalence_certified=False,all_original_experiments_complete=False))
