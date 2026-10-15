"""Isolated full-epoch training body with mandatory original-target validity.

This callable integration is not an admitted scientific job or a trained model.
Keep frozen upstream files unchanged. The existing paper epoch adapter is
extended by three explicit insertions: target validity, checkpoint-context
release after backward, and a read-only finite-gradient audit after clipping.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import importlib
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from scripts.flow_matching import moment_icm_epoch_contract_v1 as epoch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_class(models_directory):
    """Load the exact model modules without unrelated benchmark app imports."""
    import sys
    from types import ModuleType
    directory=Path(models_directory).resolve()
    name='_iia_moment_training_models_'+hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package=ModuleType(name);package.__path__=[str(directory)];package.__package__=name
        sys.modules[name]=package
    return importlib.import_module(name+'.fm_supplemental_moment').MomentICMT5


def controlled_train_node(path, expected_sha256):
    node=epoch.paper_train_node(path,expected_sha256)
    body=next(n for n in ast.walk(node) if isinstance(n,ast.For)).body
    anchors={
        'observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))':
            'observed_mask = self.target_validity(observed_mask, batch_x)',
        'self.scaler.scale(loss).backward()':'self.model.finish_backward()',
        'nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)':
            'self.audit_gradients_and_loss(loss, batch_x)'}
    found=[];extended=[]
    for statement in body:
        extended.append(statement)
        text=ast.unparse(statement)
        if text in anchors:
            extended.append(ast.parse(anchors[text]).body[0]);found.append(text)
    if set(found)!=set(anchors) or len(found)!=3:
        raise ValueError('Every exact upstream target/backward/clip anchor required')
    next(n for n in ast.walk(node) if isinstance(n,ast.For)).body=extended
    ast.fix_missing_locations(node)
    # Prove no optimizer/scaler/scheduler/loss/logging order was rewritten.
    restored=deepcopy(node)
    private=next(n for n in ast.walk(restored) if isinstance(n,ast.For))
    private.body=[n for n in private.body if ast.unparse(n) not in anchors.values()]
    original=epoch.paper_train_node(path,expected_sha256)
    if ast.dump(restored,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise AssertionError('Only the three disclosed insertions may extend the paper loop')
    return node


def load_source_symbols(source_root,pins):
    """Load only pinned masking/scheduler/dtype/optimizer source, not task init."""
    import torch
    root=Path(source_root)
    def content(relative):
        path=root/relative
        if relative not in pins or digest(path)!=pins[relative]:
            raise ValueError('Exact frozen numerical source required: '+relative)
        return path.read_text(encoding='utf-8')
    masking=dict(__name__='_iia_isolated_source_masking')
    exec(compile(content('moment/utils/masking.py'),'<pinned-source-masking>','exec'),masking)
    scheduler={}
    exec(compile(content('moment/utils/optims.py'),'<pinned-source-scheduler>','exec'),scheduler)
    tree=ast.parse(content('moment/utils/utils.py'))
    dtype=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dtype_map')
    namespace=dict(torch=torch,nn=torch.nn,optim=torch.optim,
        LinearWarmupCosineLRScheduler=scheduler['LinearWarmupCosineLRScheduler'])
    exec(compile(ast.Module(body=[dtype],type_ignores=[]),'<pinned-dtype-map>','exec'),namespace)
    task=next(n for n in ast.parse(content('moment/tasks/base.py')).body if isinstance(n,ast.ClassDef) and n.name=='Tasks')
    names={'_select_optimizer','_select_criterion','_init_lr_scheduler'}
    methods=[n for n in task.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in methods}!=names:raise ValueError('All three original optimizer/criterion/scheduler methods required')
    exec(compile(ast.Module(body=methods,type_ignores=[]),'<pinned-source-training-methods>','exec'),namespace)
    return masking['Masking'],namespace


def masked_model(core,Masking,*,gradient_checkpointing,mask_ratio=.3):
    """Wrap the paper-derived model without editing frozen numerical sources."""
    import torch
    from torch import nn
    if core.task!='reconstruction' or core.seq_len!=256 or core.patch_len!=8 or mask_ratio!=.3:
        raise ValueError('Configured reconstruction256/patch8/source mask0.3 required')
    class MaskedModel(nn.Module):
        def __init__(self):
            super().__init__();self.core=core
            self.mask_generator=Masking(mask_ratio=mask_ratio,patch_len=8)
            self.pending_backward=False;self.checkpointing=gradient_checkpointing
            if gradient_checkpointing:self.core.encoder.gradient_checkpointing_enable()

        def forward(self,x_enc,input_mask,mask=None):
            if self.pending_backward:raise RuntimeError('Finish the pending backward before another channel grouping')
            if (x_enc.ndim!=3 or x_enc.shape[-1]!=256 or input_mask.shape!=(x_enc.shape[0],256)
                    or not torch.isfinite(x_enc).all() or not torch.isin(input_mask,input_mask.new_tensor([0,1])).all()):
                raise ValueError('Finite full source values and binary source padding required')
            if mask is None:mask=self.mask_generator.generate_mask(x=x_enc,input_mask=input_mask)
            if mask.shape!=input_mask.shape or not torch.isin(mask,mask.new_tensor([0,1])).all():
                raise ValueError('Source keep1/predict0 mask required')
            if not ((input_mask*mask).sum(-1)>0).all():
                raise ValueError('No retained source observations: preflight this case; no implicit RevIN fallback or row omission')
            result=self.core(x_enc,input_mask=input_mask,reconstruction_mask=mask)
            if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite reconstruction preserved without numerical retry')
            if self.training and torch.is_grad_enabled():
                # Frozen core resets channels in its forward finally clause.
                # Checkpoint backward recomputes attention AFTER that clause.
                # Restore the same grouping until backward finishes, then reset.
                for layer in self.core.icm_layers:layer.channels=x_enc.shape[1]
                self.pending_backward=True
            return SimpleNamespace(reconstruction=result,pretrain_mask=mask)

        def finish_backward(self):
            for layer in self.core.icm_layers:layer.channels=1
            self.pending_backward=False
    return MaskedModel()


class CompleteEpochLoader:
    """Materialize bounded batches from the already independently closed plan."""
    drop_last=False
    def __init__(self,plan,reader):
        if plan.plan['sources']!=reader.sources:raise ValueError('Identical full source descriptors required')
        self.plan=plan;self.reader=reader;self.started=False
        self.batches=0;self.items=0;self.valid_targets=0;self.zero_valid_target_examples=0
        self.order_hash=hashlib.sha256()

    def __len__(self):return len(self.plan)

    def __iter__(self):
        import numpy as np
        import torch
        if self.started:raise RuntimeError('Exactly one complete traversal; no implicit restart')
        self.started=True
        for index in range(len(self.plan)):
            reference=self.plan.reference(index);batch=self.reader.batch(reference)
            self.order_hash.update(np.asarray(reference.items,dtype=np.int64).tobytes())
            self.batches+=1;self.items+=len(reference.items);self.valid_targets+=int(batch['valid_targets'].sum())
            self.zero_valid_target_examples+=batch['examples_with_no_valid_target']
            yield SimpleNamespace(timeseries=torch.from_numpy(batch['values']),input_mask=torch.from_numpy(batch['input_mask']),
                valid_targets=torch.from_numpy(batch['valid_targets']),reference=reference)

    def complete(self):
        if self.batches!=len(self.plan) or self.items!=self.plan.plan['total_items']:
            raise ValueError('Every full and partial batch/item required')
        return dict(batches=self.batches,items=self.items,valid_targets=self.valid_targets,
            examples_with_no_valid_target=self.zero_valid_target_examples,reference_order_sha256=self.order_hash.hexdigest())


class SourceTrainingTask:
    """Complete callable train body; caller still needs actual guarded admission.

    Frozen AdamW/MSE/AMP/mask/clipping/scheduler order is retained. Tensorboard/
    WandB are replaced by a supplied local logger. Checkpoint callbacks must save
    actual state. Related-source external evaluation callbacks must explicitly
    record or implement downstream work; never report it complete automatically.
    """
    def __init__(self,model,loader,args,source_root,pins,logger,*,device,save_callback,external_callback):
        import torch
        if args.max_epoch!=1 or getattr(loader,'drop_last',None) is not False or len(loader)<=0:
            raise ValueError('One full epoch and all partial batches required')
        self.model=model;self.train_dataloader=loader;self.args=args;self.logger=logger;self.device=torch.device(device)
        self.save_callback=save_callback;self.external_callback=external_callback
        self.audited_batches=0;self.active_targets=0;self.zero_target_batches=0
        Masking,namespace=load_source_symbols(source_root,pins)
        for name in ('_select_optimizer','_select_criterion','_init_lr_scheduler'):
            setattr(self,name,MethodType(namespace[name],self))
        source=Path(source_root)/'moment/tasks/pretrain.py'
        node=controlled_train_node(source,pins['moment/tasks/pretrain.py'])
        # No global package import, logger login or external service is needed.
        namespace.update(os=os,tqdm=lambda iterable,**kw:iterable,
            make_dir_if_not_exists=lambda path,**kw:Path(path).mkdir(parents=True,exist_ok=True),
            AlertLevel=SimpleNamespace(INFO='INFO'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated-valid-target-one-full-epoch>','exec'),namespace)
        self.train=MethodType(namespace['train'],self)

    def target_validity(self,mask,batch):
        import torch
        valid=batch.valid_targets
        if valid.dtype!=torch.bool or valid.shape!=mask.shape:raise ValueError('Explicit per-channel/time boolean target validity required')
        valid=valid.to(device=mask.device)
        out=mask*valid.to(dtype=mask.dtype)
        active=int(out.sum().item());self.active_targets+=active
        self.zero_target_batches+=int(active==0)
        return out

    def audit_gradients_and_loss(self,loss,batch):
        import torch
        if not torch.isfinite(loss).all():raise FloatingPointError('Finite masked loss required; no retry')
        gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Finite complete clipped gradients required; no skipped update may be certified')
        self.audited_batches+=1

    def save_model(self,model,path,steps,optimizer,scaler):
        self.save_callback(model,path,steps,optimizer,scaler,self.lr_scheduler)

    def evaluate_model_external(self,path,steps,device):
        self.external_callback(path,steps,device)

    def evaluate_and_log(self):
        raise RuntimeError('A real validation loader is required before enabling validation intervals')

    def debug_model_outputs(self,*args):
        raise RuntimeError('Debug truncation is not an admitted full experiment')
