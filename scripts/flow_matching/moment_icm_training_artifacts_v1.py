"""Durable full training state and local callbacks, without a training launcher.

Saving/loading is checked against the paused live state. This is serialization
verification, not independent checkpoint model reinference or downstream work.
"""
from __future__ import annotations
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import re
from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import sha,write


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        cuda_context_was_initialized=torch.cuda.is_initialized())


def exact(left,right):
    import numpy as np
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype==right.dtype and left.shape==right.shape and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(exact(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return type(left)==type(right) and left==right


def audit_optimizer(optimizer,step):
    import torch
    states=optimizer.state_dict();steps=[]
    for record in states['state'].values():
        if 'step' not in record:raise ValueError('Every actual AdamW state requires its full step counter')
        actual=float(record['step']);steps.append(actual)
        if actual!=step:raise ValueError('Every active optimizer parameter must retain the complete update count')
        for value in record.values():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():raise ValueError('Finite optimizer moments and counters required')
    if not steps:raise ValueError('Actual nonempty trained optimizer state required')
    return states,len(steps)


class TrainingArtifacts:
    def __init__(self,root,job,configuration_sha256):
        self.root=Path(root);self.job=dict(job);self.configuration_sha256=configuration_sha256
        if (not re.fullmatch(r'[A-Za-z0-9_-]+',job['id']) or job['epochs']!=1
                or not isinstance(job['expected_batches'],int) or job['expected_batches']<=0
                or not isinstance(job['expected_items'],int) or job['expected_items']<=0
                or not re.fullmatch(r'[0-9a-f]{64}',configuration_sha256)):
            raise ValueError('Exact oneepoch job identity, complete budgets and configuration hash required')
        if job['scientific'] and (job['expected_batches']!=39479 or job['expected_items']!=80850189 or job['sources']!=2086):
            raise ValueError('Scientific job cannot substitute a smaller corpus or training budget')
        self.root.mkdir(parents=True,exist_ok=False)
        self.name=job['id'];self.logged_steps=0;self.checkpoints=[];self.pending_evaluations=[]
        self.checkpoint_root=self.root/'checkpoints'
        self.checkpoint_root.mkdir()
        write(self.root/'identity.json',dict(job=job,configuration_sha256=configuration_sha256,
            captured_utc=datetime.now(timezone.utc).isoformat(),training_complete=False))

    def log(self,value):
        if set(value)!={'step_train_loss','learning_rate'} or any(not math.isfinite(float(v)) for v in value.values()):
            raise ValueError('Every original finite loss and learning-rate record required')
        self.logged_steps+=1
        with (self.root/'training.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(dict(step=self.logged_steps,**value),allow_nan=False)+'\n');stream.flush()

    def alert(self,**value):
        with (self.root/'local_alerts.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps({k:str(v) for k,v in value.items()},allow_nan=False)+'\n')

    def _save(self,model,optimizer,scaler,scheduler,step,phase,scope):
        import torch
        if not 0<step<=self.job['expected_batches'] or self.logged_steps!=step:
            raise ValueError('All logged source optimizer steps required')
        if model.pending_backward or any(layer.channels!=1 for layer in model.core.icm_layers):
            raise ValueError('Checkpoint only after complete backward and context release')
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('Finite complete model state required')
        state,active=audit_optimizer(optimizer,step)
        rng=capture_rng()
        scheduler_state={key:value for key,value in vars(scheduler).items() if key!='optimizer'}
        payload=dict(schema_version=1,job=self.job,configuration_sha256=self.configuration_sha256,
            optimizer_updates=step,phase=phase,scope=scope,model_state_dict=model.state_dict(),
            optimizer_state_dict=state,scaler_state_dict=scaler.state_dict(),scheduler_state=scheduler_state,
            optimizer_current_learning_rates=[g['lr'] for g in optimizer.param_groups],rng=rng)
        file=self.checkpoint_root/(f'{phase}_{step}.pt')
        with file.open('xb') as stream:torch.save(payload,stream)
        restored=torch.load(file,map_location='cpu',weights_only=False)
        if not exact(payload,restored):raise ValueError('Every saved model/optimizer/AMP/scheduler/RNG value must be exact')
        if not exact(rng,capture_rng()):raise ValueError('Checkpoint IO must not consume any training random draws')
        record=dict(path=str(file),sha256=sha(file),optimizer_updates=step,phase=phase,active_optimizer_states=active,
            all_saved_state_after_cpu_load_exact=True,training_rng_unchanged_by_checkpoint_IO=True,
            model_reconstructed_or_reinferred=False)
        self.checkpoints.append(record);write(file.with_suffix('.verification.json'),record)
        return record

    def save_periodic(self,model,path,steps,optimizer,scaler,scheduler):
        expected=self.checkpoint_root/self.name
        if Path(path).resolve()!=expected.resolve():raise ValueError('Only this job checkpoint directory is writable')
        return self._save(model,optimizer,scaler,scheduler,steps,'source_periodic_pre_scheduler',None)

    def defer_external(self,path,steps,device):
        if not self.checkpoints or self.checkpoints[-1]['optimizer_updates']!=steps:
            raise ValueError('Actual corresponding checkpoint required for deferred evaluation')
        if Path(path).resolve()!=(self.checkpoint_root/self.name).resolve():raise ValueError('Exact checkpoint path required')
        record=dict(checkpoint=self.checkpoints[-1],source_callback_step=steps,status='NOT_EXECUTED_PENDING',
            source_external_evaluation_callback_recorded=True,actual_downstream_evaluation_complete=False,
            original_source_used_next_GPU_index=True,supplied_training_device=str(device),
            boundary='Original related-source external process is not silently launched on another GPU; pending evaluation remains an explicit obligation.')
        target=self.root/f'pending_external_evaluation_{steps}.json'
        if target.exists():raise FileExistsError('Preserve every previous evaluation obligation')
        write(target,record);self.pending_evaluations.append(record)

    def finalize(self,task,loader):
        scope=loader.complete()
        if (scope['batches']!=self.job['expected_batches'] or scope['items']!=self.job['expected_items']
                or task.audited_batches!=self.job['expected_batches'] or self.logged_steps!=self.job['expected_batches']):
            raise ValueError('Final checkpoint requires the entire declared epoch and every audited update')
        final=self._save(task.model,task.optimizer,task.scaler,task.lr_scheduler,
            self.job['expected_batches'],'final_post_scheduler',scope)
        receipt=dict(passed=True,job=self.job,configuration_sha256=self.configuration_sha256,scope=scope,
            complete_declared_epoch_training_state_saved=True,final_checkpoint=final,checkpoints=self.checkpoints,
            actual_audited_optimizer_updates=task.audited_batches,active_reconstruction_targets=task.active_targets,
            zero_valid_target_batches_with_optimizer_slot_retained=task.zero_target_batches,
            deferred_external_evaluations=self.pending_evaluations,all_external_evaluations_complete=not self.pending_evaluations,
            independent_model_reinference_complete=False,benchmark_or_paper_performance_certified=False,
            fixture_is_benchmark_performance=False,all_original_experiments_complete=False)
        if (self.root/'complete_training_state.json').exists():raise FileExistsError('Preserve previous final training proof')
        write(self.root/'complete_training_state.json',receipt)
        return receipt
