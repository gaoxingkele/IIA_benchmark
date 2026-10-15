"""Restore every Reflow buffer and verify all source-FP32 teacher ODE steps.

Teacher forwards use the original saved weights and Torch kernels. A separate
state program checks Euler/Heun inputs,updates and discrete action;no refitting.
"""
from __future__ import annotations
from copy import deepcopy
import json,math


def tensor_bytes(actual,expected):
    import torch
    if (actual.device.type!='cpu' or expected.device.type!='cpu' or actual.dtype!=expected.dtype or actual.shape!=expected.shape
            or not torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),expected.contiguous().reshape(-1).view(torch.uint8))):
        raise ValueError('Every original tensor requires exact dtype,shape and content bytes')


def restore_model(checkpoint,job,model_config,features):
    import torch
    from scripts.flow_matching.run_objective_full440_checkpoint_replay_v1 import restore_model as original
    staged=model_config['entrypoint']=='iia_benchmark.models.fm_reflow:ReflowCFMDetector'
    keys={'_model','_reflow_evidence'} if staged else {'_model'}
    if set(checkpoint['modules'])!=keys or checkpoint['adjacency'] is not None:
        raise ValueError('Complete final model plus every staged evidence module required')
    private=dict(checkpoint,modules={'_model':checkpoint['modules']['_model']})
    model=original(dict(job=job,config=model_config),private,features)
    for key,value in model._model.state_dict().items():tensor_bytes(value,checkpoint['modules']['_model'][key])
    if staged:
        evidence=torch.nn.Module()
        for key,value in checkpoint['modules']['_reflow_evidence'].items():
            if value.device.type!='cpu' or not torch.isfinite(value).all():raise ValueError('Every complete CPU evidence buffer must be finite')
            evidence.register_buffer(key,value.detach().clone())
        evidence.load_state_dict(checkpoint['modules']['_reflow_evidence'],strict=True)
        for key,value in evidence.state_dict().items():tensor_bytes(value,checkpoint['modules']['_reflow_evidence'][key])
        model._reflow_evidence=evidence
    return model


def validate_stage_records(model,job,result,training_windows):
    import torch
    stages=getattr(model,'reflow_stages',1);initial=model.epochs;extra=getattr(model,'reflow_epochs',0)
    total=initial+(stages-1)*extra
    losses=result['training_losses']
    if (total!=job['total_training_epochs'] or job['additional_pair_generation_compute']!=(stages>1)
            or not isinstance(losses,list) or len(losses)!=total or not all(math.isfinite(v) and v>=0 for v in losses)):
        raise ValueError('Actual full source total training epochs and loss history required')
    if stages==1:
        if hasattr(model,'_reflow_evidence'):raise ValueError('Independent epoch control must not be relabeled as learned Reflow')
        return []
    evidence=model._reflow_evidence.state_dict();metadata=evidence.get('metadata_json_utf8')
    if metadata is None or metadata.dtype!=torch.uint8 or metadata.ndim!=1 or not len(metadata):
        raise ValueError('Complete canonical persisted stage metadata required')
    raw=bytes(metadata.tolist());records=json.loads(raw.decode('utf-8'))
    if json.dumps(records,sort_keys=True,allow_nan=False).encode('utf-8')!=raw:
        raise ValueError('Original canonical stage metadata bytes must remain unchanged')
    if len(records)!=stages:raise ValueError('Every initial and reflow stage required')
    first=records[0]
    if (first['stage']!=1 or first['kind']!='independent_rectification' or first['epochs']!=initial
            or first['training_windows']!=training_windows or len(first['losses'])!=initial):
        raise ValueError('Complete original initial rectification stage required')
    expected={'metadata_json_utf8'};flat=list(first['losses']);names=list(dict(model._model.named_parameters()))
    dimensions=int(__import__('numpy').prod(model._shape))
    for stage in range(2,stages+1):
        record=records[stage-1]
        teacher=[dict(name=n,buffer=f'stage{stage}_teacher{i}') for i,n in enumerate(names)]
        if (record['stage']!=stage or record['kind']!='learned_coupling_uniform_time_l2' or record['epochs']!=extra
                or record['training_pairs']!=training_windows or record['pair_seed']!=model.seed+1000*stage
                or record['teacher_parameters']!=teacher or record['solver']!=model.reflow_solver
                or record['ode_steps']!=model.reflow_ode_steps or not record['warm_start'] or not record['optimizer_reset']
                or len(record['losses'])!=extra):
            raise ValueError('Original complete reflow stage,seed,teacher,budget,solver and reset declarations required')
        flat.extend(record['losses'])
        expected.update(x['buffer'] for x in teacher);expected.update([f'stage{stage}_z0',f'stage{stage}_z1'])
        for role in ['z0','z1']:
            value=evidence[f'stage{stage}_{role}']
            if value.dtype!=torch.float32 or value.shape!=(training_windows,dimensions) or not torch.isfinite(value).all():
                raise ValueError('Every full original ordered endpoint pair coordinate required')
    if set(evidence)!=expected or flat!=losses:raise ValueError('All source evidence buffers and per-stage losses must remain complete')
    model.stage_records_=records;model.training_losses_=list(losses)
    return records


class CheckedTeacher:
    """Independent source-FP32 state program driven by unchanged NN velocities."""
    def __init__(self,teacher,origins,steps,solver):
        import torch
        if steps<1 or solver not in ['euler','heun']:raise ValueError('Explicit complete ODE scheme required')
        self.teacher,self.origins,self.x=teacher,origins,origins.detach().clone()
        self.steps,self.solver,self.step,self.phase=steps,solver,0,0
        self.action=torch.zeros(len(origins),dtype=torch.float64,device='cpu')
        self.forward_calls=0;self.checked_state_coordinates=0;self.v1=None

    def __call__(self,inputs):
        import torch
        if self.step>=self.steps:raise ValueError('Extra source teacher ODE step')
        t=torch.full((len(self.x),1),self.step/self.steps,dtype=self.x.dtype,device='cpu')
        expected=self.x if self.phase==0 else self.x+self.v1/self.steps
        expected_t=t if self.phase==0 else t+1/self.steps
        tensor_bytes(inputs[:,:-1],expected);tensor_bytes(inputs[:,-1:],expected_t)
        self.checked_state_coordinates+=expected.numel();self.forward_calls+=1
        out=self.teacher(inputs)
        if out.shape!=self.x.shape or out.dtype!=self.x.dtype or not torch.isfinite(out).all():
            raise ValueError('Complete original finite teacher velocity required')
        if self.solver=='heun' and self.phase==0:self.v1=out.detach();self.phase=1;return out
        effective=out if self.solver=='euler' else (self.v1+out)/2
        self.x=self.x+effective/self.steps
        self.action+=effective.double().square().mean(1)/self.steps
        self.step+=1;self.phase=0;self.v1=None
        return out

    def finish(self,terminal,record):
        if self.step!=self.steps or self.phase or self.forward_calls!=self.steps*(2 if self.solver=='heun' else 1):
            raise ValueError('Every declared ODE step and predictor/corrector forward required')
        tensor_bytes(terminal,self.x)
        displacement=(self.x-self.origins).double().square().mean(1)
        if (record['discrete_excess_action_mean']!=float((self.action-displacement).mean())
                or record['squared_displacement_per_coordinate']!=float(displacement.mean())):
            raise ValueError('Independent complete discrete action/displacement program mismatch')
        return dict(steps=self.steps,forward_calls=self.forward_calls,checked_state_coordinates=self.checked_state_coordinates)


def verify_all_couplings(model,job,result,training_windows):
    import torch
    from iia_benchmark.models.fm_reflow import transport_endpoints,pair_digest
    records=validate_stage_records(model,job,result,training_windows)
    if not records:return dict(independent_epoch_control=True,stages=[],all_source_evidence_required=True)
    evidence=model._reflow_evidence.state_dict();summaries=[]
    for record in records[1:]:
        stage=record['stage'];z0,z1=evidence[f'stage{stage}_z0'],evidence[f'stage{stage}_z1']
        generator=torch.Generator(device='cpu').manual_seed(record['pair_seed'])
        regenerated=torch.randn(tuple(z0.shape),generator=generator)
        tensor_bytes(regenerated,z0);del regenerated
        if pair_digest(z0,z1)!=record['pair_sha256']:raise ValueError('Original ordered pair digest must match every coordinate')
        teacher=deepcopy(model._model)
        weights={item['name']:evidence[item['buffer']] for item in record['teacher_parameters']}
        teacher.load_state_dict(weights,strict=True);teacher.eval().requires_grad_(False)
        for name,value in teacher.state_dict().items():tensor_bytes(value,weights[name])
        geometry=[];calls=0;coordinates=0;points=0
        for start in range(0,training_windows,model.batch_size):
            stop=min(start+model.batch_size,training_windows);origins=z0[start:stop]
            observed=CheckedTeacher(teacher,origins,model.reflow_ode_steps,model.reflow_solver)
            terminal,g=transport_endpoints(observed,origins,model.reflow_ode_steps,model.reflow_solver)
            tensor_bytes(terminal,z1[start:stop]);counts=observed.finish(terminal,g)
            calls+=counts['forward_calls'];coordinates+=counts['checked_state_coordinates'];points+=stop-start
            geometry.append((stop-start,g))
        if points!=training_windows:raise ValueError('Every source training pair must be reintegrated')
        for field in ['discrete_excess_action_mean','squared_displacement_per_coordinate']:
            value=sum(n*r[field] for n,r in geometry)/training_windows
            if value!=record['teacher_'+field]:raise ValueError('Full original weighted teacher geometry must recompute exactly')
        for name,value in teacher.state_dict().items():tensor_bytes(value,weights[name])
        summaries.append(dict(stage=stage,pairs=training_windows,dimensions=z0.shape[1],pair_seed=record['pair_seed'],
            pair_sha256=record['pair_sha256'],all_origin_coordinates_regenerated_exactly=True,
            all_teacher_endpoint_coordinates_reinferred_exactly=True,solver=model.reflow_solver,steps=model.reflow_ode_steps,
            teacher_forward_calls=calls,independent_state_program_checked_coordinates=coordinates,
            all_source_teacher_geometry_exact=True,neural_teacher_kernel_independently_rewritten=False))
    return dict(independent_epoch_control=False,stages=summaries,all_source_evidence_required=True,
        optimizer_steps_or_warm_start_trajectory_independently_replayed=False)
