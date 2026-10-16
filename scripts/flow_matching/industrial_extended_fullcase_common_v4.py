"""Observe unchanged entropic fitting and replay all native industrial points."""
import hashlib
import math
from pathlib import Path
from types import FunctionType,MethodType,SimpleNamespace
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write


def instrument(model,expected):
    """Same source fit code object; observe solver inputs and real Adam steps."""
    import torch
    from iia_benchmark.models import fm_objective_stable as original
    from iia_benchmark.models.fm_objective_stable_extended import extended_plan
    audit=dict(optimizer_updates=0,optimizer_instances=0,transport_problems=[])
    def adam(*args,**kwargs):
        result=torch.optim.Adam(*args,**kwargs);audit['optimizer_instances']+=1
        def stepped(optimizer,arguments,keywords):audit['optimizer_updates']+=1
        result.register_step_post_hook(stepped)
        return result
    class TorchView:
        optim=SimpleNamespace(Adam=adam)
        def multinomial(self,*args,**kwargs):
            result=torch.multinomial(*args,**kwargs);index=len(audit['transport_problems'])-1
            if result.tolist()!=expected[index]['pair_indices']:raise ValueError('Actual source pair sampling differs from qualified RNG sequence')
            audit['transport_problems'][index]['pair_indices']=result.tolist()
            return result
        def __getattr__(self,name):return getattr(torch,name)
    def solve(x0,x1,regularization):
        index=len(audit['transport_problems'])
        if index>=len(expected):raise ValueError('Unexpected extra original coupling problem')
        row=expected[index]
        endpoints=dict(x0_sha256=hashlib.sha256(x0.numpy().tobytes()).hexdigest(),x1_sha256=hashlib.sha256(x1.numpy().tobytes()).hexdigest())
        if any(endpoints[k]!=row[k] for k in endpoints) or regularization!=row['regularization']:
            raise ValueError('Actual full training endpoints differ from qualified original RNG sequence')
        records=[];plan=extended_plan(x0,x1,regularization,extended_iterations=model.sinkhorn_max_final_iterations,audit=records)
        digest=hashlib.sha256(plan.tobytes()).hexdigest()
        if digest!=row['coupling_sha256']:raise ValueError('Actual complete training coupling differs from qualified original plan')
        record=dict(records[0],**endpoints,coupling_sha256=digest,qualified_problem_index=index)
        audit['transport_problems'].append(record);model.transport_solver_audit_.append(record)
        return plan
    body=original.StableCFMObjectiveDetector.fit
    private=FunctionType(body.__code__,dict(body.__globals__,torch=TorchView(),stable_plan=solve),body.__name__,body.__defaults__,body.__closure__)
    def fit(self,windows):
        self.transport_solver_audit_=[]
        result=private(self,windows)
        if (audit['optimizer_instances']!=1 or audit['optimizer_updates']!=len(expected)
                or len(audit['transport_problems'])!=len(expected) or len(self.training_losses_)!=self.epochs):
            raise ValueError('Full original optimizer, coupling and epoch budgets required')
        return result
    model.fit=MethodType(fit,model)
    return audit


def restore(checkpoint,configuration,overrides,shape):
    import torch
    from torch import nn
    from scripts.flow_matching.model_registry import build_configured_method
    if checkpoint['configuration']!=configuration or checkpoint['overrides']!=overrides or checkpoint['window']!=shape[0] or set(checkpoint['modules'])!={'_model'}:
        raise ValueError('Exact full checkpoint, configuration and CPU overrides required')
    model=build_configured_method(configuration,overrides)
    if model.objective not in ['sb','sf2m'] or model.device!='cpu':raise ValueError('Registered original entropic CPU objectives only')
    model._shape=shape;model._device=torch.device('cpu');model._score_head=model.objective=='sf2m' and model.score_weight>0
    dimensions=math.prod(shape)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model.seed)
        model._model=nn.Sequential(nn.Linear(dimensions+1,model.hidden_size),nn.SiLU(),nn.Linear(model.hidden_size,model.hidden_size),nn.SiLU(),
            nn.Linear(model.hidden_size,dimensions*(2 if model._score_head else 1)))
    model._model.load_state_dict(checkpoint['modules']['_model'],strict=True)
    for key,value in model._model.state_dict().items():
        old=checkpoint['modules']['_model'][key]
        if value.dtype!=old.dtype or value.shape!=old.shape or not torch.equal(value,old) or not torch.isfinite(value).all():
            raise ValueError('Exact finite checkpoint weights required')
    model._model.eval()
    return model


def train(case):
    from scripts.flow_matching import run_industrial_tsad_experiment as frozen
    instances=[];builder=frozen.build_configured_method;expected=read(case['coupling_report'])['coupling_problems']
    def observed(*args,**kwargs):
        model=builder(*args,**kwargs);instances.append((model,instrument(model,expected)));return model
    body=frozen.run
    private=FunctionType(body.__code__,dict(body.__globals__,build_configured_method=observed),body.__name__,body.__defaults__,body.__closure__)
    result=private(ROOT,case['job'],case['evaluation'])
    if len(instances)!=1:raise ValueError('Train exactly one complete original model')
    model,audit=instances[0]
    if result['unique_fit_windows']!=case['fit_windows'] or len(result['training_losses'])!=20 or audit['optimizer_updates']!=case['optimizer_updates']:
        raise ValueError('Full original fit windows and twenty epochs required')
    target=ROOT/case['job']['output_directory']
    write(target/'complete_training_solver_audit.json',dict(passed=True,**audit,original_epochs=20,
        source_fit_code_object_unchanged=True,source_train_and_score_code_object_unchanged=True,
        optimizer_is_real_unchanged_Adam_with_readonly_post_step_counter=True,
        actual_training_endpoints_and_coupling_bytes_equal_qualified_sequence=True,
        source_result_sha256=sha(target/'result.json'),checkpoint_sha256=sha(target/'model.pt'),scores_sha256=sha(target/'scores.npz'),
        paper_equivalence=False,all_original_experiments_complete=False))


def replay(case,source_guard,target):
    import numpy as np
    import torch
    from scripts.flow_matching.audit_industrial_SB_extended_full_replay_v1 import native_windows,native_align,require_array_equal,strict_metrics,roles,SUFFIXES
    job=case['job'];base=ROOT/job['output_directory'];result=read(base/'result.json');audit=read(base/'complete_training_solver_audit.json');guard=read(source_guard)
    if (not guard['passed'] or guard['exit_code']!=0 or guard['resource_guard_aborted'] or guard['proof_sha256']!=sha(base/'complete_training_solver_audit.json')
            or not audit['passed'] or result['job']!=job or result['status']!='completed' or audit['source_result_sha256']!=sha(base/'result.json')
            or result['checkpoint_sha256']!=sha(base/'model.pt') or result['scores_sha256']!=sha(base/'scores.npz')
            or audit['optimizer_updates']!=case['optimizer_updates'] or len(result['training_losses'])!=20):
        raise ValueError('Full successful training and bound source guard required before native replay')
    torch.set_num_threads(case['evaluation']['cpu_threads'])
    dataset=next(d for d in read(ROOT/job['dataset_manifest_path'])['datasets'] if d['id']==job['dataset'])
    if sha(ROOT/dataset['input_npz'])!=dataset['input_sha256']:raise ValueError('Complete frozen input changed')
    for raw in dataset['source_files']:
        if sha(ROOT/raw['path'])!=raw['sha256']:raise ValueError('Original raw input changed')
    configuration=read(ROOT/job['model_config']);checkpoint=torch.load(base/'model.pt',map_location='cpu',weights_only=True)
    group_rows=roles(dataset);model=restore(checkpoint,configuration,job['parameter_overrides'],(job['window'],group_rows[0][1]['features']))
    if sum(p.numel() for p in model._model.parameters())!=result['parameter_count']:raise ValueError('Exact complete parameter count required')
    payload,bindings,coverage,budget={}, {}, [], []
    with np.load(ROOT/dataset['input_npz'],allow_pickle=False) as inputs,np.load(base/'scores.npz',allow_pickle=False) as old:
        for role,g in group_rows:
            prefix=g['array_prefix'];key=prefix+'_'+role;values=inputs[prefix+('_test' if role=='test' else '_values')]
            if values.shape!=(g['points'],g['features']):raise ValueError('Every native coordinate required')
            windows,starts=native_windows(values,job['window']);payload[key]=native_align(model.score(windows),starts,len(values),job['window'])
            bindings[key]=require_array_equal(payload[key],old[key],key)
            for suffix in SUFFIXES:
                name=prefix+suffix
                if name in inputs:payload[name]=inputs[name].copy();bindings[name]=require_array_equal(payload[name],old[name],name)
            if role=='train':budget.append(dict(group_id=g['group_id'],source_interval=g['source_interval'],points=len(values),windows=len(windows)))
            coverage.append(dict(role=role,group=g['group_id'],points=len(values),windows=len(windows),tail_included=True))
        if set(payload)!=set(old.files):raise ValueError('Exact complete original score archive required')
    if budget!=checkpoint['unique_training_groups'] or budget!=result['fit_group_budget'] or sum(g['windows'] for g in budget)!=case['fit_windows']:
        raise ValueError('Distinct complete original training groups/windows required')
    metrics=strict_metrics(dataset,payload,result['metrics']['strict'],dict(rtol=2e-6,atol=2e-6),result['checkpoint_sha256'])
    target=Path(target);target.mkdir(parents=True,exist_ok=False);np.savez_compressed(target/'replayed_scores.npz',**payload)
    write(target/'report.json',dict(passed=True,canonical_original_id=case['canonical_original_id'],actual_execution_id=job['id'],
        native_coverage=coverage,all_score_and_metadata_array_payloads=bindings,all_score_arrays_exact_bytes=True,
        independent_strict_metrics=metrics,replayed_archive_sha256=sha(target/'replayed_scores.npz'),
        source_result_sha256=sha(base/'result.json'),source_checkpoint_sha256=sha(base/'model.pt'),source_scores_sha256=sha(base/'scores.npz'),
        source_optimizer_updates=case['optimizer_updates'],new_optimizer_updates=0,new_training_seed_slots=0,
        bootstrap_independently_recomputed=False,VUS_independently_recomputed=False,affiliation_independently_recomputed=False,
        full_TAB_equivalence=False,paper_equivalence=False,all_original_experiments_complete=False))
