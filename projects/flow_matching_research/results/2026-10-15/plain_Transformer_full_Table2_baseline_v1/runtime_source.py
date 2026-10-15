"""Table2 plain scaled-dot-product Transformer, trained/scored by reconstruction.

Remove the prior computational path and trainables while preserving paired
initialization of every shared attention/FFN/embedding parameter.
"""
import argparse
import ast
import inspect
import json
from types import SimpleNamespace
import sys

from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent
from scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1 import MaxWorkerBinding

ROOT,read,sha,write=parent.ROOT,parent.read,parent.sha,parent.write
PARENT='configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'
MODULE='scripts.flow_matching.run_plain_transformer_paper_baseline_v1'
STREAMS=['all.reconstruction']


def reconstruction_losses(reconstruction,target,priors,series):
    if any(p is not None for p in priors):raise ValueError('Plain Transformer must have no prior path')
    loss=(reconstruction-target).square().sum(dim=(-2,-1)).mean()
    return loss,loss,loss


def reconstruction_scores(reconstruction,target,priors,series):
    import torch
    if any(p is not None for p in priors):raise ValueError('No prior association is allowed in this baseline')
    rec=(reconstruction-target).square().mean(dim=-1)
    entropy=[]
    for values in series:
        distribution=values.mean(dim=1)
        entropy.append(-(distribution*torch.log(distribution.clamp_min(torch.finfo(distribution.dtype).tiny))).sum(dim=-1))
    return {'all.reconstruction':rec},rec,torch.stack(entropy)


def reconstruction_evaluation(*args,**kwargs):
    result=parent.evaluation(*args,**kwargs)
    result['fixed_delta_is_diagnostic_only_for_plain_Transformer']=True
    result['primary_calibration']='predeclared_validation_ratio_inherited_from_comparison_contract'
    return result


class PlainAttentionBinding(ast.NodeTransformer):
    def __init__(self):self.removed_distance=0;self.removed_prior=0;self.sigma_projection=0;self.sigma_call=0
    def visit_ClassDef(self,node):
        if node.name=='AnomalyAttention':
            for function in node.body:
                if not isinstance(function,ast.FunctionDef):continue
                if function.name=='__init__':
                    keep=[]
                    for statement in function.body:
                        targets=statement.targets if isinstance(statement,ast.Assign) else []
                        remove=any(isinstance(t,ast.Name) and t.id=='window_size' or
                                   isinstance(t,ast.Attribute) and t.attr=='distances' for t in targets)
                        remove=remove or (isinstance(statement,ast.For) and isinstance(statement.target,ast.Name) and statement.target.id=='i')
                        if remove:self.removed_distance+=1
                        else:keep.append(statement)
                    function.body=keep
                elif function.name=='forward':
                    keep=[];inserted=False
                    for statement in function.body:
                        targets=statement.targets if isinstance(statement,ast.Assign) else []
                        if any(isinstance(t,ast.Name) and t.id in {'sigma','prior','window_size'} for t in targets):
                            self.removed_prior+=1;continue
                        if any(isinstance(t,ast.Name) and t.id=='series' for t in targets):
                            keep.extend(ast.parse('prior=None\nsigma=None').body);inserted=True
                        keep.append(statement)
                    if not inserted:raise ValueError('Original standard attention computation changed')
                    function.body=keep
        elif node.name=='AttentionLayer':
            for function in node.body:
                if not isinstance(function,ast.FunctionDef):continue
                output=[]
                for statement in function.body:
                    targets=statement.targets if isinstance(statement,ast.Assign) else []
                    if function.name=='__init__' and any(isinstance(t,ast.Attribute) and t.attr=='sigma_projection' for t in targets):
                        output.append(statement)
                        output.extend(ast.parse('self.sigma_projection=None').body);self.sigma_projection+=1;continue
                    if function.name=='forward' and any(isinstance(t,ast.Name) and t.id=='sigma' for t in targets):
                        statement.value=ast.Constant(value=None);self.sigma_call+=1
                    output.append(statement)
                function.body=output
        return node


def patched_attention(relative,source):
    source=parent.author.patched(relative,source)
    if relative!='model/attn.py':return source
    tree=ast.parse(source);binding=PlainAttentionBinding();tree=binding.visit(tree)
    if (binding.removed_distance,binding.removed_prior,binding.sigma_projection,binding.sigma_call)!=(3,7,1,1):
        raise ValueError('Frozen source prior structure changed; refuse incomplete removal')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def verify(config):
    original=read(ROOT/PARENT);contract=parent.verify(original)
    if config['parent_registration']!=PARENT or config['parent_registration_sha256']!=sha(ROOT/PARENT):
        raise ValueError('Frozen parent baseline contract changed')
    for key in ['model','settings','seeds','data_root','data_cache_root','contract_sha256','original_scope']:
        if config[key]!=original[key]:raise ValueError('Keep the full original architecture/data/resource contract: '+key)
    expected_training=dict({k:v for k,v in original['training'].items() if k not in ['discrepancy_weight','kl_log_epsilon']},
                          backward_strategy='one_reconstruction_backward_then_one_Adam_step',
                          loss_reduction='per_window_Frobenius_squared_then_batch_mean')
    if config['base_training']!=original['training'] or config['training']!=expected_training:
        raise ValueError('Full reconstruction-only training budget required')
    expected=[dict(id=f"plain_transformer__paper_Table2__{j['dataset'].lower()}__seed{j['seed']}",
                   dataset=j['dataset'],seed=j['seed'],variant='plain_attention_reconstruction') for j in original['jobs']]
    if config['jobs']!=expected or len(expected)!=25:raise ValueError('Every dataset/seed plain Transformer model required')
    if config['score_streams']!=STREAMS or config['primary_score_stream']!='all.reconstruction':
        raise ValueError('Plain Transformer primary criterion must be reconstruction')
    if config['prior_path_present'] or not config['paired_shared_parameter_initialization']:
        raise ValueError('Remove prior computation/trainables and retain paired shared initialization')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen plain Transformer source changed: '+pin['path'])
    return contract


class PlainWorkerBinding(MaxWorkerBinding):
    def __init__(self):super().__init__('learnable_prior_max');self.coverage=0;self.diagnostic=0
    def visit_Constant(self,node):
        if node.value=='all.multiplication':
            self.coverage+=1;return ast.copy_location(ast.Constant(value='all.reconstruction'),node)
        if node.value=='component.layer_discrepancies':
            self.diagnostic+=1;return ast.copy_location(ast.Constant(value='diagnostic.layer_attention_entropy'),node)
        return node
    def visit_keyword(self,node):
        node=super().visit_keyword(node)
        if node.arg=='numerical_algorithm_changes':
            node.value=ast.Constant(value='Prior distance matrices,Gaussian prior computation and sigma-projection trainables removed; shared initialization RNG retained')
        return node


def bound_worker(verify_function=verify,root=ROOT):
    tree=ast.parse(inspect.getsource(parent.run_worker));binding=PlainWorkerBinding();tree=binding.visit(tree)
    if (binding.backward,binding.calls,binding.patch_note,binding.coverage,binding.diagnostic)!=(1,1,1,2,1):
        raise ValueError('Frozen full worker/scoring layout changed')
    ast.fix_missing_locations(tree)
    namespace=dict(vars(parent),ROOT=root,verify=verify_function,STREAMS=STREAMS,
        minimax_losses=reconstruction_losses,score_components=reconstruction_scores,evaluation=reconstruction_evaluation,
        author=SimpleNamespace(patched=patched_attention))
    exec(compile(tree,__file__+':exact_plain_Transformer_binding','exec'),namespace)
    return namespace['run_worker']


def run_worker(config,job,path):
    verify(config);bound_worker()(config,job,path)


class PlainControllerBinding(ast.NodeTransformer):
    def __init__(self):self.module=0;self.count=0
    def visit_Constant(self,node):
        if node.value=='scripts.flow_matching.run_anomaly_transformer_paper_contract_v1':
            self.module+=1;return ast.copy_location(ast.Constant(value=MODULE),node)
        return node
    def visit_keyword(self,node):
        if node.arg=='selected_main_jobs':
            if not isinstance(node.value,ast.Constant) or node.value.value!=25:raise ValueError('Main scope changed')
            self.count+=1;node.arg='selected_plain_Transformer_training_jobs'
        return self.generic_visit(node)


def controller(config,path):
    verify(config)
    tree=ast.parse(inspect.getsource(parent.controller));binding=PlainControllerBinding();tree=binding.visit(tree)
    if (binding.module,binding.count)!=(1,1):raise ValueError('Frozen complete controller changed')
    ast.fix_missing_locations(tree);namespace=dict(vars(parent),verify=verify)
    exec(compile(tree,__file__+':exact_frozen_controller','exec'),namespace)
    namespace['controller'](config,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.worker:run_worker(config,next(j for j in config['jobs'] if j['id']==args.job_id),ROOT/args.config);return
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades()
    if args.probe_only:
        print(json.dumps(dict(passed=True,new_full_training_jobs=25,prior_path_present=False,
            snapshot=api['resource_snapshot'](),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,all_original_experiments_complete=False)))
    else:controller(config,args.config)


if __name__=='__main__':main()
