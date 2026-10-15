"""Table2 joint direct-Max reconstruction with learnable or fixed-sigma priors.

These are new full training jobs, distinct from minimax and posthoc criteria.
The frozen main model, data split, normalization and all budgets remain intact.
"""
import argparse
import ast
from functools import partial
import inspect
import json
from types import SimpleNamespace
import sys

from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent

ROOT,read,sha,write=parent.ROOT,parent.read,parent.sha,parent.write
PARENT='configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'
MODULE='scripts.flow_matching.run_anomaly_transformer_paper_max_ablation_v1'
VARIANTS=['learnable_prior_max','fixed_sigma1_prior_max']


def direct_max_losses(reconstruction,target,priors,series):
    rec=(reconstruction-target).square().sum(dim=(-2,-1)).mean()
    discrepancy,_=parent.head_mean_discrepancy(priors,series,epsilon=.0001)
    loss=rec-3*discrepancy.abs().sum(dim=-1).mean()
    return loss,loss,rec


def patched_attention(relative,source,variant):
    output=parent.author.patched(relative,source)
    if variant=='fixed_sigma1_prior_max' and relative=='model/attn.py':
        output=parent.author.replace_once(output,
            'self.sigma_projection = nn.Linear(d_model,\n                                          n_heads)',
            'self.sigma_projection = nn.Linear(d_model,\n                                          n_heads)\n'
            '        self.sigma_projection = None  # Preserve paired initialization RNG, remove unused trainables')
        output=parent.author.replace_once(output,
            'sigma = self.sigma_projection(x).view(B, L, H)',
            'sigma = torch.ones((B, L, H), device=x.device, dtype=x.dtype)')
        output=parent.author.replace_once(output,
            'sigma = torch.sigmoid(sigma * 5) + 1e-5\n        sigma = torch.pow(3, sigma) - 1',
            'sigma = torch.ones_like(sigma)  # Table2 fixes the actual Gaussian scale to exactly1')
    return output


def verify(config):
    original=read(ROOT/PARENT);contract=parent.verify(original)
    if config['parent_registration']!=PARENT or config['parent_registration_sha256']!=sha(ROOT/PARENT):
        raise ValueError('Frozen main reconstruction changed')
    for key in ['model','settings','seeds','data_root','data_cache_root','contract_sha256','original_scope']:
        if config[key]!=original[key]:raise ValueError('Keep the full original numerical/data/resource contract: '+key)
    if config['base_training']!=original['training'] or config['variants']!=VARIANTS:
        raise ValueError('Both paper direct-Max variants and full base budget required')
    expected_training=dict(original['training'],backward_strategy='one_joint_direct_Max_backward_then_one_Adam_step',
        loss_reduction='per_window_Frobenius_squared_minus_lambda_L1_joint_discrepancy_then_batch_mean')
    if config['training']!=expected_training:
        raise ValueError('Actual direct-Max training configuration must match the executed update')
    expected=[dict(id=f"anomaly_transformer__paper_ablation__{v}__{j['dataset'].lower()}__seed{j['seed']}",
                   dataset=j['dataset'],seed=j['seed'],variant=v)
              for v in VARIANTS for j in original['jobs']]
    if config['jobs']!=expected or len(expected)!=50:raise ValueError('Both complete five-dataset/five-seed ablations required')
    if config['score_streams']!=parent.STREAMS or config['primary_score_stream']!='all.multiplication':
        raise ValueError('Complete scoring and prespecified multiplication criterion required')
    if config['optimizer_update']!='one_joint_direct_Max_backward_then_one_Adam_step':
        raise ValueError('Direct Max must not retain the minimax two-backward update')
    if config['fixed_prior_scale']!=1 or not config['paired_shared_parameter_initialization']:
        raise ValueError('Exact sigma1 and paired shared parameter initialization required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen direct-Max source changed: '+pin['path'])
    return contract


class MaxWorkerBinding(ast.NodeTransformer):
    def __init__(self,variant):self.variant=variant;self.backward=0;self.calls=0;self.patch_note=0
    def visit_Expr(self,node):
        if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
            and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='minimum'
            and node.value.func.attr=='backward'):
            self.backward+=1;return None
        return self.generic_visit(node)
    def visit_keyword(self,node):
        if node.arg=='actual_backward_calls':
            if not (isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Mult)
                    and isinstance(node.value.left,ast.Name) and node.value.left.id=='updates'
                    and isinstance(node.value.right,ast.Constant) and node.value.right.value==2):
                raise ValueError('Original optimizer-count observation changed')
            node.value=ast.Name(id='updates',ctx=ast.Load());self.calls+=1
        elif node.arg=='numerical_algorithm_changes':
            node.value=ast.Constant(value=('CPU compatibility; fixed Gaussian sigma1; sigma projection removed '
                 'after consuming its paired initialization RNG' if self.variant=='fixed_sigma1_prior_max'
                 else 'CPU device compatibility only; learnable prior retained'))
            self.patch_note+=1
        return self.generic_visit(node)


def bound_worker(variant,verify_function=verify,root=ROOT):
    if variant not in VARIANTS:raise ValueError('Unknown Table2 Max variant')
    tree=ast.parse(inspect.getsource(parent.run_worker));binding=MaxWorkerBinding(variant);tree=binding.visit(tree)
    if (binding.backward,binding.calls,binding.patch_note)!=(1,1,1):
        raise ValueError('Frozen complete main worker changed; refuse partial direct-Max binding')
    ast.fix_missing_locations(tree)
    namespace=dict(vars(parent),ROOT=root,verify=verify_function,minimax_losses=direct_max_losses,
                   author=SimpleNamespace(patched=partial(patched_attention,variant=variant)))
    exec(compile(tree,__file__+':exact_joint_Max_binding','exec'),namespace)
    return namespace['run_worker']


def run_worker(config,job,path):
    verify(config)
    return bound_worker(job['variant'])(config,job,path)


class MaxControllerBinding(ast.NodeTransformer):
    def __init__(self):self.module=0;self.count=0
    def visit_Constant(self,node):
        if node.value=='scripts.flow_matching.run_anomaly_transformer_paper_contract_v1':
            self.module+=1;return ast.copy_location(ast.Constant(value=MODULE),node)
        return node
    def visit_keyword(self,node):
        if node.arg=='selected_main_jobs':
            if not isinstance(node.value,ast.Constant) or node.value.value!=25:raise ValueError('Main scope changed')
            self.count+=1;node.arg='selected_direct_Max_training_jobs';node.value=ast.Constant(value=50)
        return self.generic_visit(node)


def controller(config,path):
    verify(config)
    tree=ast.parse(inspect.getsource(parent.controller));binding=MaxControllerBinding();tree=binding.visit(tree)
    if (binding.module,binding.count)!=(1,1):raise ValueError('Frozen controller dispatch contract changed')
    ast.fix_missing_locations(tree);namespace=dict(vars(parent),verify=verify)
    exec(compile(tree,__file__+':exact_frozen_controller','exec'),namespace)
    namespace['controller'](config,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.worker:
        run_worker(config,next(j for j in config['jobs'] if j['id']==args.job_id),ROOT/args.config)
        return
    from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
    api=install_facades()
    if args.probe_only:
        print(json.dumps(dict(passed=True,new_full_training_jobs=50,variants=VARIANTS,
            snapshot=api['resource_snapshot'](),numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules,all_original_experiments_complete=False)))
    else:controller(config,args.config)


if __name__=='__main__':main()
