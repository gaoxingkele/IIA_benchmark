"""Full Figure7/Tables8-9 single-factor sensitivity around the frozen main model.

Rebind only the reported axis; preserve source bytes, independent validation,
all actual window coverage, optimizer budget, scoring and resource protection.
"""
import argparse
import ast
from functools import partial
import inspect
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_anomaly_transformer_paper_contract_v1 as parent

ROOT,read,sha,write=parent.ROOT,parent.read,parent.sha,parent.write
AXES=dict(window=[50,100,150,200,250,300],weight=[0,1,2,3,4,5,6],layers=[1,2,3,4],hidden=[256,512,1024])
BASE=dict(window=100,weight=3,layers=3,hidden=512)
PARENT='configs/experiments/fm_anomaly_transformer_paper_contract.v1.json'
MODULE='scripts.flow_matching.run_anomaly_transformer_paper_sensitivity_v1'


def variants():
    return [(axis,value) for axis,values in AXES.items() for value in values if value!=BASE[axis]]


def streams(layers):
    return ['all.reconstruction','all.association','all.addition','all.multiplication']+[
        f'layer{i}.multiplication' for i in range(1,layers+1)]


def verify(config):
    original=read(ROOT/PARENT);contract=parent.verify(original)
    if config['parent_registration']!=PARENT or config['parent_registration_sha256']!=sha(ROOT/PARENT):
        raise ValueError('Frozen main registration changed')
    if config['axes']!=AXES or config['base']!=BASE or config['base_model_reuse']!=original['jobs']:
        raise ValueError('All paper axes and deduplicated main-model references required')
    for key in ['training','model','settings','original_scope','seeds','data_root','data_cache_root','contract_sha256']:
        if config[key]!=original[key]:raise ValueError('Preserve unchanged main numerical/data/resource contract: '+key)
    expected=[]
    for axis,value in variants():
        for job in original['jobs']:
            values=dict(BASE);values[axis]=value
            expected.append(dict(id=f"anomaly_transformer__paper_sensitivity__{axis}{value}__{job['dataset'].lower()}__seed{job['seed']}",
                                 dataset=job['dataset'],seed=job['seed'],axis=axis,value=value,parameters=values))
    if config['jobs']!=expected or len(expected)!=400:
        raise ValueError('Every original single-factor dataset/seed run required, with no duplicate base training')
    if config['primary_score_stream']!='all.multiplication' or config['score_stream_policy']!='all_four_criteria_plus_every_individual_layer':
        raise ValueError('Complete scoring and prespecified primary criterion required')
    if config['score_streams_by_layers']!={str(n):streams(n) for n in AXES['layers']}:
        raise ValueError('Every score stream for each tested model depth required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Frozen sensitivity source changed: '+pin['path'])
    return contract


class WorkerBinding(ast.NodeTransformer):
    def __init__(self):self.windows=0;self.model=0;self.reshape=0
    def visit_Constant(self,node):
        if type(node.value) is int and node.value==100:
            self.windows+=1
            return ast.copy_location(ast.Name(id='SELECTED_WINDOW',ctx=ast.Load()),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='AnomalyTransformer':
            replacements={'d_model':'SELECTED_HIDDEN','e_layers':'SELECTED_LAYERS'}
            for keyword in node.keywords:
                if keyword.arg in replacements:keyword.value=ast.Name(id=replacements[keyword.arg],ctx=ast.Load())
            self.model+=1
        if (isinstance(node.func,ast.Attribute) and node.func.attr=='reshape' and len(node.args)==2
            and isinstance(node.args[1],ast.Constant) and node.args[1].value==3):
            node.args[1]=ast.Name(id='SELECTED_LAYERS',ctx=ast.Load());self.reshape+=1
        return node


def bound_worker(values,verify_function=verify,root=ROOT):
    tree=ast.parse(inspect.getsource(parent.run_worker));binding=WorkerBinding();tree=binding.visit(tree)
    if (binding.windows,binding.model,binding.reshape)!=(7,1,1):
        raise ValueError('Frozen main worker structure changed; refuse partial parameter binding')
    ast.fix_missing_locations(tree)
    namespace=dict(vars(parent),ROOT=root,verify=verify_function,
        SELECTED_WINDOW=values['window'],SELECTED_HIDDEN=values['hidden'],SELECTED_LAYERS=values['layers'],
        STREAMS=streams(values['layers']),minimax_losses=partial(parent.minimax_losses,weight=values['weight']))
    exec(compile(tree,__file__+':exact_frozen_worker_binding','exec'),namespace)
    return namespace['run_worker']


def run_worker(config,job,path):
    verify(config)
    return bound_worker(job['parameters'])(config,job,path)


class ControllerBinding(ast.NodeTransformer):
    def __init__(self):self.module=0;self.count=0
    def visit_Constant(self,node):
        if node.value=='scripts.flow_matching.run_anomaly_transformer_paper_contract_v1':
            self.module+=1;return ast.copy_location(ast.Constant(value=MODULE),node)
        return node
    def visit_keyword(self,node):
        if node.arg=='selected_main_jobs':
            if not isinstance(node.value,ast.Constant) or node.value.value!=25:raise ValueError('Original main count changed')
            self.count+=1;node.arg='selected_sensitivity_training_jobs';node.value=ast.Constant(value=400)
        return self.generic_visit(node)


def controller(config,path):
    verify(config)
    tree=ast.parse(inspect.getsource(parent.controller));binding=ControllerBinding();tree=binding.visit(tree)
    if (binding.module,binding.count)!=(1,1):raise ValueError('Frozen complete controller interface changed')
    ast.fix_missing_locations(tree);namespace=dict(vars(parent),verify=verify)
    exec(compile(tree,__file__+':exact_frozen_controller_binding','exec'),namespace)
    namespace['controller'](config,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--job-id');parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args();config=read(ROOT/args.config);verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True,new_training_jobs=400,reused_main_models=25,
            unique_models_across_all_four_axes=425,nonbase_variants=16,torch_imported='torch' in sys.modules,
            all_original_experiments_complete=False)))
    elif args.worker:run_worker(config,next(j for j in config['jobs'] if j['id']==args.job_id),ROOT/args.config)
    else:controller(config,args.config)


if __name__=='__main__':main()
