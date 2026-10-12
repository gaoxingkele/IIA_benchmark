"""Callable untouched SHCL released VAE core and its native CLI frontend.

Paper: https://doi.org/10.1016/j.ins.2025.122680.
Author source: Zhangzzbzzb/SHCL,022520104f2d0d3bda563557ebc99279e6812511.
The current refactored VAE release is not proof of paper-era Transformer/LSTM
equivalence. Its stochastic evaluation, model, losses and script budgets remain
unchanged. Exactly one unused --threshold parser declaration is added in memory.
"""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha


def verify(config):
    if (config['source_commit']!='022520104f2d0d3bda563557ebc99279e6812511'
            or config['paper_id']!='shcl' or config['reproduction_status']!='released_author_VAE_callable_original_environment_data_and_full_training_equivalence_pending'):
        raise ValueError('Exact released-author VAE provenance and incomplete reproduction boundary required')
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:
            raise ValueError('Frozen original SHCL release changed: '+pin['path'])
    return ROOT/config['source_root']


def native_modules(config):
    source=verify(config)
    name='_iia_shcl_author_'+config['source_commit'][:12]
    if name not in sys.modules:
        package=ModuleType(name);package.__path__=[str(source/'model')]
        sys.modules[name]=package
    module_name=name+'.SHCLVae'
    if module_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(module_name,str(source/'model/SHCLVae.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    import torch
    tree=ast.parse((source/'solver.py').read_text(encoding='utf-8'))
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('kl_divergence','compute_loss')]
    if [n.name for n in definitions]!=['kl_divergence','compute_loss']:
        raise ValueError('Exact untouched native Gaussian KL and two-branch loss required')
    namespace=dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),str(source/'solver.py'),'exec'),namespace)
    return sys.modules[module_name],namespace['compute_loss'],namespace['kl_divergence']


def build_model(config,arguments=None):
    module,loss,kl=native_modules(config)
    import torch
    params=dict(config['constructor_parameters'] if arguments is None else arguments)
    if set(params)!={'win_size','seq_size','c_in','c_out','d_model','e_layers','fr','tr','device'}:
        raise ValueError('Every original native constructor argument must be explicit')
    device=torch.device(params.pop('device'))
    model=module.SHCLVae(**params,dev=device).to(device)
    return model,loss


def corrected_main_ast(source):
    tree=ast.parse(source)
    if any(isinstance(n,ast.Constant) and n.value=='--threshold' for n in ast.walk(tree)):
        raise ValueError('Author parser already declares threshold; review a new source version')
    guards=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__']
    if len(guards)!=1:raise ValueError('Exact original command-line frontend required')
    guard=guards[0]
    indices=[i for i,n in enumerate(guard.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse_args']
    if len(indices)!=1:raise ValueError('Exactly one native parse_args call required')
    guard.body.insert(indices[0],ast.parse("parser.add_argument('--threshold', type=float, default=None)").body[0])
    return ast.fix_missing_locations(tree)


def parse_author_cli(config,argv,corrected=True):
    source=verify(config)
    tree=(corrected_main_ast((source/'main.py').read_text(encoding='utf-8')) if corrected
          else ast.parse((source/'main.py').read_text(encoding='utf-8')))
    guard=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')
    body=[]
    for node in guard.body:
        body.append(copy.deepcopy(node))
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in node.targets):break
    call=body[-1].value
    call.args=[ast.Name(id='_argv',ctx=ast.Load())]
    namespace=dict(argparse=argparse,_argv=argv)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source/'main.py'),'exec'),namespace)
    return namespace['config']


def released_script_commands(config):
    """Enumerate every provided Bash loop; never execute shell text."""
    source=verify(config);commands=[]
    for dataset in ('MSL','SMAP','PSM','SWAT'):
        text=(source/'scripts'/(dataset+'.sh')).read_text(encoding='utf-8')
        found=re.findall(r'python main\.py(.*?)(?=\necho|\Z)',text,re.S)
        if len(found)!=2:raise ValueError('Every original train/test command required: '+dataset)
        for chunk in found:
            argv=chunk.replace('\\\n',' ').split()
            parsed=parse_author_cli(config,argv)
            commands.append(dict(script=dataset+'.sh',argv=argv,dataset=parsed.dataset,mode=parsed.mode,
                win_size=parsed.win_size,num_epochs=parsed.num_epochs,seed=parsed.seed,batch_size=parsed.batch_size,
                anormly_ratio=parsed.anormly_ratio,ignored_script_threshold=parsed.threshold))
    text=(source/'scripts/SMD.sh').read_text(encoding='utf-8')
    for marker in ["machines=('machine-3-11')",'for win_size in $(seq 50 10 130);',
                   'for epoch in {1..5};','for anormly_ratio in $(seq 0.05 0.05 0.4);']:
        if marker not in text:raise ValueError('Exact full released SMD search grid required')
    assignments=dict(re.findall(r'^([a-z_]+)=([^\n]+)$',text,re.M))
    chunks=re.findall(r'python main\.py(.*?)(?=\n\s*for anormly_ratio|\n\s*done)',text,re.S)
    if len(chunks)!=2:raise ValueError('Original full SMD train and test templates required')
    for win in range(50,131,10):
        for epochs in range(1,6):
            for mode,ratios in [('train',[None]),('test',[i/20 for i in range(1,9)])]:
                for ratio in ratios:
                    values=dict(assignments,win_size=str(win),epoch=str(epochs),machine='machine-3-11',
                        anormly_ratio=str(ratio),train_mode='train',test_mode='test')
                    chunk=chunks[0 if mode=='train' else 1].replace('\\\n',' ')
                    argv=re.sub(r'\$([a-z_]+)',lambda m:values[m.group(1)].strip('"'),chunk).split()
                    parsed=parse_author_cli(config,argv)
                    commands.append(dict(script='SMD.sh',argv=argv,dataset=parsed.dataset,mode=parsed.mode,
                        win_size=win,num_epochs=epochs,seed=parsed.seed,batch_size=parsed.batch_size,
                        anormly_ratio=parsed.anormly_ratio,ignored_script_threshold=parsed.threshold))
    if (len(commands)!=413 or sum(c['mode']=='train' for c in commands)!=49
            or sum(c['mode']=='test' for c in commands)!=364):
        raise ValueError('All49 released training commands and364 complete test commands required')
    return commands


def execute_author_command(config,argv,working_directory):
    """Run the full original frontend in a new job process, after external admission.

    Training/test processes must be distinct, as in the released scripts. This
    function does not grant resources or declare data/environment equivalence.
    """
    source=verify(config);parsed=parse_author_cli(config,argv)
    output=(ROOT/config['runtime_output_root']).resolve()
    directory=Path(working_directory).resolve();checkpoint=Path(parsed.model_save_path).resolve()
    if (not directory.is_relative_to(output) or directory.exists() or not checkpoint.is_relative_to(output)
            or not Path(parsed.data_path).is_absolute() or parsed.mode not in ('train','test')):
        raise ValueError('Fresh external job directory, explicit data view and preserved job checkpoint path required')
    directory.mkdir(parents=True);(directory/'result').mkdir()
    os.chdir(directory);sys.path.insert(0,str(source));sys.argv=[str(source/'main.py')]+argv
    namespace=dict(__name__='__main__',__file__=str(source/'main.py'))
    exec(compile(corrected_main_ast((source/'main.py').read_text(encoding='utf-8')),str(source/'main.py'),'exec'),namespace)
