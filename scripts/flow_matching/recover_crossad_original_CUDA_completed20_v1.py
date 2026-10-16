"""Continue the actual completed20 CUDA training at its failed evaluation.

Reuse the pinned best checkpoint and real full training trace. Import the
already verified lazy optional-PATE evaluator in this process only. No retrain.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from types import FunctionType

from scripts.flow_matching import run_crossad_original_cuda_first_full_v1 as first
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha,write
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

MODULE='scripts.flow_matching.recover_crossad_original_CUDA_completed20_v1'
CONFIG='configs/experiments/fm_crossad_original_CUDA_completed20_evaluation_recovery.v1.json'


def registration():
    base=read(ROOT/first.CONFIG)
    old=ROOT/base['job']['output_directory']
    paths=[first.CONFIG,MODULE.replace('.','/')+'.py','tests/test_fm_crossad_CUDA_completed20_recovery_v1.py',
        'src/iia_benchmark/models/fm_crossad_evaluator_compat.py',
        'docs/reports/fm_crossad_evaluator_compat_2026-10-09.json',
        str(old/'training_trace.json'),str(old/'stages/weights/receipt.json'),
        str(old/'workspace/configs/GECCO/checkpoints_0/checkpoint.pth'),
        str(old/'workspace/test_results/GECCO/results_0/a_gt.npy'),
        str(old/'workspace/test_results/GECCO/results_0/a_test_energy.npy'),
        base['state_root']+'/train.guard_receipt.json',base['state_root']+'/train.stderr.log',base['state_root']+'/train.stdout.log']
    c=copy.deepcopy(base)
    c['state_root']='experiments/runs/fm_crossad_original_CUDA_completed20_evaluation_recovery_v1'
    c['artifact_root']='F:/aicoding/IIA_Data/experiments/crossad_original_CUDA_completed20_evaluation_recovery_v1'
    c['job'].update(id=base['job']['id']+'__completed20_evaluation_recovery_v1',runner=MODULE,
        output_directory=c['state_root']+'/author_job')
    c['settings']['output_root']=c['state_root']
    c.update(original_completed_training_configuration=first.CONFIG,original_training_output=str(old),
        optional_evaluator_root=read(ROOT/first.preflight.CONFIG)['cpu_reference_root']+'/ts_ad_evaluation',
        original_training_completed_epochs=20,original_training_forward_calls=8640,
        additional_training_updates_in_recovery=0,source_training_seconds_unknown=True,
        original_failed_training_and_evaluation_guard_preserved=True)
    pins={p['path']:p['sha256'] for p in base['source_receipts']}
    for path in paths:pins[path]=sha(ROOT/path)
    c['source_receipts']=[dict(path=p,sha256=h) for p,h in sorted(pins.items())]
    return c


def verify(c):
    if c!=registration():raise ValueError('Actual completed20 checkpoint,full trace,metric failure and unchanged recipe required')
    for item in c['source_receipts']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen completed-training dependency changed')
    previous=read(ROOT/first.CONFIG);old=Path(c['original_training_output'])
    guard=read(ROOT/previous['state_root']/'train.guard_receipt.json')
    trace=read(old/'training_trace.json');weights=read(old/'stages/weights/receipt.json')
    if (guard['passed'] or guard['exit_code']!=1 or guard['resource_guard_aborted']
            or trace['epochs_finished']!=20 or trace['training_forward_calls']!=8640
            or len(trace['validation_losses'])!=20 or weights['status']!='completed'
            or weights['details']['epochs_completed']!=20
            or weights['details']['checkpoint_sha256']!=sha(old/'workspace/configs/GECCO/checkpoints_0/checkpoint.pth')):
        raise ValueError('Original full training must be complete,finite and preserved before evaluation recovery')
    error=(ROOT/previous['state_root']/'train.stderr.log').read_text()
    if "cannot import name '_binary_clf_curve'" not in error:raise ValueError('Pinned observed optional-evaluator failure required')
    import psutil
    for pid,created in [(27456,1792126992.0307407),(19408,1792126993.355283),(32224,1792126993.4017453)]:
        try:
            p=psutil.Process(pid)
            if p.create_time()==created:raise ValueError('Original source process is still live;do not restart')
        except psutil.NoSuchProcess:pass


def install_evaluator(c):
    from iia_benchmark.models.fm_crossad_evaluator_compat import patch_evaluator
    from iia_benchmark.models.fm_crossad_author_runtime import import_author
    import_author(ROOT/c['settings']['runtime_root'])
    original=ROOT/c['settings']['runtime_root']/'ts_ad_evaluation'
    target=ROOT/c['optional_evaluator_root']
    for path in original.rglob('*.py'):
        relative=path.relative_to(original)
        expected=patch_evaluator(path.read_text(encoding='utf-8')) if relative.as_posix()=='evaluator.py' else path.read_text(encoding='utf-8')
        if (target/relative).read_text(encoding='utf-8')!=expected:
            raise ValueError('Only the existing proven optional-PATE import patch may differ')
    if 'pate' in c['settings']['author_metrics']:
        raise ValueError('Requested PATE compatibility requires its own full verification')
    if 'ts_ad_evaluation' in sys.modules:raise ValueError('Evaluator must be selected before any old package import')
    spec=importlib.util.spec_from_file_location('ts_ad_evaluation',target/'__init__.py',submodule_search_locations=[str(target)])
    module=importlib.util.module_from_spec(spec);sys.modules['ts_ad_evaluation']=module;spec.loader.exec_module(module)
    if any(name.startswith('ts_ad_evaluation.pate') for name in sys.modules):raise ValueError('Unused PATE was still eagerly imported')
    return dict(evaluator_source=str(target),all_requested_metric_implementations_unchanged=True,
        requested_author_metrics=c['settings']['author_metrics'],PATE_requested=False,unused_PATE_import_deferred=True)


class CompletedForwardCount:
    def __init__(self,count):self.count=count
    def __len__(self):return self.count
    def append(self,value):raise ValueError('Recovery may not execute an additional training forward')


def evaluation_only_function(run,namespace,restore,trace):
    tree=ast.parse(inspect.getsource(run));changes=[]
    class Recover(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='(losses, validation_losses)':
                node.value=ast.parse('restored_training_accounting()',mode='eval').body;changes.append('trace')
            elif len(node.targets)==1 and ast.unparse(node.targets[0])=='training_seconds' and ast.unparse(node.value)=='time.monotonic() - started':
                node.value=ast.Constant(None);changes.append('unknown_original_training_seconds')
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='SimpleNamespace':
                for kw in node.keywords:
                    if kw.arg=='use_gpu' and ast.unparse(kw.value)=='False':
                        kw.value=ast.Constant(True);changes.append('use_gpu')
                    elif kw.arg=='gpu_type' and ast.unparse(kw.value)=="'cpu'":
                        kw.value=ast.Constant('cuda');changes.append('gpu_type')
                    elif kw.arg=='device' and ast.unparse(kw.value)=="torch.device('cpu')":
                        kw.value=ast.parse("torch.device('cuda:0')",mode='eval').body;changes.append('device')
            if ast.unparse(node.func)=='exp.train':
                node=ast.Call(func=ast.Name(id='restore_completed_training',ctx=ast.Load()),args=[ast.Name(id='exp',ctx=ast.Load())],keywords=[])
                changes.append('restore_exact_best_checkpoint')
            return node
    tree=Recover().visit(tree)
    if sorted(changes)!=['device','gpu_type','restore_exact_best_checkpoint','trace','unknown_original_training_seconds','use_gpu']:
        raise ValueError('Only already completed training accounting and restore may replace the train call')
    ns=dict(namespace,restore_completed_training=restore,
        restored_training_accounting=lambda:(CompletedForwardCount(trace['training_forward_calls']),list(trace['validation_losses'])))
    exec(compile(ast.fix_missing_locations(tree),__file__+'::continue_after_completed20','exec'),ns)
    return ns['run']


def worker(c,phase):
    if phase=='replay':first.worker(c,phase);return
    first.precision(c)
    evaluator=install_evaluator(c)
    from scripts.flow_matching import run_crossad_author_job as original
    from iia_benchmark.models.fm_crossad_author_runtime import load_release
    _,scores=first.configured_functions(original)
    old=Path(c['original_training_output']);trace=read(old/'training_trace.json');loaded=[]
    output=ROOT/c['job']['output_directory']
    if not output.is_junction() or output.resolve()!=Path(c['artifact_root'])/'author_job':raise ValueError('Fresh F artifact junction required')
    shutil.copy2(old/'training_trace.json',output/'training_trace.json')
    def restore(exp):
        destination=Path(exp.model_save_path)/'checkpoint.pth'
        if destination.exists():raise FileExistsError('Never overwrite a checkpoint')
        shutil.copy2(old/'workspace/configs/GECCO/checkpoints_0/checkpoint.pth',destination)
        loaded.append(load_release(exp.model,destination))
    def recorded(path,data):
        path=Path(path)
        if path.name=='execution_details.json':
            data.update(original_completed_training_epochs=20,original_training_forward_calls=8640,
                additional_training_updates_in_recovery=0,source_training_seconds_unknown=True,
                actual_best_epoch_from_original_validation_trace=1+min(range(20),key=lambda i:trace['validation_losses'][i]),
                resumed_original_checkpoint_strict_load=loaded)
        if path.name=='result.json':
            for name in ['a_gt.npy','a_test_energy.npy']:
                if sha(output/'workspace/test_results/GECCO/results_0'/name)!=sha(old/'workspace/test_results/GECCO/results_0'/name):
                    raise ValueError('All saved original GPU test predictions and labels must repeat byte-for-byte')
            data.update(recovery_of_original_completed_training=first.CONFIG,original_failed_guard_preserved=True,
                saved_original_GPU_test_arrays_byte_identical=True,optional_evaluator_compatibility=evaluator)
        original.write_json(path,data)
    namespace=dict(original.__dict__,native_scores=scores,write_json=recorded)
    run=evaluation_only_function(original.run,namespace,restore,trace)
    run(c['job'],c['settings'])
    exclusive(ROOT/c['state_root']/'source_execution_device_proof.json',dict(passed=True,
        original_CUDA_source=True,full_original_training_reused=True,actual_original_epochs=20,
        actual_original_forward_calls=8640,additional_training_updates=0,checkpoint_strict_load=loaded,
        all_saved_original_GPU_test_arrays_repeated=True,evaluator=evaluator))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--register',action='store_true');parser.add_argument('--phase',choices=['train','replay']);args=parser.parse_args()
    if args.register:exclusive(ROOT/CONFIG,registration());return
    c=read(ROOT/CONFIG);verify(c)
    if args.phase:worker(c,args.phase)
    else:
        controller=FunctionType(first.controller.__code__,dict(first.controller.__globals__,verify=verify,MODULE=MODULE),argdefs=first.controller.__defaults__)
        controller(c)


if __name__=='__main__':main()
