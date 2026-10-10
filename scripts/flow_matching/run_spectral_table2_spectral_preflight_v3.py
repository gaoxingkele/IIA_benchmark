"""Full original Spectral model/batch checks, separately from completed SDFormer proof."""
import argparse
import ast
import json
import os
from pathlib import Path

from scripts.flow_matching import run_spectral_table2_original_v1 as original
from scripts.flow_matching import register_spectral_table2_original_v1 as registration
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT,sha
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def selected_pipelines(pipelines):
    selected={key:stages for key,stages in pipelines.items() if key[0]=='spectral_flow'}
    if len(pipelines)!=6 or set(selected)!={('spectral_flow',d) for d in ('sine','stock','mujoco')}:
        raise ValueError('All three original full Spectral pipelines are required')
    return selected


def full_spectral_function():
    source=Path(original.__file__)
    function=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='preflight_gpu')
    replacements=0
    class Binding(ast.NodeTransformer):
        def visit_Compare(self,node):
            nonlocal replacements
            if (isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Name)
                and node.left.func.id=='len' and len(node.left.args)==1
                and isinstance(node.left.args[0],ast.Name) and node.left.args[0].id=='proof'
                and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)
                and node.comparators[0].value==9):
                replacements+=1
                node.comparators[0]=ast.copy_location(ast.Constant(3),node.comparators[0])
            return node
    function=Binding().visit(function)
    if replacements!=1:raise ValueError('Original full-method count assertion changed')
    namespace=dict(vars(original))
    old_workspace=namespace['workspace']
    def private_workspace(queue,destination):
        private=old_workspace(queue,Path(destination).with_name('gpu_workspace_spectral_v3'))
        os.chdir(private)
        return private
    namespace['workspace']=private_workspace
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['preflight_gpu']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();queue=original.read(ROOT/args.queue)
    original.verify(queue);original.verify_environment(queue)
    output=ROOT/args.output
    if output.exists():raise FileExistsError('Preserve existing capacity proof or partial output')
    pipelines=registration.source_pipelines
    registration.source_pipelines=lambda private:selected_pipelines(pipelines(private))
    try:full_spectral_function()(queue,output)
    finally:registration.source_pipelines=pipelines
    proof=original.read(output)
    if (len(proof['native_full_shape_checks'])!=3
            or {r['method'] for r in proof['native_full_shape_checks']}!={'spectral_flow'}):
        raise ValueError('All full Spectral model/batch checks are required')
    proof.update(tested_methods=['spectral_flow'],tested_datasets=['sine','stock','mujoco'],
        registered_queue_scope_retained=len(queue['jobs']),all_methods_gpu_preflight_passed=False,
        preflight_adapter=Path(__file__).relative_to(ROOT).as_posix(),preflight_adapter_sha256=sha(Path(__file__)),
        original_full_model_batch_and_optimizer_checks_retained=True,
        boundary='Method-specific diagnostic, not benchmark performance. All30 original scientific jobs remain registered. Three full Spectral forward/backward/optimizer steps and three complete original eight-worker loaders; full source architecture, batch, updates and sampler retained. Completed SDFormer capacity proof remains separate.')
    write(output,proof)


if __name__=='__main__':main()
