"""Check every full SDFormer-AR Table2 interface independently of Spectral resource failure."""
import argparse
import ast
import json
import os
from pathlib import Path

from scripts.flow_matching import run_spectral_table2_original_v1 as original
from scripts.flow_matching import register_spectral_table2_original_v1 as registration
from scripts.flow_matching.freeze_spectral_table2_environment_v1 import ROOT, sha
from scripts.flow_matching.spectral_table2_bootstrap_v1 import write


def sdformer_function():
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight_gpu')
    replacements = 0
    class CountBinding(ast.NodeTransformer):
        def visit_Compare(self, node):
            nonlocal replacements
            if (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name)
                and node.left.func.id == 'len' and len(node.left.args) == 1
                and isinstance(node.left.args[0], ast.Name) and node.left.args[0].id == 'proof'
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 9):
                replacements += 1
                node.comparators[0] = ast.copy_location(ast.Constant(6), node.comparators[0])
            return node
    function = CountBinding().visit(function)
    if replacements != 1:
        raise ValueError('Original full interface coverage assertion changed')
    namespace = dict(vars(original))
    old_workspace = namespace['workspace']
    def private_workspace(queue, destination):
        private = old_workspace(queue, Path(destination).with_name('gpu_workspace_sdformer_v1'))
        os.chdir(private)
        return private
    namespace['workspace'] = private_workspace
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(original.__file__), 'exec'), namespace)
    return namespace['preflight_gpu']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); queue = original.read(ROOT / args.queue)
    original.verify(queue); original.verify_environment(queue)
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve completed or failed registered diagnostic')
    full_pipelines = registration.source_pipelines
    def selected(source):
        pipelines = full_pipelines(source)
        groups = {key: stages for key, stages in pipelines.items() if key[0] == 'sdformer_ar'}
        if len(pipelines) != 6 or set(groups) != {('sdformer_ar', d) for d in ('sine', 'stock', 'mujoco')}:
            raise ValueError('All three original SDFormer pipelines are required')
        return groups
    registration.source_pipelines = selected
    try:
        sdformer_function()(queue, output)
    finally:
        registration.source_pipelines = full_pipelines
    proof = original.read(output)
    if (len(proof['native_full_shape_checks']) != 6
            or {r['method'] for r in proof['native_full_shape_checks']} != {'sdformer_vq', 'sdformer_ar'}):
        raise ValueError('Complete original SDFormer full-batch checks absent')
    proof.update(tested_methods=['sdformer_ar'], tested_datasets=['sine', 'stock', 'mujoco'],
        registered_queue_scope_retained=len(queue['jobs']), all_methods_gpu_preflight_passed=False,
        preflight_adapter=Path(__file__).relative_to(ROOT).as_posix(), preflight_adapter_sha256=sha(Path(__file__)),
        original_full_model_batch_and_optimizer_checks_retained=True,
        boundary='Only diagnostic dispatch/coverage is split by method. All30 original jobs remain registered. '
            'Original source functions, model sizes, full batches and optimizer calls are unchanged. '
            'Spectral model full-batch GPU readiness is still unproven; this proof covers SDFormer-AR only.')
    write(output, proof)


if __name__ == '__main__':
    main()
