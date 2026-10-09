import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_cuda_allocator_cold_start_reset_works_with_reviewed_initialization():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware required for cold-start allocator integration')
    environment={k.upper():v for k,v in os.environ.items()}
    environment.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    code='from scripts.flow_matching.grasp_protocol_cuda_runtime import initialize_cuda;initialize_cuda();import torch;torch.cuda.reset_peak_memory_stats(0);print(torch.cuda.max_memory_allocated(0))'
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],env=environment,
                          capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='0'


def test_runtime_dispatch_binding_changes_only_queue_default_and_child_entrypoint():
    from scripts.flow_matching.run_grasp_protocol_queue_v3 import bound_main
    from scripts.flow_matching.run_grasp_protocol_queue import main as original
    bound=bound_main()
    changes={('configs/experiments/fm_grasp_protocol_queue.v2.json','configs/experiments/fm_grasp_protocol_queue.v3.json'),
             ('scripts.flow_matching.grasp_protocol_runner','scripts.flow_matching.grasp_protocol_cuda_runtime')}
    pairs={(a,b) for a,b in zip(original.__code__.co_consts,bound.__code__.co_consts) if a!=b}
    assert pairs==changes
    import ast
    tree=ast.parse(Path(sys.modules[original.__module__].__file__).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),original.__code__.co_filename,'exec'),namespace)
    # Same compile context avoids module symbol-table ordering differences.
    assert bound.__code__.co_code==namespace['main'].__code__.co_code
