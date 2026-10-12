import ast
import inspect
import subprocess
import sys
from scripts.flow_matching import audit_full500_image_checkpoint_CPU_v2 as current
from scripts.flow_matching import audit_full401_image_checkpoint_CPU_v1 as prior


def reverse(tree):
    for node in ast.walk(tree):
        if not isinstance(node,ast.Constant):continue
        if node.value==500:node.value=401
        elif node.value==1500:node.value=1599
        elif isinstance(node.value,str):
            node.value=node.value.replace('epoch_0500.pt','resume.pt').replace('500','401').replace('1500','1599')
    return tree


def test_every_original_model_optimizer_rng_environment_and_full_epoch_gate_preserved():
    for name in ('verify','worker'):
        new=reverse(ast.parse(inspect.getsource(getattr(current,name))))
        old=ast.parse(inspect.getsource(getattr(prior,name)))
        assert ast.dump(new)==ast.dump(old)


def test_cpu_restore_does_not_call_any_GPU_restore_or_training_operation():
    tree=ast.parse(inspect.getsource(current.worker))
    cuda_calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('torch.cuda.')]
    assert cuda_calls==['torch.cuda.is_initialized']
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func).endswith('.step') for n in ast.walk(tree))


def test_cold_controller_import_does_not_load_numerical_model_or_CUDA_backend():
    code="import scripts.flow_matching.run_full500_image_checkpoint_CPU_audit_v2,sys;assert not {'numpy','torch'}&set(sys.modules)"
    assert subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True).returncode==0
