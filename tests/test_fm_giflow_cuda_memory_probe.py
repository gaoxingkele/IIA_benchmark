import ast
from pathlib import Path

from scripts.flow_matching import run_giflow_native_job as original
from scripts.flow_matching.giflow_cuda_memory_probe import bound_main


def test_cuda_probe_preserves_author_calls_and_one_batch_diagnostic_boundary():
    bound, changes = bound_main()
    assert changes == dict.fromkeys(['budget_device', 'cpu_override', 'batch_receipt', 'boundary'], 1)
    assert bound.__globals__['verify'] is original.verify
    assert bound.__globals__['configured_loader'] is original.configured_loader
    assert 'cpu_batch' not in bound.__code__.co_consts
    assert 'cuda_batch' in bound.__code__.co_consts
    assert 'integration_passed_not_benchmark' in bound.__code__.co_consts
    assert bound.__code__.co_names.count('run_experiment') == 1
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert len([n for n in ast.walk(function) if isinstance(n, ast.ClassDef) and n.name == 'DiagnosticData']) == 1


def test_cuda_probe_does_not_modify_frozen_formal_runner():
    text = Path(original.__file__).read_text(encoding='utf-8')
    assert "args.batch_size, args.training_epoch, args.device, args.cuda = 2, 1, 'cpu', False" in text
    assert 'torch.cuda.is_available = lambda: False' in text
    assert "artifact_root = output / 'diagnostic_predictions' if cli.integration_check" in text
