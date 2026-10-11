import ast
from pathlib import Path
from scripts.flow_matching import run_crnngan_native_original_queue_v2 as original
from scripts.flow_matching.audit_crnngan_full_epoch_noop_v4 import NATIVE_ENVIRONMENT, host_execution_tree


def test_registered_environment_equals_actual_original_native_worker_launch():
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    popen = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'Popen')
    expression = next(k.value for k in popen.keywords if k.arg == 'env')
    actual = {k.arg: ast.literal_eval(k.value) for k in expression.keywords}
    assert NATIVE_ENVIRONMENT == actual


def test_native_comparison_launch_passes_the_frozen_environment():
    tree = host_execution_tree()
    popen = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'Popen')
    expression = next(k.value for k in popen.keywords if k.arg == 'env')
    assert ast.dump(expression) == ast.dump(ast.parse("dict(os.environ, **config['native_execution_environment'])", mode='eval').body)
