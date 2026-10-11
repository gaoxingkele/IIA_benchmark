import ast
from pathlib import Path
import subprocess

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read
from scripts.flow_matching.run_crnngan_bounded_noop_queue_v2 import controller_tree


def test_complete_controller_retains_every_statement_except_native_worker_binding():
    from scripts.flow_matching import run_crnngan_native_original_queue_v2 as old
    tree = controller_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'scripts.flow_matching.crnngan_bounded_noop_runtime_v1':
            node.module = 'scripts.flow_matching.crnngan_native_training_runtime_v1'
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('scripts.flow_matching.run_crnngan_bounded_noop_queue_v2',
                                            'scripts.flow_matching.run_crnngan_native_original_queue_v2')
    assert ast.dump(tree) == ast.dump(ast.parse(Path(old.__file__).read_text(encoding='utf-8')))


def test_designated_python37_compiles_complete_dispatcher_without_host_dependencies():
    queue = read(ROOT/'configs/experiments/fm_crnngan_native_classical_released_queue.v2.json')
    code = ("import sys;sys.path.insert(0,"+repr(str(ROOT))+ ");"
            "from scripts.flow_matching.run_crnngan_bounded_noop_queue_v2 import controller_tree;"
            "compile(controller_tree(),'complete_native_dispatcher','exec');"
            "from scripts.flow_matching.audit_crnngan_full_epoch_noop_v3 import main;"
            "assert 'psutil' not in sys.modules;print('native_python37_bridge_passed')")
    result = subprocess.run([queue['python'], '-X', 'utf8', '-c', code], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout
    assert 'native_python37_bridge_passed' in result.stdout
