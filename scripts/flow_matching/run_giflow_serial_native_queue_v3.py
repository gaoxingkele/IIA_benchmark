"""Unchanged GiFlow full jobs after successful serial full-batch memory evidence."""
import ast
import json
import os
from pathlib import Path
import sys
import time
from scripts.flow_matching import run_giflow_guarded_gpu_queue_v2 as previous
from scripts.flow_matching.giflow_native_protocol import ROOT, sha, verify, write_json


def memory_preflight_ready(queue, root=ROOT):
    path = root / queue['memory_preflight_config']
    if sha(path) != queue['memory_preflight_config_sha256']:
        raise ValueError('Frozen serial memory preflight registration changed')
    config = json.loads(path.read_text(encoding='utf-8'))
    from scripts.flow_matching.giflow_cuda_memory_probe import verify_config
    verify_config(config, root)
    if config['resource_lock'] != queue['resource_lock']:
        raise ValueError('Memory preflight must share the native GPU training mutex')
    for profile in config['profiles']:
        output = root / profile['output_directory']
        result_path = output / 'cuda_memory_result.json'
        usage_path = root / config['state_root'] / (profile['id'] + '.resource_usage.json')
        if not result_path.exists() or not usage_path.exists():
            return False
        result = json.loads(result_path.read_text(encoding='utf-8'))
        usage = json.loads(usage_path.read_text(encoding='utf-8'))
        if (result['status'] != 'cuda_full_batch_memory_preflight_passed_not_benchmark'
                or result['probe_config_sha256'] != sha(path) or result['formal_metrics_published']
                or result['full_registered_batch_size'] != 128 or result['optimizer_updates'] != 1
                or result['integration_result_sha256'] != sha(output / 'integration_result.json')
                or usage['exit_code'] != 0 or usage['resource_guard_aborted']
                or usage['result_sha256'] != sha(result_path)):
            raise ValueError('Full-batch CUDA memory preflight evidence is incomplete')
        margin = 1024 ** 3
        if (queue['settings']['minimum_commit_headroom_bytes'] <
                usage['peak_tree_private_bytes'] + queue['settings']['emergency_commit_headroom_bytes'] + margin
                or queue['settings']['minimum_gpu_free_bytes'] <
                result['peak_cuda_reserved_bytes'] + queue['settings']['emergency_gpu_free_bytes'] + margin):
            return False
    return True


def bound_main():
    tree = ast.parse(Path(previous.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bound_main')
    replacements = 0
    class DefaultBinding(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal replacements
            if node.value == 'configs/experiments/fm_giflow_native_queue.v2.json':
                replacements += 1
                return ast.copy_location(ast.Constant('configs/experiments/fm_giflow_native_queue.v3.json'), node)
            return node
    function = DefaultBinding().visit(function)
    if replacements != 1:
        raise ValueError('Original native dispatch default changed')
    namespace = dict(vars(previous))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(previous.__file__), 'exec'), namespace)
    return namespace['bound_main']()


def main():
    path = sys.argv[sys.argv.index('--queue') + 1]
    queue = json.loads((ROOT / path).read_text(encoding='utf-8'))
    verify(queue)
    if '--verify-only' not in sys.argv:
        while not memory_preflight_ready(queue):
            write_json(ROOT / queue['state_root'] / 'status.json', {
                'pid': os.getpid(), 'queue_sha256': sha(ROOT / path), 'active_pid': None,
                'state': 'waiting_serial_full_batch_memory_preflight', 'jobs': {}})
            time.sleep(10)
    bound_main()()


if __name__ == '__main__':
    main()
