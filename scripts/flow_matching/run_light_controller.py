"""Execute pinned controller functions without importing unused data libraries.

Only orchestration functions are selected, from their unchanged source ASTs.
Model/evaluator children still execute their original frozen entry points.
"""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = {
    'scripts/flow_matching/prepare_tsad_execution.py': ['sha', 'write_json'],
    'scripts/flow_matching/phase_gate.py': ['check_queue'],
    'scripts/flow_matching/run_tsad_queue.py': ['exclusive_lock', 'complete_result'],
    'scripts/flow_matching/run_maelnet_author_queue.py': ['verify_artifacts'],
    'scripts/flow_matching/heavy_resources.py': ['resource_snapshot', 'has_headroom', 'waited_lock',
        'wait_for_headroom', 'terminate_tree', 'guarded_wait'],
    'scripts/flow_matching/heavy_scheduler_v2.py': ['resource_slot', 'bind_controller', 'run_bound_controller'],
    'scripts/flow_matching/tab_sparse_execution.py': ['config_hash', 'reuse_completed'],
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load_functions(root, path, names, namespace):
    tree = ast.parse((root / path).read_text(encoding='utf-8'), filename=path)
    selected = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in names:
        node = selected[name]
        # Compile the entire original definition, decorators/defaults included.
        unit = ast.Module(body=[node], type_ignores=[])
        exec(compile(unit, str(root / path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def load_controller(root, config, name):
    for record in config['source_receipts']:
        if digest(root / record['path']) != record['sha256']:
            raise ValueError('Frozen controller source changed: ' + record['path'])
    binding = next(b for b in config['controllers'] if b['name'] == name)
    if digest(root / binding['queue_path']) != binding['queue_sha256']:
        raise ValueError('Frozen queue changed')
    namespace = dict(globals(), ROOT=root, __name__='pinned_light_controller')
    for path, functions in FUNCTIONS.items():
        load_functions(root, path, functions, namespace)
    selected = load_functions(root, binding['source_path'], binding['source_functions'], namespace)
    return SimpleNamespace(ROOT=root, waited_lock=namespace['waited_lock'],
                           wait_for_headroom=namespace['wait_for_headroom'], **selected), namespace, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    controller, namespace, binding = load_controller(ROOT, config, args.controller)
    if args.probe_only:
        memory = psutil.Process().memory_info()
        print(json.dumps({'controller':args.controller, 'private_bytes':getattr(memory,'private',memory.rss),
                          'rss_bytes':memory.rss, 'numpy_imported':'numpy' in sys.modules,
                          'pandas_imported':'pandas' in sys.modules, 'torch_imported':'torch' in sys.modules,
                          'queue_sha256':binding['queue_sha256']}))
        return
    sys.argv = [binding['source_path'], binding['queue_argument'], binding['queue_path']]
    if binding['resource_scheduler_binding']:
        # The original binding function mutates controller attributes, while
        # compiled functions refer to the shared namespace: publish both names.
        queue = json.loads((ROOT / binding['queue_path']).read_text(encoding='utf-8'))
        namespace['bind_controller'](controller, ROOT, queue['settings'])
        namespace['waited_lock'] = controller.waited_lock
        namespace['wait_for_headroom'] = controller.wait_for_headroom
    controller.main()


if __name__ == '__main__':
    main()
