"""Run unchanged full saved-score evaluations in a separate serial CPU lane."""
import argparse
import ast
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller

OLD_LOCK = 'experiments/runs/fm_heavy_recovery_cpu.lock'
NAMES = ('range_main', 'range_reflow', 'range_industrial', 'range_moment')


def stronger_limits(original, registered):
    for field in ['minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes']:
        if registered[field] < original[field]:
            raise ValueError('Every original physical/commit safeguard must be retained or strengthened')
    if (registered['maximum_worker_rss_bytes'] <= 0
            or registered['maximum_worker_private_bytes'] < registered['maximum_worker_rss_bytes']
            or registered['minimum_commit_headroom_bytes'] < registered['emergency_commit_headroom_bytes']
            + registered['maximum_worker_private_bytes']):
        raise ValueError('Bounded readonly worker requires enough commit for its cap plus emergency reserve')
    return copy.deepcopy(registered)


def readonly_main_ast(source, lock):
    tree = ast.parse(source)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    function = copy.deepcopy(original)
    count = 0
    class Mutex(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal count
            if node.value == OLD_LOCK:
                count += 1
                return ast.copy_location(ast.Constant(lock), node)
            return node
    function = Mutex().visit(function)
    if count != 1 or lock == OLD_LOCK:
        raise ValueError('Exactly the original evaluation mutex may change; training mutex stays untouched')
    return ast.fix_missing_locations(function)


def bind(root, runtime, name, api=None):
    if name not in NAMES:
        raise ValueError('Only the four original readonly range queues may enter')
    for receipt in runtime['source_receipts']:
        if digest(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen readonly orchestration input changed: ' + receipt['path'])
    path = root / runtime['parent_runtime']
    if digest(path) != runtime['parent_runtime_sha256']:
        raise ValueError('Original frozen runtime changed')
    parent = json.loads(path.read_text(encoding='utf-8'))
    _, namespace, binding = load_controller(root, parent, name)
    if (binding['source_path'] != 'scripts/flow_matching/run_tab_ranges_sparse_queue.py'
            or binding['source_functions'] != ['main'] or binding['queue_argument'] != '--config'):
        raise ValueError('Unchanged full original sparse evaluation controller required')
    selected = runtime['controllers'][name]
    if selected['queue'] != binding['queue_path'] or selected['queue_sha256'] != binding['queue_sha256']:
        raise ValueError('Frozen scientific evaluation configuration changed')
    config = json.loads((root / selected['queue']).read_text(encoding='utf-8'))
    if config['cpu_threads'] != runtime['cpu_threads']:
        raise ValueError('Original evaluator thread count must remain unchanged')
    limits = stronger_limits(config['runtime_resources'], runtime['settings'])
    api = resource_api() if api is None else api
    @contextmanager
    def slot(path, settings, update):
        if Path(path).resolve() != (root / runtime['resource_lock']).resolve() or settings != config['runtime_resources']:
            raise ValueError('Only the unchanged readonly score evaluator may acquire this lane')
        with api['resource_slot'](path, limits, update) as snapshot:
            yield snapshot
    def guard(child, settings, update):
        if settings != config['runtime_resources']:
            raise ValueError('Original evaluator safeguards changed')
        def evaluation_update(values):
            values = dict(values)
            if values.get('state') == 'replaying_full_checkpoints':
                values['state'] = 'evaluating_full_saved_scores'
            update(values)
        return capped_wait(child, dict(settings=limits), api, evaluation_update)
    namespace.update(resource_slot=slot, guarded_wait=guard)
    source = (root / binding['source_path']).read_text(encoding='utf-8')
    function = readonly_main_ast(source, runtime['resource_lock'])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / binding['source_path']), 'exec'), namespace)
    return namespace['main'], config, limits, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', choices=NAMES, required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    runtime = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    function, config, limits, binding = bind(ROOT, runtime, args.controller)
    if args.probe_only:
        counts = [len(json.loads((ROOT / path).read_text(encoding='utf-8'))['jobs']) for path in config['queue_paths']]
        print(json.dumps(dict(controller=args.controller, original_queue=binding['queue_path'],
            registered_full_attempts=sum(counts), cpu_threads=config['cpu_threads'], safeguards=limits,
            scientific_config_sha256=digest(ROOT / binding['queue_path']),
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            unchanged_original_evaluator=True, additional_training_seed_slots=0)))
        return
    sys.argv = [binding['source_path'], '--config', binding['queue_path']]
    function()


if __name__ == '__main__':
    main()
