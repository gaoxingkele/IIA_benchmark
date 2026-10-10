"""Dispatch full GiFlow budgets with observed drop-last update accounting."""
import ast
import json
from pathlib import Path
import sys

from scripts.flow_matching import run_grasp_protocol_queue as guard
from scripts.flow_matching import giflow_native_protocol as protocol
from scripts.flow_matching.run_giflow_serial_native_queue_v3 import memory_preflight_ready
from scripts.flow_matching.giflow_update_budget_v2 import validate_observed_updates


def complete(job):
    if not protocol.complete(job):
        return False
    result = json.loads((protocol.ROOT / job['output_directory'] / 'result.json').read_text(encoding='utf-8'))
    validate_observed_updates(result['epochs_completed'], job['arguments']['training_epoch'],
        result['optimizer_updates'], result['native_train_loader'], result['epoch_optimizer_steps'])
    observation = protocol.ROOT / job['output_directory'] / 'training_observation.json'
    if not any(Path(item['path']).name == observation.name and item['sha256'] == protocol.sha(observation)
               for item in result['artifacts']):
        raise ValueError('Persisted observed full GiFlow updates missing from receipts')
    return True


def bound_main():
    source = Path(guard.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    changes = {
        'configs/experiments/fm_grasp_protocol_queue.v2.json': 'configs/experiments/fm_giflow_native_queue.v4.json',
        'scripts.flow_matching.grasp_protocol_runner': 'scripts.flow_matching.run_giflow_native_job_v2',
        'Full GRASP GPU queue already owned': 'Full GiFlow GPU queue already owned'
    }
    counts = dict.fromkeys(changes, 0)
    class Binding(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in changes:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(changes[node.value]), node)
            return node
    main = Binding().visit(main)
    if any(n != 1 for n in counts.values()):
        raise ValueError('Unchanged native GPU guard binding differs')
    # Yield only between full jobs, after the original GPU mutex is released.
    # This gives other registered full pipelines an opportunity to acquire it.
    loop = next(n for n in main.body if isinstance(n, ast.For))
    loop.body.append(ast.parse('time.sleep(queue["controller_yield_seconds"])').body[0])
    namespace = dict(vars(guard))
    namespace.update(ROOT=protocol.ROOT, complete=complete, verify=protocol.verify,
        sha=protocol.sha, write_json=protocol.write_json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['main']


def main():
    path = sys.argv[sys.argv.index('--queue') + 1]
    queue = json.loads((protocol.ROOT / path).read_text(encoding='utf-8'))
    protocol.verify(queue)
    if '--verify-only' not in sys.argv and not memory_preflight_ready(queue):
        raise ValueError('Unchanged full-batch native CUDA memory proof required')
    bound_main()()


if __name__ == '__main__':
    main()
