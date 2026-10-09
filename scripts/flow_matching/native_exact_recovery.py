"""Bind resource-failed native attempts to original slots without changing experiment semantics."""
from __future__ import annotations
import ast
import json
from pathlib import Path

from scripts.flow_matching.grasp_protocol_runner import ROOT, sha


def components(track):
    if track == 'grasp':
        from scripts.flow_matching import grasp_protocol_runner as worker
        from scripts.flow_matching import run_grasp_protocol_queue as controller
    elif track == 'statistical':
        from scripts.flow_matching import mtsbench_stat_protocol as worker
        from scripts.flow_matching import run_mtsbench_stat_queue as controller
    else:
        raise ValueError('Unreviewed native recovery track')
    return worker, controller


def verify(queue, root=ROOT):
    worker, _ = components(queue['recovery_track'])
    original_path = root / queue['recovery_original_queue']
    if sha(original_path) != queue['recovery_original_queue_sha256']:
        raise ValueError('Original frozen experiment queue changed')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    worker.verify(original, root)
    for receipt in queue['source_receipts']:
        if sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Recovery source changed: ' + receipt['path'])
    originals = {j['id']: j for j in original['jobs']}
    mapped = []
    for job in queue['jobs']:
        original_id = job['recovery_original_job_id']
        original_job = originals[original_id]
        normalized = dict(job)
        normalized.pop('recovery_original_job_id')
        normalized['id'] = original_job['id']
        normalized['output_directory'] = original_job['output_directory']
        if normalized != original_job:
            raise ValueError('Recovery model/data/seed/full-budget experiment semantics changed')
        if job['output_directory'] == original_job['output_directory'] or job['id'] == original_id:
            raise ValueError('Recovery must preserve failed original artifacts')
        mapped.append(original_id)
    if len(set(mapped)) != len(mapped) or mapped != queue['recovery_original_job_ids']:
        raise ValueError('Recovery seed slots duplicated or changed')
    for receipt in queue['recovery_failure_receipts']:
        if receipt['original_job_id'] not in mapped or sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Original terminal failure evidence changed')
    if len(queue['recovery_failure_receipts']) != len(mapped):
        raise ValueError('Every recovery requires a recorded terminal original failure')
    # Every source, environment, dataset and original training parameter remains authoritative.
    allowed = {'jobs', 'source_receipts', 'state_root', 'artifact_root', 'settings', 'cohort_count',
               'recovery_track', 'recovery_original_queue', 'recovery_original_queue_sha256',
               'recovery_original_job_ids', 'recovery_failure_receipts', 'lock_poll_seconds', 'recovery_policy'}
    for key, value in original.items():
        if key not in allowed and queue.get(key) != value:
            raise ValueError('Original native queue binding changed: ' + key)
    for key, value in original['settings'].items():
        if queue['settings'][key] != value:
            raise ValueError('Original native resource policy changed: ' + key)
    return original


def worker_main(track):
    worker, _ = components(track)
    tree = ast.parse(Path(worker.__file__).read_text(encoding='utf-8'))
    definitions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                   for name in ['run', 'main']]
    namespace = dict(vars(worker), verify=verify)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(worker.__file__), 'exec'), namespace)
    return namespace['main']


def controller_main(track):
    worker, controller = components(track)
    tree = ast.parse(Path(controller.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    counts = {'worker': 0, 'resource_slot': 0, 'statistical_mutex': 0}
    worker_module = 'scripts.flow_matching.grasp_protocol_runner' if track == 'grasp' else 'scripts.flow_matching.mtsbench_stat_protocol'
    class Dispatch(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == worker_module:
                counts['worker'] += 1
                return ast.copy_location(ast.Constant('scripts.flow_matching.run_native_exact_recovery_job'), node)
            if track == 'statistical' and node.value == 'experiments/runs/fm_heavy_recovery_cpu.lock':
                counts['statistical_mutex'] += 1
                return ast.copy_location(ast.parse("queue['resource_lock']", mode='eval').body, node)
            return node
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Subscript) and isinstance(node.func.slice, ast.Constant)
                    and node.func.slice.value == 'resource_slot'):
                counts['resource_slot'] += 1
                node.keywords.append(ast.keyword(arg='poll_seconds', value=ast.parse("queue['lock_poll_seconds']", mode='eval').body))
            return self.generic_visit(node)
    function = Dispatch().visit(function)
    if counts != {'worker': 1, 'resource_slot': 1, 'statistical_mutex': int(track == 'statistical')}:
        raise ValueError('Original native recovery dispatch needs review: ' + str(counts))
    namespace = dict(vars(controller), verify=verify)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(controller.__file__), 'exec'), namespace)
    return namespace['main']
