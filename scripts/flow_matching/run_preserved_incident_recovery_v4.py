"""Execute every incident case in the existing bounded serial full-model lane."""
import argparse
import ast
import copy
import json
import sys

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller, load_functions
from scripts.flow_matching.run_readonly_tab_range_queue_v1 import readonly_main_ast
from scripts.flow_matching.register_preserved_incident_recovery_v4 import validate_case


def capped_guard_ast(source):
    node = copy.deepcopy(next(n for n in ast.parse(source).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'guarded_wait'))
    conditions = [n for n in ast.walk(node) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "current < settings['emergency_commit_headroom_bytes']"]
    if len(conditions) != 1:
        raise ValueError('Exactly the original emergency guard must remain identifiable')
    conditions[0].test = ast.BoolOp(op=ast.Or(), values=[conditions[0].test,
        ast.parse("peaks['peak_tree_rss_bytes'] > settings['maximum_worker_rss_bytes']", mode='eval').body,
        ast.parse("peaks['peak_tree_private_bytes'] > settings['maximum_worker_private_bytes']", mode='eval').body])
    return ast.fix_missing_locations(node)


def bind(registration):
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen incident queue execution source changed: ' + entry['path'])
    parent = json.loads((ROOT / registration['parent_controller_runtime']).read_text(encoding='utf-8'))
    controller, namespace, binding = load_controller(ROOT, parent, 'recovery')
    if binding['source_path'] != 'scripts/flow_matching/run_resource_recovery.py':
        raise ValueError('Original full incident controller required')
    strict = json.loads((ROOT / registration['existing_bounded_strict_registration']).read_text(encoding='utf-8'))
    original = json.loads((ROOT / binding['queue_path']).read_text(encoding='utf-8'))
    queue = json.loads((ROOT / registration['queue']).read_text(encoding='utf-8'))
    if (registration['resource_lock'] != strict['resource_lock']
            or len(queue['jobs']) != 13 or len({c['original_id'] for c in queue['jobs']}) != 13
            or set(c['original_id'] for c in original['jobs']) - set(c['original_id'] for c in queue['jobs'])):
        raise ValueError('All original canonical cases and existing serial mutex required')
    settings = queue['settings']
    for key in ('minimum_available_memory_bytes', 'minimum_commit_headroom_bytes', 'emergency_commit_headroom_bytes'):
        if settings[key] < max(original['settings'][key], strict['settings'][key]):
            raise ValueError('Every original and existing serial-lane safeguard must remain or strengthen')
    for key in ('maximum_worker_rss_bytes', 'maximum_worker_private_bytes'):
        if settings[key] != strict['settings'][key]:
            raise ValueError('Existing bounded full-model caps must remain')
    if settings['minimum_commit_headroom_bytes'] < settings['emergency_commit_headroom_bytes'] + settings['maximum_worker_private_bytes']:
        raise ValueError('Full private cap must fit above the emergency reserve')
    for case in queue['jobs']:
        validate_case(ROOT, case)
    main = readonly_main_ast((ROOT / binding['source_path']).read_text(encoding='utf-8'), registration['resource_lock'])
    guard = capped_guard_ast((ROOT / 'scripts/flow_matching/heavy_resources.py').read_text(encoding='utf-8'))
    exec(compile(ast.Module(body=[guard, main], type_ignores=[]), str(ROOT / binding['source_path']), 'exec'), namespace)
    return namespace['main'], queue, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    function, queue, binding = bind(registration)
    if args.probe_only:
        print(json.dumps(dict(original_canonical_cases=len(queue['jobs']), additional_training_seed_slots=0,
                              queue_sha256=digest(ROOT / registration['queue']), resource_lock=registration['resource_lock'],
                              numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                              original_worker='scripts.flow_matching.run_resource_recovery_job')))
        return
    sys.argv = [binding['source_path'], '--queue', registration['queue']]
    function()


if __name__ == '__main__':
    main()
