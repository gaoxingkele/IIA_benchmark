"""Preserve all three original CPU queues in the existing bounded serial lane."""
import argparse
import ast
import json
import sys
from types import FunctionType

from scripts.flow_matching import run_resource_gated_light_controller_v1 as original
from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import ROOT, capped_class_ast, digest, validate_limits


SPECS = {'entropic_cpu': (105, 4), 'pi_cpu': (1140, 2), 'moment_cpu': (72, 2)}


def bind(registration, name):
    if name not in SPECS or set(registration['controllers']) != set(SPECS):
        raise ValueError('All three remaining original CPU queues must be retained')
    parent_path = ROOT / registration['parent_resource_registration']
    strict_path = ROOT / registration['existing_bounded_strict_registration']
    if (digest(parent_path) != registration['parent_resource_registration_sha256']
            or digest(strict_path) != registration['existing_bounded_strict_registration_sha256']):
        raise ValueError('Original and already-live resource registrations must be frozen')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    strict = json.loads(strict_path.read_text(encoding='utf-8'))
    validate_limits(parent['settings'], registration['settings'])
    if (registration['original_runtime'] != parent['original_runtime']
            or registration['settings'] != strict['settings']
            or registration['resource_lock'] != strict['resource_lock']
            or registration['resource_lock'] == parent['resource_lock']):
        raise ValueError('Reuse the exact live bounded serial lane and all its safeguards')
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen remaining CPU orchestration source changed: ' + entry['path'])
    for key, (count, threads) in SPECS.items():
        selected = registration['controllers'][key]
        if selected != parent['controllers'][key]:
            raise ValueError('Original entire queue binding must remain unchanged')
        path = ROOT / selected['queue']
        queue = json.loads(path.read_text(encoding='utf-8'))
        actual_threads = (queue['settings'] if key == 'pi_cpu' else queue['evaluation'])['cpu_threads']
        if (digest(path) != selected['queue_sha256'] or len(queue['jobs']) != count
                or selected['registered_jobs'] != count or actual_threads != threads):
            raise ValueError('Every original job and its original thread budget must be retained')
    namespace = dict(vars(original))
    source = (ROOT / 'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text(encoding='utf-8')
    node = capped_class_ast(source)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(ROOT / 'scripts/flow_matching/run_resource_gated_light_controller_v1.py'), 'exec'), namespace)
    namespace['guarded_factory'] = FunctionType(original.guarded_factory.__code__, namespace)
    return FunctionType(original.main.__code__, namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', choices=SPECS, required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    function = bind(registration, args.controller)
    sys.argv = ['bounded-original-remaining-cpu', '--runtime-config', args.runtime_config,
                '--controller', args.controller] + (['--probe-only'] if args.probe_only else [])
    function()


if __name__ == '__main__':
    main()
