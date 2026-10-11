"""Dispatch one unresolved full incident case through the unchanged bounded lane.

The complete13-case registry remains; other cases stay with their live owner.
Only the final dispatch loop gains an explicit selection, without changing the
worker, resource guard, original model or scientific settings.
"""
import argparse
import ast
import json
import sys

from scripts.flow_matching.run_light_controller import ROOT
from scripts.flow_matching.run_preserved_incident_recovery_v4 import bind
from scripts.flow_matching.run_readonly_tab_range_queue_v1 import readonly_main_ast


def selected_main_ast(source, resource_lock):
    function = readonly_main_ast(source, resource_lock)
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'case'
             and ast.unparse(n.iter) == "queue['jobs']"]
    if len(loops) != 1:
        raise ValueError('Exactly one original full-case dispatch loop required')
    loops[0].iter = ast.parse(
        "[case for case in queue['jobs'] if case['job']['id'] == selected_job_id]",
        mode='eval').body
    return ast.fix_missing_locations(function)


def bind_selected(registration):
    original_main, queue, binding = bind(registration)
    selected = [c for c in queue['jobs'] if c['job']['id'] == registration['selected_job_id']]
    if len(selected) != 1 or selected[0]['original_id'] != registration['selected_original_id']:
        raise ValueError('Exactly one registered original canonical case must be selected')
    if selected[0]['kind'] != 'pi' or selected[0]['job']['branch'] != 'causal_axis_repaired':
        raise ValueError('This registered full recovery is the causal-axis Pi control')
    namespace = original_main.__globals__
    namespace['selected_job_id'] = registration['selected_job_id']
    source = (ROOT/binding['source_path']).read_text(encoding='utf-8')
    function = selected_main_ast(source, registration['resource_lock'])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT/binding['source_path']), 'exec'), namespace)
    return namespace['main'], queue, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = json.loads((ROOT/args.runtime_config).read_text(encoding='utf-8'))
    function, queue, binding = bind_selected(registration)
    if args.probe_only:
        print(json.dumps(dict(full_original_incident_cases=len(queue['jobs']), selected_cases=1,
            selected_job_id=registration['selected_job_id'],
            numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
            unchanged_full_worker='scripts.flow_matching.run_resource_recovery_job',
            settings=queue['settings'], additional_training_seed_slots=0)))
        return
    sys.argv = [binding['source_path'], '--queue', registration['queue']]
    function()


if __name__ == '__main__':
    main()
