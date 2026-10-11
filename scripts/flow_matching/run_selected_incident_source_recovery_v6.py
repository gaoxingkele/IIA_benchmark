"""Run one stopped original Pi source case through the unchanged full bounded lane."""
import argparse
import ast
import json
import sys

from scripts.flow_matching.run_light_controller import ROOT
from scripts.flow_matching.run_preserved_incident_recovery_v4 import bind
from scripts.flow_matching.run_selected_incident_full_recovery_v5 import selected_main_ast


def bind_source(registration):
    original_main, queue, binding = bind(registration)
    cases = [case for case in queue['jobs'] if case['job']['id'] == registration['selected_job_id']]
    if (len(cases) != 1 or cases[0]['original_id'] != registration['selected_original_id']
            or cases[0]['kind'] != 'pi' or cases[0]['job']['branch'] != 'source_runtime'):
        raise ValueError('Exactly one unchanged full original Pi source canonical case required')
    namespace = original_main.__globals__
    namespace['selected_job_id'] = registration['selected_job_id']
    source = (ROOT / binding['source_path']).read_text(encoding='utf-8')
    function = selected_main_ast(source, registration['resource_lock'])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / binding['source_path']), 'exec'), namespace)
    return namespace['main'], queue, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = json.loads((ROOT / args.runtime_config).read_text(encoding='utf-8'))
    function, queue, binding = bind_source(registration)
    if args.probe_only:
        print(json.dumps(dict(full_original_incident_cases=len(queue['jobs']), selected_cases=1,
            selected_job_id=registration['selected_job_id'], settings=queue['settings'],
            original_full_worker='scripts.flow_matching.run_resource_recovery_job',
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            additional_independent_training_seed_slots=0)))
        return
    sys.argv = [binding['source_path'], '--queue', registration['queue']]
    function()


if __name__ == '__main__':
    main()
