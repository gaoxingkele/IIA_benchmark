"""Select one terminal full-job retry while retaining its entire original registry."""
import argparse
import ast
import copy
import inspect
import json
import sys

from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller
from scripts.flow_matching.audit_post_resource_incident_execution_v1 import process_identity


def full_queue_retry(original, queue, registration):
    selected_id = registration['selected_job_id']
    selected = [job for job in original['jobs'] if job['id'] == selected_id]
    if len(selected) != 1 or len(queue['jobs']) != len(original['jobs']):
        raise ValueError('Exactly one original canonical job and complete registry required')
    expected = copy.deepcopy(original)
    job = next(job for job in expected['jobs'] if job['id'] == selected_id)
    job['output_directory'] = registration['selected_output_directory']
    if registration['method'] == 'timegan':
        expected['queue_path'] = registration['queue']
        expected['state_root'] = registration['state_root']
    elif registration['method'] == 'crossad':
        expected['settings']['output_root'] = registration['state_root']
    else:
        raise ValueError('Only registered full native TimeGAN/CrossAD controllers supported')
    if expected != queue or job['output_directory'] == selected[0]['output_directory']:
        raise ValueError('Only selected output and controller metadata may change')
    return selected[0], job


def selected_main_ast(source):
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loops = [node for node in ast.walk(main) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'job'
        and ast.unparse(node.iter) == "queue['jobs']"]
    if len(loops) != 1:
        raise ValueError('Exactly one original complete-job dispatch loop required')
    loops[0].iter = ast.parse(
        "[job for job in queue['jobs'] if job['id'] == selected_job_id]", mode='eval').body
    return ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))


def bind(registration):
    for item in registration['source_receipts']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full retry registration changed: ' + item['path'])
    original = json.loads((ROOT / registration['original_queue']).read_text(encoding='utf-8'))
    queue = json.loads((ROOT / registration['queue']).read_text(encoding='utf-8'))
    canonical, retry = full_queue_retry(original, queue, registration)
    stopped = registration['stopped_native_handle']
    handle = process_identity(stopped['pid'], stopped['create_time'], stopped['command_token'])
    if handle['live']:
        raise ValueError('Preserve original active native training')
    evidence = json.loads((ROOT / registration['terminal_incident_evidence']).read_text(encoding='utf-8'))
    if evidence['original_job'] != canonical or evidence['actual_resource_observations']['resource_guard_aborted'] is not True:
        raise ValueError('Full original terminal memory-aborted job must bind the retry')
    if registration['method'] == 'timegan':
        from scripts.flow_matching import run_timegan_native_queue_v1 as controller
        from scripts.flow_matching.timegan_native_common_v1 import verify_queue
        verify_queue(queue)
        namespace = dict(controller.__dict__)
        source = inspect.getsource(controller.main)
    else:
        runtime = json.loads((ROOT / registration['light_binding']).read_text(encoding='utf-8'))
        controller, namespace, binding = load_controller(ROOT, runtime, 'crossad')
        namespace['bind_controller'](controller, ROOT, queue['settings'])
        namespace['waited_lock'] = controller.waited_lock
        namespace['wait_for_headroom'] = controller.wait_for_headroom
        source = (ROOT / binding['source_path']).read_text(encoding='utf-8')
    namespace['selected_job_id'] = registration['selected_job_id']
    exec(compile(selected_main_ast(source), '<unchanged full native main with selected retry>', 'exec'), namespace)
    return namespace['main'], queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    registration = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    function, queue = bind(registration)
    if args.probe_only:
        print(json.dumps(dict(method=registration['method'], complete_registry_jobs=len(queue['jobs']),
            selected_full_jobs=1, selected_job_id=registration['selected_job_id'],
            settings=queue['settings'], scientific_job_fields_changed=0,
            numpy_imported='numpy' in sys.modules, tensorflow_imported='tensorflow' in sys.modules,
            additional_independent_training_seed_slots=0)))
        return
    sys.argv = ['original-native-full-controller', '--queue', registration['queue']]
    function()


if __name__ == '__main__':
    main()
