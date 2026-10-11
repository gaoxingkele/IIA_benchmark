"""Run full corrected-data CPU controls in the already-existing serial CPU lane."""
import argparse
import ast
import json
import os
import sys
from types import FunctionType, SimpleNamespace

import psutil

from scripts.flow_matching import run_resource_gated_light_controller_v1 as original_guard
from scripts.flow_matching.run_bounded_strict_cpu_queue_v2 import capped_class_ast
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
from scripts.flow_matching.run_light_controller import ROOT, digest, load_controller


def validate_queue(root, registration, name):
    for source in registration['source_receipts']:
        if digest(root / source['path']) != source['sha256']:
            raise ValueError('Frozen full corrected CPU orchestration differs')
    binding = next(b for b in registration['controllers'] if b['original_lane'] == name)
    queue = json.loads((root / binding['path']).read_text('utf-8'))
    if digest(root / binding['path']) != binding['sha256'] or len(queue['jobs']) != binding['jobs']:
        raise ValueError('Entire corrected original CPU queue must remain')
    original = json.loads((root / binding['original_source_queue']).read_text('utf-8'))
    if queue['evaluation'] != original['evaluation'] or queue['python'] != original['python']:
        raise ValueError('Original evaluation, threads and actual worker environment required')
    before = {j['id']: j for j in original['jobs']}
    selected = {j['id'] for j in original['jobs'] if j['dataset'] in registration['affected_datasets']}
    if {j['preprocessing_control_original_id'] for j in queue['jobs']} != selected:
        raise ValueError('Every original affected method/ablation/seed required')
    allowed = {'id', 'model_config', 'dataset_manifest_path', 'output_directory', 'frozen_files', 'preprocessing_control_original_id'}
    for job in queue['jobs']:
        old = before[job['preprocessing_control_original_id']]
        if (job['id'] != old['id'] + '__constant_fit_v2'
                or job['dataset_manifest_path'] != registration['corrected_manifest']
                or job['output_directory'] == old['output_directory']
                or job['model_config'] == old['model_config']):
            raise ValueError('Distinct corrected data, configuration and output required')
        if {k:v for k,v in job.items() if k not in allowed} != {k:v for k,v in old.items() if k not in allowed}:
            raise ValueError('Original complete scientific job changed')
        if job['parameter_overrides']['device'] != 'cpu':
            raise ValueError('Original GPU jobs require their original GPU gate')
        previous = json.loads((root / old['model_config']).read_text('utf-8'))
        current = json.loads((root / job['model_config']).read_text('utf-8'))
        if current['parameters'] != previous['parameters'] or current['entrypoint'] != previous['entrypoint']:
            raise ValueError('Same full model and original capacity/epoch budget required')
        frozen = {r['path']:r['sha256'] for r in job['frozen_files']}
        if any(frozen.get(r['path']) != r['sha256'] for r in old['frozen_files']):
            raise ValueError('Original source and old data receipts must remain')
        if (job['dataset_manifest_path'] not in frozen or job['model_config'] not in frozen):
            raise ValueError('Corrected manifest and model identity must be frozen')
        for source in job['frozen_files']:
            if digest(root / source['path']) != source['sha256']:
                raise ValueError('Frozen corrected data/source differs')
    strict = json.loads((root / registration['existing_bounded_cpu_registration']).read_text('utf-8'))
    if registration['settings'] != strict['settings'] or registration['resource_lock'] != strict['resource_lock']:
        raise ValueError('Reuse the exact existing serial CPU lock and all safeguards')
    return binding, queue


def bind(registration, name):
    binding, queue = validate_queue(ROOT, registration, name)
    parent = json.loads((ROOT / registration['original_controller_runtime']).read_text('utf-8'))
    controller, namespace, _ = load_controller(ROOT, parent, binding['original_controller_name'])
    guards = dict(vars(original_guard))
    node = capped_class_ast((ROOT / 'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text('utf-8'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'unchanged-original-guard-with-existing-caps', 'exec'), guards)
    factory = FunctionType(original_guard.guarded_factory.__code__, guards)
    return controller, namespace, binding, queue, factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.runtime_config
    registration = json.loads(path.read_text('utf-8'))
    controller, namespace, binding, queue, factory = bind(registration, args.controller)
    if args.probe_only:
        print(json.dumps(dict(original_lane=args.controller, complete_original_cpu_slots=len(queue['jobs']),
            original_cpu_threads=queue['evaluation']['cpu_threads'], numpy_imported='numpy' in sys.modules,
            torch_imported='torch' in sys.modules, same_existing_serial_resource_lock=True)))
        return
    base = ROOT / registration['state_root'] / args.controller
    api = resource_api()
    lane = api['exclusive_lock'](base / 'adapter.lock')
    if lane is None:
        raise RuntimeError('This corrected-data CPU controller already has a live owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), runtime_sha256=digest(path), state='starting')
    def update(values):
        state.update(values)
        original_guard.write(base / 'status.json', state)
    proxy = SimpleNamespace(**vars(namespace['subprocess']))
    proxy.Popen = factory(api, registration['settings'], ROOT / registration['resource_lock'], base, update, ROOT / binding['path'], queue)
    namespace['subprocess'] = proxy
    sys.argv = ['original-entire-corrected-data-controller', '--queue', binding['path']]
    try:
        code = controller.main()
        update(dict(state='controller_terminal', original_returncode=code, active_pid=None))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
