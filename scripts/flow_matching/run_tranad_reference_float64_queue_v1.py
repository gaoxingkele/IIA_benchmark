"""Run the complete five-dataset/five-seed author-precision TranAD control."""
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


def validate_queue(root, registration):
    parent = json.loads((root / registration['original_queue']).read_text(encoding='utf-8'))
    queue = json.loads((root / registration['queue']).read_text(encoding='utf-8'))
    originals = {j['id']: j for j in parent['jobs'] if j['id'].startswith('fm_strict_tranad__')}
    if (len(originals) != 25 or len(queue['jobs']) != 25 or queue['evaluation'] != parent['evaluation']
            or queue['evaluation']['cpu_threads'] != 2
            or {j['precision_control_original_id'] for j in queue['jobs']} != set(originals)):
        raise ValueError('Every original TranAD dataset and all five original seeds required')
    for job in queue['jobs']:
        before = originals[job['precision_control_original_id']]
        allowed = {'id', 'model_config', 'output_directory', 'frozen_files', 'precision_control_original_id'}
        if {k: v for k, v in job.items() if k not in allowed} != {k: v for k, v in before.items() if k not in allowed}:
            raise ValueError('Full original data, seed, window and parameter overrides must remain')
        old_config = json.loads((root / before['model_config']).read_text(encoding='utf-8'))
        new_config = json.loads((root / job['model_config']).read_text(encoding='utf-8'))
        if (new_config['parameters'] != old_config['parameters']
                or new_config['entrypoint'] != 'iia_benchmark.models.fm_tranad_reference_float64_v1:ReferenceFloat64TranADAdapter'
                or new_config['reproduction_status'] != 'author_float64_precision_local_protocol_adaptation_not_author_pipeline_equivalence'):
            raise ValueError('Only the declared float64 implementation may differ; original capacity/budgets required')
        old_sources = {x['path']: x['sha256'] for x in before['frozen_files'] if x['path'] != before['model_config']}
        new_sources = {x['path']: x['sha256'] for x in job['frozen_files']}
        if any(new_sources.get(path) != sha for path, sha in old_sources.items()):
            raise ValueError('Original frozen data and unchanged implementation sources must remain')
        for entry in job['frozen_files']:
            if digest(root / entry['path']) != entry['sha256']:
                raise ValueError('Frozen precision-control source/data differs')
    return queue


def bind(registration):
    for entry in registration['source_receipts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Frozen full precision-control orchestration source changed')
    strict = json.loads((ROOT / registration['existing_bounded_strict_registration']).read_text(encoding='utf-8'))
    if registration['resource_lock'] != strict['resource_lock'] or registration['settings'] != strict['settings']:
        raise ValueError('The existing serial resource mutex and every safeguard must remain')
    queue = validate_queue(ROOT, registration)
    parent = json.loads((ROOT / registration['parent_controller_runtime']).read_text(encoding='utf-8'))
    controller, namespace, binding = load_controller(ROOT, parent, 'strict_cpu')
    if binding['source_path'] != 'scripts/flow_matching/run_tsad_queue.py':
        raise ValueError('Original full TSAD runner required')
    guards = dict(vars(original_guard))
    node = capped_class_ast((ROOT / 'scripts/flow_matching/run_resource_gated_light_controller_v1.py').read_text(encoding='utf-8'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'original-guard-with-existing-caps', 'exec'), guards)
    factory = FunctionType(original_guard.guarded_factory.__code__, guards)
    return controller, namespace, queue, factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.runtime_config
    registration = json.loads(path.read_text(encoding='utf-8'))
    controller, namespace, queue, factory = bind(registration)
    if args.probe_only:
        print(json.dumps(dict(full_original_tranad_dataset_seed_controls=len(queue['jobs']), cpu_threads=2,
                              numpy_imported='numpy' in sys.modules, torch_imported='torch' in sys.modules,
                              resource_lock=registration['resource_lock'], same_seed_precision_variant_not_exact_float32_recovery=True)))
        return
    base = ROOT / registration['state_root']
    api = resource_api()
    lane = api['exclusive_lock'](base / 'adapter.lock')
    if lane is None:
        raise RuntimeError('Precision-control adapter already has a live owner')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), runtime_sha256=digest(path),
                 queue_sha256=digest(ROOT / registration['queue']), state='starting')
    def update(values):
        state.update(values)
        original_guard.write(base / 'status.json', state)
    proxy = SimpleNamespace(**vars(namespace['subprocess']))
    proxy.Popen = factory(api, registration['settings'], ROOT / registration['resource_lock'],
                         base, update, ROOT / registration['queue'], queue)
    namespace['subprocess'] = proxy
    sys.argv = ['original-full-tsad-precision-control', '--queue', registration['queue']]
    try:
        code = controller.main()
        update(dict(state='controller_terminal', original_returncode=code, active_pid=None))
    finally:
        lane.close()


if __name__ == '__main__':
    main()
