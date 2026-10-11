"""Bind all170 corrected-data GPU controls to the unchanged native GPU runner."""
import copy
import json

from scripts.flow_matching.run_light_controller import ROOT, digest


REGISTRATION = 'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2/full_recomparison_registration.json'
ORIGINAL_GPU = 'configs/runtime/fm_tsad_gpu_native_gpu_resource_resume.v1.json'
LIGHT = 'configs/runtime/fm_tsad_constant_fit_gpu_light_controller.v2.json'
RUNTIME = 'configs/runtime/fm_tsad_constant_fit_gpu_native_resource.v2.json'
REPORT = 'projects/flow_matching_research/results/2026-10-11/constant_fit_gpu_execution_v2'


def validate_scientific_controls(root, registry):
    binding = next(b for b in registry['queue_bindings'] if b['original_lane'] == 'gpu')
    queue = json.loads((root / binding['path']).read_text('utf-8'))
    before = json.loads((root / binding['original_source_queue']).read_text('utf-8'))
    if digest(root / binding['path']) != binding['sha256']:
        raise ValueError('Frozen all170 GPU queue differs')
    if queue['lane'] != 'gpu' or before['lane'] != 'gpu':
        raise ValueError('Keep original native GPU lane and phase gates')
    for field in ('evaluation', 'python', 'gpu_prerequisite_queues'):
        if queue[field] != before[field]:
            raise ValueError('Original full threads/environment/phase gates must remain')
    originals = {j['id']:j for j in before['jobs'] if j['dataset'] in registry['actual_affected_datasets']}
    identities = [j['preprocessing_control_original_id'] for j in queue['jobs']]
    if len(queue['jobs']) != 170 or len(set(identities)) != 170 or set(identities) != set(originals):
        raise ValueError('Every original affected GPU method/ablation/seed required')
    allowed = {'id','model_config','dataset_manifest_path','output_directory','frozen_files','preprocessing_control_original_id'}
    for job in queue['jobs']:
        original = originals[job['preprocessing_control_original_id']]
        if {k:v for k,v in job.items() if k not in allowed} != {k:v for k,v in original.items() if k not in allowed}:
            raise ValueError('Full original scientific CUDA job contract differs')
        if (job['parameter_overrides']['device'] != 'cuda:0'
                or job['dataset_manifest_path'] != registry['corrected_manifest']
                or job['id'] != original['id'] + '__constant_fit_v2'
                or job['model_config'] == original['model_config']
                or job['output_directory'] == original['output_directory']):
            raise ValueError('Distinct corrected data and full original CUDA model required')
        old_model = json.loads((root / original['model_config']).read_text('utf-8'))
        new_model = json.loads((root / job['model_config']).read_text('utf-8'))
        if old_model['parameters'] != new_model['parameters'] or old_model['entrypoint'] != new_model['entrypoint']:
            raise ValueError('Same original architecture and full training capacity required')
        frozen = {r['path']:r['sha256'] for r in job['frozen_files']}
        if any(frozen.get(r['path']) != r['sha256'] for r in original['frozen_files']):
            raise ValueError('All original data and algorithm provenance must remain')
        if job['dataset_manifest_path'] not in frozen or job['model_config'] not in frozen:
            raise ValueError('Corrected data and model must be independently frozen')
        for source in job['frozen_files']:
            if digest(root / source['path']) != source['sha256']:
                raise ValueError('Frozen full GPU data/source differs')
    return binding, queue


def write_new(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    registry = json.loads((ROOT / REGISTRATION).read_text('utf-8'))
    binding, queue = validate_scientific_controls(ROOT, registry)
    old = json.loads((ROOT / ORIGINAL_GPU).read_text('utf-8'))
    light = copy.deepcopy(json.loads((ROOT / old['original_runtime']).read_text('utf-8')))
    original_binding = light['controllers'][0]
    original_binding.update(name='tsad_gpu_constant_fit_v2', queue_path=binding['path'], queue_sha256=binding['sha256'],
        status_path=queue['state_root'] + '/gpu_status.json')
    original_binding.pop('expected_old_pid', None)
    original_binding.pop('expected_old_create_time', None)
    light['source_receipts'] += [dict(path=p, sha256=digest(ROOT / p)) for p in [binding['path'], REGISTRATION,
        'scripts/flow_matching/register_tsad_constant_fit_gpu_runtime_v2.py',
        'tests/test_fm_tsad_constant_fit_gpu_runtime_v2.py']]
    light['boundary'] = 'Full original TSAD GPU controller, prerequisites, phase gates, shared legacy queue lock and CUDA commands retained for all170 corrected SMAP methods/ablations/seeds. New independent dataset/config/output; no prior-controller handoff or model stopping.'
    write_new(LIGHT, light)
    runtime = copy.deepcopy(old)
    runtime.update(original_runtime=LIGHT, controller='tsad_gpu_constant_fit_v2',
        state_root='experiments/runs/fm_tsad_constant_fit_gpu_native_resource_v2',
        original_source_runtime=ORIGINAL_GPU, original_source_runtime_sha256=digest(ROOT / ORIGINAL_GPU),
        original_gpu_controls=170, new_completed_training_slots=0,
        corrected_manifest=registry['corrected_manifest'], original_scope=registry['original_scope'])
    runtime['source_receipts'] += [dict(path=p, sha256=digest(ROOT / p)) for p in [LIGHT, ORIGINAL_GPU,
        binding['path'], REGISTRATION, 'scripts/flow_matching/register_tsad_constant_fit_gpu_runtime_v2.py',
        'tests/test_fm_tsad_constant_fit_gpu_runtime_v2.py']]
    runtime['boundary'] = 'All170 original full CUDA methods/ablations/seeds on corrected full SMAP data. Exact unchanged native resource slot, GPU UUID/free/emergency safeguards, full original worker, phase gates, legacy lane lock, shared GPU mutex and35second yield. No algorithm/epoch/capacity/seed/device reduction or original-result replacement; waiting is not completed training. Full45paper scope remains.'
    write_new(RUNTIME, runtime)
    write_new(REPORT + '/registration.json', dict(configuration=RUNTIME, configuration_sha256=digest(ROOT / RUNTIME),
        light_controller_configuration=LIGHT, light_configuration_sha256=digest(ROOT / LIGHT), original_binding=binding,
        all170_full_scientific_controls_source_and_data_verified=True, original_GPU_settings_exact=runtime['settings'] == old['settings'],
        original_shared_GPU_lock_exact=runtime['resource_lock'] == old['resource_lock'],
        unchanged_native_GPU_runner='scripts/flow_matching/run_tsad_native_gpu_resource_queue_v1.py',
        original_queue_threads=queue['evaluation']['cpu_threads'], source_resource_setting_cpu_threads=old['settings']['cpu_threads'],
        original_full_worker_threads_retained=True, original_scope=registry['original_scope'],
        new_completed_training_slots=0, author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    print(json.dumps(dict(full_original_corrected_GPU_slots=170, original_worker_threads=queue['evaluation']['cpu_threads'],
        resource_lock=runtime['resource_lock'], unchanged_native_GPU_runner=True)))


if __name__ == '__main__':
    main()
