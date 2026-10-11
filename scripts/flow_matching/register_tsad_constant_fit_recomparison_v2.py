"""Register every original method/ablation/seed on affected full prepared data."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.run_light_controller import ROOT, digest


SCOPE_PATH = 'configs/reproducibility/fm_tsad_constant_fit_recomparison_scope.v2.json'
REPORT = 'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2'


def write_new(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def derive_job(original, model_config, manifest, output, config_receipts):
    job = copy.deepcopy(original)
    job.update(id=original['id'] + '__constant_fit_v2', model_config=model_config,
        dataset_manifest_path=manifest, output_directory=output + '/jobs/' + original['id'] + '__constant_fit_v2',
        preprocessing_control_original_id=original['id'])
    # Retain all original scientific/data provenance, including old archives.
    receipts = {r['path']: r['sha256'] for r in job['frozen_files']}
    for record in config_receipts:
        if record['path'] in receipts and receipts[record['path']] != record['sha256']:
            raise ValueError('Original source must not be silently overwritten')
        receipts[record['path']] = record['sha256']
    job['frozen_files'] = [dict(path=p, sha256=s) for p, s in receipts.items()]
    return job


def main():
    scope = json.loads((ROOT / SCOPE_PATH).read_text('utf-8'))
    proof_path = ROOT / scope['full_data_differential_required']
    proof = json.loads(proof_path.read_text('utf-8'))
    prep = json.loads((ROOT / scope['preparation_configuration']).read_text('utf-8'))
    state = json.loads((ROOT / prep['controller_directory'] / 'status.json').read_text('utf-8'))
    if state['state'] != 'completed' or state['exit_code'] != 0 or state['resource_guard_aborted']:
        raise ValueError('Full original preparation must finish successfully before registration')
    manifest = json.loads((ROOT / scope['corrected_manifest']).read_text('utf-8'))
    if digest(ROOT / scope['corrected_manifest']) != proof['corrected_manifest_sha256'] or len(proof['datasets']) != 7:
        raise ValueError('Full seven-dataset corrected preparation proof required')
    affected = {d['dataset'] for d in proof['datasets'] if d['original_input_sha256'] != d['corrected_input_sha256']}
    if affected != {d['dataset'] for d in proof['datasets'] if d['changed_constant_fit_entities']}:
        raise ValueError('Every changed dataset must have disclosed constant-fit evidence')
    datasets = {d['id']: d for d in manifest['datasets']}
    for data in datasets.values():
        if digest(ROOT / data['input_npz']) != data['input_sha256']:
            raise ValueError('Full corrected prepared input differs')
    configs, queues, selected_ids, receipts = {}, [], set(), []
    for group in scope['original_queue_groups']:
        original_path = ROOT / group['original_queue']
        if digest(original_path) != group['original_queue_sha256']:
            raise ValueError('Full original source queue differs')
        original = json.loads(original_path.read_text('utf-8'))
        candidates = [j for j in original['jobs'] if j.get('dataset_manifest_path') == scope['original_shared_manifest']]
        if [j['id'] for j in candidates] != group['original_full_job_ids'] or original['evaluation'] != group['original_evaluation']:
            raise ValueError('Original full method/seed/evaluation scope differs')
        selected = [j for j in candidates if j['dataset'] in affected]
        name = group['original_lane']
        output = 'experiments/runs/fm_tsad_constant_fit_recomparison_v2/' + name
        queue = copy.deepcopy(original)
        queue.update(state_root=output, jobs=[], original_queue=group['original_queue'],
            preprocessing_variant='constant_fit_v2', boundary='Every original affected dataset-method-ablation-seed complete control; same model budgets/devices/evaluation, separate corrected fit-only normalization. Not old canonical recovery, extra independent seed or author equivalence.')
        for before in selected:
            if before['id'] in selected_ids:
                raise ValueError('Original corrected preprocessing slot repeated')
            selected_ids.add(before['id'])
            path = before['model_config']
            if path not in configs:
                previous = json.loads((ROOT / path).read_text('utf-8'))
                new = copy.deepcopy(previous)
                new['id'] += '__constant_fit_v2'
                new['preprocessing_control'] = dict(original_model_config=path, original_model_config_sha256=digest(ROOT / path),
                    data_variant='constant_fit_v2', unchanged_model_entrypoint_and_parameters=True,
                    original_fidelity_limitations_preserved=True, author_equivalence_certified=False)
                destination = 'configs/models/fm_constant_fit_v2/' + Path(path).name
                write_new(destination, new)
                configs[path] = destination
            data = datasets[before['dataset']]
            additions = [SCOPE_PATH, 'scripts/flow_matching/prepare_tsad_constant_fit_v2.py',
                'scripts/flow_matching/register_tsad_constant_fit_recomparison_v2.py', scope['preparation_configuration'],
                scope['corrected_manifest'], data['input_npz'], configs[path]]
            frozen = [dict(path=p, sha256=digest(ROOT / p)) for p in additions]
            queue['jobs'].append(derive_job(before, configs[path], scope['corrected_manifest'], output, frozen))
        queue_path = 'configs/experiments/fm_tsad_constant_fit_' + name + '_queue.v2.json'
        write_new(queue_path, queue)
        queues.append(dict(original_lane=name, path=queue_path, sha256=digest(ROOT / queue_path),
            original_source_queue=group['original_queue'], jobs=len(selected), cpu_threads=original['evaluation']['cpu_threads'],
            all_original_devices=sorted({j['parameter_overrides']['device'] for j in selected}),
            original_controller_name='moment_cpu' if name == 'moment_pretrained' else 'strict_cpu'))
    expected = sum(group['candidate_dataset_counts'].get(dataset, 0) for group in scope['original_queue_groups'] for dataset in affected)
    if len(selected_ids) != expected:
        raise ValueError('Every affected original method/ablation/seed is required')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), original_scope=scope['original_scope'],
        original_candidate_jobs=scope['candidate_original_jobs'], actual_affected_datasets=sorted(affected),
        registered_full_corrected_preprocessing_slots=len(selected_ids), full_expected_original_affected_slots=expected,
        model_configs=list(configs.values()), queue_bindings=queues, corrected_manifest=scope['corrected_manifest'],
        corrected_manifest_sha256=digest(ROOT / scope['corrected_manifest']), full_data_differential_sha256=digest(proof_path),
        original_source_queue_and_model_budget_seed_device_evaluation_contracts_preserved=True,
        new_complete_corrected_training_slots=0, original_outputs_preserved=True,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False)
    write_new(REPORT + '/full_recomparison_registration.json', report)
    print(json.dumps(dict(affected_datasets=sorted(affected), full_corrected_slots=len(selected_ids), model_configurations=len(configs), queues=queues)))


if __name__ == '__main__':
    main()
