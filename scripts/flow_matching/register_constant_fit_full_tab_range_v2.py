"""Follow every one of all291 corrected-data models with original full TAB metrics."""
import copy
import json

from scripts.flow_matching.run_light_controller import ROOT, digest


SCIENCE = 'configs/experiments/fm_constant_fit_tab_range.sparse.v2.json'
PARENT = 'configs/runtime/fm_constant_fit_tab_range_light.v2.json'
RUNTIME = 'configs/runtime/fm_constant_fit_tab_range_readonly.v2.json'
REGISTRY = 'projects/flow_matching_research/results/2026-10-11/constant_fit_preprocessing_v2/full_recomparison_registration.json'
REPORT = 'projects/flow_matching_research/results/2026-10-11/constant_fit_gpu_execution_v2'


def derive_science(old, registry):
    new = copy.deepcopy(old)
    new.update(id='fm_constant_fit_full_TAB_range_sparse_v2',
        output_root='experiments/runs/fm_constant_fit_tab_range_sparse_v2',
        queue_paths=[b['path'] for b in registry['queue_bindings']],
        boundary='Full pinned TAB VUS/Affiliation on all291 original corrected SMAP method/ablation/seed outputs. Every entity/timestamp, all250thresholds, every integer buffer, saved original validation thresholds and undefined statuses retained. Original training/splits/data variants remain separate; not full TAB training-harness or author equivalence.')
    return new


def write_new(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    registry = json.loads((ROOT / REGISTRY).read_text('utf-8'))
    science = derive_science(json.loads((ROOT / 'configs/experiments/fm_tab_range_execution.sparse.v2.json').read_text('utf-8')), registry)
    for binding in registry['queue_bindings']:
        if digest(ROOT / binding['path']) != binding['sha256']:
            raise ValueError('Full corrected queue differs before TAB registration')
    new_sources = [REGISTRY, 'scripts/flow_matching/register_constant_fit_full_tab_range_v2.py',
        'tests/test_fm_constant_fit_full_tab_range_v2.py', *science['queue_paths']]
    science['source_receipts'] += [dict(path=p, sha256=digest(ROOT / p)) for p in new_sources]
    jobs = [j for p in science['queue_paths'] for j in json.loads((ROOT / p).read_text('utf-8'))['jobs']]
    if len(jobs) != 291 or len({j['id'] for j in jobs}) != 291:
        raise ValueError('Every corrected original method/ablation/seed evaluation required')
    write_new(SCIENCE, science)
    old_runtime = json.loads((ROOT / 'configs/runtime/fm_readonly_tab_range_queues.v2.json').read_text('utf-8'))
    parent = copy.deepcopy(json.loads((ROOT / old_runtime['parent_runtime']).read_text('utf-8')))
    binding = copy.deepcopy(next(b for b in parent['controllers'] if b['name'] == 'range_main'))
    binding.update(queue_path=SCIENCE, queue_sha256=digest(ROOT / SCIENCE),
        status_path=science['output_root'] + '/status.json')
    parent['controllers'] = [binding]
    parent['source_receipts'] += [dict(path=p, sha256=digest(ROOT / p)) for p in [SCIENCE, *new_sources]]
    parent['boundary'] = 'Unchanged complete original range evaluator main/worker/metrics; distinct corrected SMAP input queues and outputs. Original phase-independent saved-score followup, full threshold/buffer/entity contracts retained.'
    write_new(PARENT, parent)
    runtime = copy.deepcopy(old_runtime)
    runtime.update(parent_runtime=PARENT, parent_runtime_sha256=digest(ROOT / PARENT),
        controllers=dict(range_main=dict(queue=SCIENCE, queue_sha256=digest(ROOT / SCIENCE),
            status_path=science['output_root'] + '/status.json', registered_full_attempts=291,
            model_result_files_present=sum((ROOT / j['output_directory'] / 'result.json').exists() for j in jobs))),
        registered_full_attempts=291, model_result_files_present=sum((ROOT / j['output_directory'] / 'result.json').exists() for j in jobs))
    runtime['source_receipts'] += [dict(path=p, sha256=digest(ROOT / p)) for p in [PARENT, SCIENCE, *new_sources]]
    runtime['boundary'] = 'All291 corrected original model/ablation/seed full saved-score evaluations. Exact existing readonlyCPUmutex, strengthened physical/commit/RSS/private guards, unchanged original evaluator/math,250thresholds and every integer buffer. No training changes, reduced evaluation or extra independent seeds. Full45paper scope retained.'
    write_new(RUNTIME, runtime)
    write_new(REPORT + '/full_TAB_range_registration.json', dict(configuration=SCIENCE,
        configuration_sha256=digest(ROOT / SCIENCE), readonly_runtime=RUNTIME,
        readonly_runtime_sha256=digest(ROOT / RUNTIME), original_registered_full_evaluations=291,
        original_registered_scientific_queue_count=7, original_full_model_ids=[j['id'] for j in jobs],
        actual_cpu_threads=1, exact_original_metric_contract_preserved=True,
        existing_resource_lock=runtime['resource_lock'], unchanged_readonly_runner='scripts/flow_matching/run_readonly_tab_range_queue_v1.py',
        current_model_results=sum((ROOT / j['output_directory'] / 'result.json').exists() for j in jobs),
        original_scope=registry['original_scope'], completed_evaluations=0, additional_training_seed_slots=0,
        author_equivalence_certified=False, strict_TAB_result=False, all_original_experiments_complete=False))
    print(json.dumps(dict(full_corrected_model_TAB_evaluations=291, all_original_math_and_entity_coverage_retained=True,
        shared_original_readonly_cpu_lock=runtime['resource_lock'])))


if __name__ == '__main__':
    main()
