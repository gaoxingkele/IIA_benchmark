"""Register immutable full-array and full-model audits of one completed ablation."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.audit_grasp_checkpoint_replay_v1 import ROOT, read, sha, verify_registration


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    canonical = 'grasp_protocol_v3__tau_0__swan__swan_sf__seed1104'
    result_path = f'experiments/runs/fm_grasp_protocol_v3/jobs/{canonical}/result.json'
    replay_path = 'configs/reproducibility/fm_grasp_tau01104_checkpoint_replay.2026-10-11_v1.json'
    array_path = 'configs/reproducibility/fm_grasp_tau01104_full_array_validation.2026-10-11_v1.json'
    evidence = ROOT / 'projects/flow_matching_research/results/2026-10-11/grasp_tau01104_audit_registration_v1'
    for path in (ROOT / replay_path, ROOT / array_path, evidence):
        if path.exists():
            raise FileExistsError('Preserve all prior registrations: ' + str(path))
    result = read(ROOT / result_path)
    queue_path = 'configs/experiments/fm_grasp_protocol_queue.v4.json'
    queue = read(ROOT / queue_path)
    job = next(j for j in queue['jobs'] if j['id'] == canonical)
    if (result['status'] != 'completed' or result['job'] != job
            or result['epochs_completed'] != 1500
            or result['optimizer_updates'] != job['expected_optimizer_updates']
            or result['validation_points'] != job['validation_points']
            or result['scored_points'] != job['test_points']
            or result['score_recipes'] != job['score_recipes']
            or result['diagnostic'] or result['paper_equivalence_certified']
            or result['strict_TAB_result']):
        raise ValueError('A completed canonical original-budget ablation is required')
    for item in result['artifacts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Completed original training artifact changed')
    templates = {
        'replay': 'configs/reproducibility/fm_grasp_lin1104_checkpoint_replay.2026-10-11_v1.json',
        'array': 'configs/reproducibility/fm_grasp_lin1104_full_array_validation.2026-10-11_v1.json',
    }
    configs = {name: deepcopy(read(ROOT / path)) for name, path in templates.items()}
    sources = [queue_path, result_path, job['model_config'], queue['environment_lock'],
               'scripts/flow_matching/audit_grasp_checkpoint_replay_v1.py',
               'scripts/flow_matching/audit_registered_grasp_results_v1.py',
               'scripts/flow_matching/audit_original_native_execution.py',
               'scripts/flow_matching/grasp_protocol_runner.py',
               'src/iia_benchmark/models/grasp_protocol_v2.py',
               'src/iia_benchmark/models/fm_native.py', queue['native_metrics'],
               'scripts/flow_matching/run_cfm_ts_low_memory_queue_v2.py',
               'scripts/flow_matching/audit_cfm_checkpoint_replay_v1.py',
               'scripts/flow_matching/heavy_resources.py',
               'scripts/flow_matching/heavy_scheduler_v2.py',
               'scripts/flow_matching/run_light_controller.py',
               'scripts/flow_matching/run_resource_gated_light_controller_v1.py',
               Path(__file__).relative_to(ROOT).as_posix(), *templates.values(),
               result['data_audit']['prepared_path']]
    sources += [a['path'] for a in result['data_audit']['raw_sources'] + result['artifacts']]
    receipts = [dict(path=p, sha256=sha(ROOT / p)) for p in dict.fromkeys(sources)]
    boundary = ('Original tau=0/SWAN/swan_sf/seed1104, complete1500epochs/93000updates; '
                'all3935validation and72000test points, original5source/10flow recipe. '
                'Readonly full saved-score audit and native full-model checkpoint re-inference; '
                'unchanged prior CPU threads, resource locks/guards, score/metric tolerances. '
                'Retain all480GRASP cohorts/9120entity jobs/10seeds and45paper scope. '
                'Test-label oracle Best-F1 and validation-only control remain separate. '
                'No additional training seed, author-equivalence or strictTAB claim.')
    scope = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
    for kind, config in configs.items():
        config['full_results'] = [dict(
            **({'canonical_id': canonical} if kind == 'replay' else {'id': canonical}),
            result_path=result_path, result_sha256=sha(ROOT / result_path))]
        suffix = 'full_checkpoint_replay' if kind == 'replay' else 'full_array_audit'
        config.update(output_directory=f'projects/flow_matching_research/results/2026-10-11/grasp_tau01104_{suffix}_v1',
                      state_root=f'experiments/runs/fm_grasp_tau01104_{suffix}_v1',
                      source_receipts=receipts, boundary=boundary, original_scope=scope)
        if kind == 'replay':
            config['artifact_root'] = 'F:/aicoding/IIA_Data/experiments/grasp_tau01104_full_checkpoint_replay_v1'
            verify_registration(config)
        prior = read(ROOT / templates[kind])
        if any(config[k] != prior[k] for k in ('python', 'settings', 'resource_lock')):
            raise ValueError('Original audit execution resource safeguards must remain unchanged')
        if kind == 'replay' and any(config[k] != prior[k] for k in (
                'score_tolerance', 'metric_tolerance', 'cpu_threads', 'full_job_count', 'full_score_recipe_count')):
            raise ValueError('Original full-model replay tolerances and execution budget must remain unchanged')
    exclusive_json(ROOT / replay_path, configs['replay'])
    exclusive_json(ROOT / array_path, configs['array'])
    exclusive_json(evidence / 'registration.json', dict(
        captured_utc=datetime.now(timezone.utc).isoformat(), canonical_id=canonical,
        complete_training_epochs=result['epochs_completed'], optimizer_updates=result['optimizer_updates'],
        parameter_count=result['parameter_count'], validation_points=result['validation_points'],
        test_points=result['scored_points'], score_recipes=result['score_recipes'],
        original_full_training_completed=True, full_model_reinference_completed=False,
        independent_full_array_audit_completed=False, additional_training_seed_slots=0,
        original_independent_training_seeds_completed_in_this_cohort=2,
        original_independent_training_seeds_required_in_this_cohort=10,
        replay_configuration=replay_path, array_configuration=array_path,
        configurations_sha256={p: sha(ROOT / p) for p in (replay_path, array_path)},
        source_receipts=receipts, original_scope=scope, published_numeric_table_unchanged=True,
        all_original_experiments_complete=False, author_equivalence_certified=False,
        strict_TAB_result=False, boundary=boundary))
    print(json.dumps(dict(full_training_completed=True, full_replay_registered=True,
                          full_array_audit_registered=True, canonical_id=canonical)))


if __name__ == '__main__':
    main()
