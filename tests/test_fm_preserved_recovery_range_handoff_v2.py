import json

from scripts.flow_matching.run_readonly_tab_range_queue_v1 import ROOT, bind
from scripts.flow_matching.publish_ready_strict_metrics_v33 import verify_recovery_binding


def test_all_six_original_scientific_budgets_and_native_evaluator_preserved():
    runtime = json.loads((ROOT/'configs/runtime/fm_recovery_tab_range_readonly.v2.json').read_text(encoding='utf-8'))
    _, config, limits, binding = bind(ROOT, runtime, 'range_main')
    old = json.loads((ROOT/'configs/experiments/fm_recovery_tab_ranges.v1.json').read_text(encoding='utf-8'))
    assert binding['source_path'] == 'scripts/flow_matching/run_tab_ranges_sparse_queue.py'
    for field in ('metric_semantics', 'tab_commit', 'strict_calibration', 'tab_calibration',
        'undefined_policy', 'engine', 'runtime_resources', 'cpu_threads'):
        assert config[field] == old[field]
    assert config['parent_config'] == 'configs/experiments/fm_recovery_tab_ranges.v1.json'
    assert config['cpu_threads'] == 1
    assert runtime['resource_lock'] == 'experiments/runs/fm_readonly_tab_range_eval_cpu.lock'
    assert limits['minimum_commit_headroom_bytes'] >= config['runtime_resources']['minimum_commit_headroom_bytes']
    mirror_path = config['queue_paths'][0]
    jobs = json.loads((ROOT/mirror_path).read_text(encoding='utf-8'))['jobs']
    assert len(jobs) == 6
    registration = json.loads((ROOT/'projects/flow_matching_research/results/2026-10-11/preserved_recovery_range_handoff_v2/registration.json').read_text(encoding='utf-8'))
    cohorts = [dict(runs=[dict(id=j['id'],seed=j['seed'],result=j['output_directory']+'/result.json')]) for j in jobs]
    for recovery in registration['recovery_bindings']:
        proof = verify_recovery_binding(recovery, cohorts)
        assert proof['new_training_seed_slots'] == 0
    corrected = next(j for j in jobs if '__cicids__' in j['id'])
    assert corrected['total_training_epochs'] == 60
    assert corrected['parameter_overrides']['device'] == 'cpu'
    assert corrected['seed'] == 1103
