import json

from scripts.flow_matching.run_readonly_tab_range_queue_v1 import ROOT, bind, OLD_LOCK


def test_actual_recovery_configuration_retains_all_original_jobs_and_evaluator():
    runtime = json.loads((ROOT/'configs/runtime/fm_recovery_tab_range_readonly.v1.json').read_text(encoding='utf-8'))
    _, config, limits, binding = bind(ROOT, runtime, 'range_main')
    assert binding['source_path'] == 'scripts/flow_matching/run_tab_ranges_sparse_queue.py'
    assert binding['queue_path'] == 'configs/experiments/fm_recovery_tab_ranges.v1.json'
    assert config == json.loads((ROOT/binding['queue_path']).read_text(encoding='utf-8'))
    jobs = [j for p in config['queue_paths'] for j in json.loads((ROOT/p).read_text(encoding='utf-8'))['jobs']]
    assert len(jobs) == 6
    assert runtime['resource_lock'] != OLD_LOCK
    assert runtime['resource_lock'] == 'experiments/runs/fm_readonly_tab_range_eval_cpu.lock'
    assert limits['minimum_commit_headroom_bytes'] >= config['runtime_resources']['minimum_commit_headroom_bytes']
    assert runtime['cpu_threads'] == config['cpu_threads'] == 1
