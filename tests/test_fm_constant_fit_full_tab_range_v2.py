import copy
import json

from scripts.flow_matching.register_constant_fit_full_tab_range_v2 import derive_science, REGISTRY
from scripts.flow_matching.run_light_controller import ROOT


def test_every_original_metric_math_threshold_buffer_and_guard_field_unchanged():
    old = json.loads((ROOT / 'configs/experiments/fm_tab_range_execution.sparse.v2.json').read_text('utf-8'))
    registry = json.loads((ROOT / REGISTRY).read_text('utf-8'))
    new = derive_science(old, registry)
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {'id', 'output_root', 'queue_paths', 'boundary'}
    assert new['cpu_threads'] == old['cpu_threads'] == 1
    assert new['reference_metric_root'] == old['reference_metric_root']
    assert new['metric_semantics'] == old['metric_semantics']
    assert new['strict_calibration'] == old['strict_calibration']
    assert new['runtime_resources'] == old['runtime_resources']
    assert new['engine'] == old['engine']


def test_all291_corrected_full_method_ablation_seed_results_are_followed():
    old = json.loads((ROOT / 'configs/experiments/fm_tab_range_execution.sparse.v2.json').read_text('utf-8'))
    registry = json.loads((ROOT / REGISTRY).read_text('utf-8'))
    new = derive_science(old, registry)
    jobs = [j for path in new['queue_paths'] for j in json.loads((ROOT / path).read_text('utf-8'))['jobs']]
    assert len(new['queue_paths']) == 7
    assert len(jobs) == len({j['id'] for j in jobs}) == 291
    assert {j['dataset'] for j in jobs} == {'smap'}
    assert all(j['id'].endswith('__constant_fit_v2') for j in jobs)
    assert all(j['dataset_manifest_path'] == registry['corrected_manifest'] for j in jobs)
