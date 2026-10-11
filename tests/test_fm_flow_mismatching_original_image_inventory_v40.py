from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read
from scripts.flow_matching.publish_flow_mismatching_original_image_inventory_v40 import missing_rows


def test_all_original520_targets_are_missing_without_synthetic_or_zero_performance():
    queue = read(ROOT / 'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json')
    rows = missing_rows(queue, 'frozen-capture', 'frozen-config')
    assert len(rows) == 520 and all(row['n'] == 0 and row['required_repeats'] == 5 for row in rows)
    assert all(row['mean'] == row['std'] == row['ci95_low'] == row['ci95_high'] == '' for row in rows)
    assert all(row['author_equivalence_certified'] is False and row['strict_TAB_result'] is False for row in rows)
    assert sum(row['comparison'] == 'original_comparator_training_or_native_adapter_settings_pending' for row in rows) == 35
    assert sum(row['comparison'] == 'original_A10080GB_efficiency_pending' for row in rows) == 30
    assert all(row['epochs'] == 50 for row in rows if 'tccm' in row['algorithm'].lower()) is True
    # TCCM identity is in the frozen target method, not inferred from a table-row id.
    tccm_id = next(t['id'] for t in queue['paper_targets'] if t['method'] == 'TCCM')
    assert all(row['epochs'] == 50 for row in rows if row['algorithm'].endswith(tccm_id))
    for target in queue['paper_targets']:
        actual = {r['metric']: r['paper_mean'] for r in rows if r['algorithm'].endswith('/' + target['id'])}
        assert actual == target['paper_reported']
