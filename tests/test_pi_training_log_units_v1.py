from pathlib import Path
import pytest
from scripts.flow_matching.audit_pi_training_log_units_v1 import ROOT, verify_counter_source, timing_summary


def test_pinned_original_source_has_constant_counter_and_logs_before_update():
    source = (ROOT / 'experiments/runs/fm_pi_transformer_author_v1/runtime/causal_axis_repaired/main.py').read_text(encoding='utf-8')
    proof = verify_counter_source(source)
    assert proof['counter_never_incremented'] and proof['logging_before_current_optimizer_step']


def test_incremented_counter_cannot_receive_the100fold_correction():
    source = (ROOT / 'experiments/runs/fm_pi_transformer_author_v1/runtime/causal_axis_repaired/main.py').read_text(encoding='utf-8')
    source = source.replace('self.optimiser.zero_grad()', 'iter_count += 1\n                self.optimiser.zero_grad()', 1)
    with pytest.raises(ValueError, match='constant-one'): verify_counter_source(source)


def test_only_interior_intervals_are_corrected_and_current_update_is_excluded():
    text = '\n'.join('speed: 600.0000s/iter; left time: 60000.0000s' for _ in range(7))
    result = timing_summary(text, 'Epoch: 1, Steps: 518 | Train Loss: 1', 518, 3)
    assert result['required_optimizer_updates'] == 1554
    assert result['completed_optimizer_updates_lower_bound'] == 717
    assert result['interval_records'][0]['corrected_seconds_per_update'] is None
    assert result['interval_records'][5]['corrected_seconds_per_update'] is None
    assert result['latest_valid_interior_interval']['corrected_seconds_per_update'] == 6
    assert not result['full_training_completion_certified']
    assert not result['validation_and_scoring_time_estimated']


def test_partial_epoch_tail_and_epoch_boundary_are_excluded():
    result = timing_summary('\n'.join('speed: 100.0000s/iter; left time: 50000.0000s' for _ in range(6)),
                            'Epoch: 1, Steps: 518 | Train Loss: 1', 518, 3)
    assert result['last_observed_logging_boundary']['epoch_one_based'] == 2
    assert result['last_observed_logging_boundary']['corrected_remaining_optimizer_only_seconds'] is None
    assert result['completed_optimizer_updates_lower_bound'] == 617


def test_extra_boundaries_or_duplicate_epoch_receipts_do_not_pass_budget():
    text = '\n'.join('speed: 100.0000s/iter; left time: 50000.0000s' for _ in range(16))
    with pytest.raises(ValueError, match='extra logged'): timing_summary(text, '', 518, 3)
    with pytest.raises(ValueError, match='ordered original'):
        timing_summary('', 'Epoch: 1, Steps: 518 |\nEpoch: 1, Steps: 518 |', 518, 3)
