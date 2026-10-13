import numpy as np
import pytest
from scripts.flow_matching.audit_cfm_gp_five_seed_targets_v1 import SEEDS, identities, select_complete_pairs, grouped_summary, verify_config, ROOT, read


def test_all_five_seed_pairs_are_required_and_duplicates_cannot_count_twice():
    records = [{'original_canonical_id': i} for pair in identities() for i in pair]
    selected, missing = select_complete_pairs(records, identities())
    assert len(selected) == 5 and missing == []
    selected, missing = select_complete_pairs(records[:-1], identities())
    assert selected is None and missing == [identities()[-1][-1]]
    with pytest.raises(ValueError, match='Duplicate'): select_complete_pairs(records + records[:1], identities())


def test_constant_pair_difference_is_preserved_in_both_bootstrap_levels():
    literal = np.ones((5, 80))
    result = grouped_summary(literal, literal + 2, SEEDS, 7, 1000)
    assert result['paired_difference_mean'] == 2
    assert result['hierarchical_paired_bootstrap_ci95'] == [2, 2]
    assert result['paired_difference_joint_seed_std'] == 0
    assert not result['isolated_training_rng_uncertainty_estimated']


def test_seed_variation_does_not_disappear_when_each_seed_has_constant_trajectories():
    literal = np.ones((5, 80))
    conditional = literal + np.arange(5)[:, None]
    result = grouped_summary(literal, conditional, SEEDS, 19, 2000)
    low, high = result['hierarchical_paired_bootstrap_ci95']
    assert low < result['paired_difference_mean'] < high
    assert result['paired_difference_joint_seed_std'] > 0


def test_successful_subset_missing_groups_or_cell_flattening_cannot_pass():
    for literal, conditional, seeds in [(np.ones((4, 80)), np.ones((4, 80)), SEEDS[:4]),
                                         (np.ones((5, 40000)), np.ones((5, 40000)), SEEDS),
                                         (np.ones((5, 80)), np.ones((5, 80)), [1103] * 5)]:
        with pytest.raises(ValueError): grouped_summary(literal, conditional, seeds, 7, 1000)


def test_registered_full_original_budget_data_and_scope_are_unchanged():
    config = read(ROOT / 'configs/reproducibility/fm_cfm_GP400_all_five_original_target_pairs.2026-10-14_v1.json')
    runtime = verify_config(config)
    assert len(runtime['routes']) == 285
    assert config['paired_canonical_ids'] == identities()
