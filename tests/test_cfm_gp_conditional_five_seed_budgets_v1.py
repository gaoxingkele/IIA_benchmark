from copy import deepcopy
import pytest
from scripts.flow_matching import audit_cfm_gp_conditional_five_seed_budgets_v1 as audit


def test_every_full_five_seed_pair_required_without_duplicate_repetitions():
    records=[dict(original_canonical_id=i) for pair in audit.identities() for i in pair]
    selected,missing=audit.select_complete_pairs(records,audit.identities())
    assert len(selected)==5 and not missing
    selected,missing=audit.select_complete_pairs(records[:-1],audit.identities())
    assert selected is None and len(missing)==1
    with pytest.raises(ValueError):audit.select_complete_pairs(records+records[:1],audit.identities())


def test_loss_prefix_difference_remains_visible_without_inventing_weight_equivalence():
    report=audit.prefix_evidence([1.]*400,[1.]*600)
    assert report['first400_epoch_losses_bitwise_identical']
    assert not report['matching_full_optimizer_state_or_intermediate_weights_claimed']
    changed=[1.]*600;changed[71]=2.
    report=audit.prefix_evidence([1.]*400,changed)
    assert not report['first400_epoch_losses_bitwise_identical']
    assert report['first_differing_epoch_1based']==72 and report['maximum_first400_epoch_loss_difference']==1.
    with pytest.raises(ValueError):audit.prefix_evidence([1.]*399,[1.]*600)


def test_frozen_actual_pairs_reject_changed_solver_data_target_or_grid():
    config=audit.read(audit.ROOT/'configs/reproducibility/fm_cfm_GP_conditional_all_five_budget_pairs.2026-10-14_v1.json')
    runtime=audit.verify_config(config)
    jobs=[next(r['job'] for r in runtime['routes'] if r['original_canonical_id']==i) for i in audit.identities()[0]]
    parameters=[audit.read(audit.ROOT/j['model_config'])['parameters'] for j in jobs]
    for key,value in [('rtol',.1),('target_mode','paper_literal')]:
        altered=deepcopy(parameters);altered[1][key]=value
        with pytest.raises(ValueError):audit.validate_pair(jobs,altered,1103)
    for key,value in [('data_sha256','0'*64),('test_points_per_trajectory',499),('expected_optimizer_updates',374999)]:
        altered=deepcopy(jobs);altered[1][key]=value
        with pytest.raises(ValueError):audit.validate_pair(altered,parameters,1103)
