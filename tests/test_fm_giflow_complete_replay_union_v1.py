import copy
import json
from pathlib import Path

import pytest

from scripts.flow_matching.audit_giflow_complete_replay_union_v1 import exact_scope

ROOT = Path(__file__).resolve().parents[1]


def original_evidence():
    config = json.loads((ROOT / 'configs/reproducibility/fm_giflow_complete_replay_union.2026-10-11_v1.json').read_text(encoding='utf-8'))
    queue = json.loads((ROOT / config['queue']).read_text(encoding='utf-8'))
    identities = {entry['job_id'] for entry in config['models']}
    jobs = [job for job in queue['jobs'] if job['id'] in identities]
    proofs = [json.loads((ROOT / entry['proof']).read_text(encoding='utf-8')) for entry in config['models']]
    return jobs, proofs


def test_all_three_original_completed_models_have_distinct_canonical_evidence():
    jobs, proofs = original_evidence()
    before = copy.deepcopy((jobs, proofs))
    assert set(exact_scope(jobs, proofs)) == {job['id'] for job in jobs}
    assert len(jobs) == 3 and (jobs, proofs) == before


def test_missing_model_or_repeated_replay_cannot_claim_full_coverage():
    jobs, proofs = original_evidence()
    with pytest.raises(ValueError, match='Every completed original'):
        exact_scope(jobs, proofs[:2])
    with pytest.raises(ValueError, match='cannot add a training seed'):
        exact_scope(jobs, proofs + [proofs[0]])


@pytest.mark.parametrize('field,value', [('seed', 999), ('original_Euler_steps', 19),
    ('full_test_windows', 1746), ('full_test_batches', 13), ('observed_forward_calls', 279),
    ('full_checkpoint_contents_and_reinference_passed', False), ('actual_tested_weights_restored_bitwise', False),
    ('additional_training_seed_slots', 1), ('author_equivalence_certified', True)])
def test_partial_or_unbound_model_replay_is_not_complete(field, value):
    jobs, proofs = original_evidence()
    proofs[2][field] = value
    with pytest.raises(ValueError, match='full original CPU model replay'):
        exact_scope(jobs, proofs)


def test_published_status_update_does_not_change_any_numerical_export():
    base = ROOT / 'projects/flow_matching_research/results/2026-10-11/all_algorithm_dataset_metrics_v23'
    target = ROOT / 'projects/flow_matching_research/results/2026-10-11/all_algorithm_dataset_metrics_v24'
    receipt = json.loads((base / 'publication_validation.json').read_text(encoding='utf-8'))
    for name in receipt['outputs']:
        if name not in ['README.rst', 'README.md']:
            assert (base / name).read_bytes() == (target / name).read_bytes()
    new = json.loads((target / 'publication_validation.json').read_text(encoding='utf-8'))
    assert new['giflow_full_native_models_replayed'] == 3
    assert new['giflow_released_air36_main_training_seeds'] == 2
    assert new['giflow_released_air36_main_required_training_seeds'] == 5
    assert new['newly_measured_metric_rows'] == new['additional_training_seed_slots'] == 0
    assert not new['all_original_experiments_complete']
