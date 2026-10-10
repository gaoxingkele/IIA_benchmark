import copy

import pytest

from scripts.flow_matching.audit_cfm_replay_union_v1 import exact_replay_union


def evidence():
    jobs = [dict(id=name, status='completed', result_path=name+'/result.json',
                 result_sha256=name+'-digest') for name in ('old', 'increment')]
    def proof(job):
        return dict(full_checkpoint_contents_and_re_inference_passed=True,
                    full_job_count=1, cuda_context_initialized=False,
                    additional_training_seed_slots=0,
                    records=[dict(original_canonical_id=job['id'],
                                  result_path=job['result_path'], result_sha256=job['result_sha256'],
                                  checkpoint_contents_and_full_budget_verified=True,
                                  original_full_grid_replayed=True)])
    return dict(jobs=jobs), [proof(j) for j in jobs]


def test_preserved_proof_and_increment_cover_exact_completed_scope():
    snapshot, proofs = evidence()
    before = copy.deepcopy((snapshot, proofs))
    expected, union = exact_replay_union(snapshot, proofs)
    assert set(expected) == set(union) == {'old', 'increment'}
    assert (snapshot, proofs) == before


def test_repeated_recovery_or_metric_proof_cannot_add_training_seed():
    snapshot, proofs = evidence()
    with pytest.raises(ValueError, match='Duplicate canonical'):
        exact_replay_union(snapshot, proofs + [proofs[0]])


def test_partial_increment_is_not_full_current_coverage():
    snapshot, proofs = evidence()
    with pytest.raises(ValueError, match='Every currently completed'):
        exact_replay_union(snapshot, proofs[:1])


def test_old_proof_must_match_current_checkpoint_attempt():
    snapshot, proofs = evidence()
    proofs[1]['records'][0]['result_sha256'] = 'different-attempt'
    with pytest.raises(ValueError, match='current full canonical result'):
        exact_replay_union(snapshot, proofs)


@pytest.mark.parametrize('field,value', [('full_checkpoint_contents_and_re_inference_passed', False),
                                      ('full_job_count', 2), ('cuda_context_initialized', True),
                                      ('additional_training_seed_slots', 1)])
def test_incomplete_or_retrained_proof_is_not_readonly_evidence(field, value):
    snapshot, proofs = evidence()
    proofs[1][field] = value
    with pytest.raises(ValueError, match='complete CPU checkpoint proof'):
        exact_replay_union(snapshot, proofs)
