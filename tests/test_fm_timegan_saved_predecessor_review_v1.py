"""Actual native proof review rejects missing stages and forged full-array digests."""
import copy
from pathlib import Path

import pytest

from scripts.flow_matching import review_timegan_saved_predecessor_native_v1 as review


def config():
    return review.original.read(review.original.ROOT / 'configs/runtime/fm_timegan_saved_predecessor_native_audit.v1.json')


def test_actual_complete_saved_stage_native_proof_is_bound_to_every_original_array():
    result = review.review(config())
    assert result['passed'] and len(result['independently_bound_generator_stages']) == 3
    assert len(result['independently_bound_all10_complete_prediction_arrays']) == 10
    assert sum(r['complete_test_points'] for r in result['independently_bound_all10_complete_prediction_arrays']) == 14660
    assert result['experiment_complete'] is False


@pytest.mark.parametrize('mutation', ['bad_probability_digest','bad_generation_digest','missing_discriminator',
    'missing_generator_stage','unverified_rng','native_failure','claimed_complete','weakened_final_scope'])
def test_self_report_cannot_replace_actual_native_receipt_full_arrays_and_final_obligations(monkeypatch, mutation):
    value = config()
    root = review.original.ROOT
    proof_path = root / value['output_root'] / value['canonical_job']['id'] / 'saved_predecessor_native_replay.json'
    receipt_path = root / value['state_root'] / 'receipt.json'
    original_read = review.original.read
    proof = copy.deepcopy(original_read(proof_path))
    receipt = copy.deepcopy(original_read(receipt_path))
    if mutation == 'bad_probability_digest': proof['evaluators'][0]['full_prediction_array_sha256'][0] = '0'*64
    elif mutation == 'bad_generation_digest': proof['generation_array_sha256'] = '0'*64
    elif mutation == 'missing_discriminator': proof['evaluators'].pop()
    elif mutation == 'missing_generator_stage': proof['stage_checkpoint_restores'].pop()
    elif mutation == 'unverified_rng': proof['evaluators'][1]['full_post_evaluator_NumPy_state_bitwise_verified'] = False
    elif mutation == 'native_failure': receipt['exit_code'] = 1
    elif mutation == 'claimed_complete': proof['experiment_complete'] = True
    else: proof['final_full45_trained_evaluator_replays_still_required'] = 450

    def read(path):
        if Path(path).resolve() == proof_path.resolve(): return proof
        if Path(path).resolve() == receipt_path.resolve(): return receipt
        return original_read(path)

    monkeypatch.setattr(review.original, 'read', read)
    with pytest.raises(ValueError): review.review(value)
