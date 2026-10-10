from pathlib import Path

import pytest

from scripts.flow_matching import audit_grasp_checkpoint_replay_resume_v2 as resume


def inventory(monkeypatch, cached, pending=1, total=2):
    result = {'metrics': [{'recipe': {'source_samples': 5, 'flow_evaluations': 10}},
                          {'recipe': {'source_samples': 5, 'flow_evaluations': 50}}]}
    monkeypatch.setattr(resume.base, 'read', lambda path: result)
    monkeypatch.setattr(resume.base, 'sha', lambda path: 'frozen')
    return {'full_results': [{'canonical_id': 'full-model', 'result_path': 'result.json'}],
            'full_score_recipe_count': total, 'reusable_complete_recipes': cached,
            'pending_full_recipe_count': pending}


def cached(recipe=None):
    return {'canonical_id': 'full-model', 'recipe': recipe or {'source_samples': 5, 'flow_evaluations': 10},
            'proof_path': 'proof.json', 'proof_sha256': 'frozen'}


def test_complete_inventory_keeps_all_pending_original_recipes(monkeypatch):
    assert set(resume.validate_complete_inventory(inventory(monkeypatch, [cached()]))) == {'full-model'}
    with pytest.raises(ValueError, match='remaining original full recipes'):
        resume.validate_complete_inventory(inventory(monkeypatch, [cached()], pending=0))


def test_prior_partial_receipts_cannot_drop_duplicate_or_original_recipes(monkeypatch):
    with pytest.raises(ValueError, match='duplicate or unregistered'):
        resume.validate_complete_inventory(inventory(monkeypatch, [cached(), cached()], pending=0))
    with pytest.raises(ValueError, match='duplicate or unregistered'):
        resume.validate_complete_inventory(inventory(monkeypatch, [cached({'source_samples': 1, 'flow_evaluations': 1})]))
    with pytest.raises(ValueError, match='every original full score recipe'):
        resume.validate_complete_inventory(inventory(monkeypatch, [cached()], total=1))


def test_prior_full_proof_bytes_are_frozen(monkeypatch):
    config = inventory(monkeypatch, [cached()])
    monkeypatch.setattr(resume.base, 'sha', lambda path: 'changed')
    with pytest.raises(ValueError, match='proof changed'):
        resume.validate_complete_inventory(config)
