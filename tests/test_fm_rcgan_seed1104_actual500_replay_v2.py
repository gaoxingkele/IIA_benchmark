"""Actual completion metadata and strict full-scope continuation regression."""
import copy
import ast
import hashlib
import json

import pytest

from scripts.flow_matching import run_rcgan_full500_completion_replay_queue_v2 as replay


def configuration():
    return replay.original.read(replay.ROOT / 'configs/runtime/fm_rcgan_full500_completion_replay_queue.v2.json')


def test_actual500_completion_and_preserved_scope():
    config = configuration()
    _, routes = replay.verify(config)
    route = routes[1]
    result = replay.original.read(route['result_path'])
    receipt = replay.original.read(replay.ROOT / route['native_receipt'])
    replay.original.complete_metadata(route, result, receipt)
    assert result['verified_prior_full_epochs'] == 366
    assert result['resume_segment_full_epochs'] == 134
    assert len(result['checkpoints']) == len(result['evaluators']) == 500
    assert result['optimizer_updates'] == dict(D=642000, G=642000, sigma=663665)
    assert [r['canonical_job']['seed'] for r in routes] == list(range(1103, 1108))


@pytest.mark.parametrize('change', ['route', 'other_seed', 'guard', 'tolerance', 'epochs'])
def test_changed_mapping_scope_or_numerics_rejected(change):
    config = copy.deepcopy(configuration())
    if change == 'route':
        config['routes'][1]['execution_job']['seed'] = 42
    elif change == 'other_seed':
        config['routes'][4]['result_path'] = 'wrong.json'
    elif change == 'guard':
        config['settings']['minimum_commit_headroom_bytes'] = 1
    elif change == 'tolerance':
        config['equivalence_gate'] = 'allclose'
    else:
        config['required_epochs_per_job'] = 1
    with pytest.raises(ValueError):
        replay.verify(config)


def test_native_verifier_and_worker_dispatch_mapping_only():
    bound = replay.bind()
    assert bound['verify'] is replay.verify
    assert 'scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v2' in bound['main'].__code__.co_consts
    assert 'scripts.flow_matching.rcgan_native_training_resume_runtime_v3' in bound['native_result'].__code__.co_names
    source = replay.ROOT / 'scripts/flow_matching/run_rcgan_full500_completion_replay_queue_v1.py'
    old = ast.parse(source.read_text(encoding='utf-8'))
    before = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    after = next(n for n in replay.mapped_tree().body if n.name == 'worker')
    assert ast.dump(before) == ast.dump(after)


def test_completion_archive_is_exact_and_does_not_claim_replay():
    base = replay.ROOT / 'projects/flow_matching_research/results/2026-10-13/rcgan_seed1104_actual500_completion_v1'
    manifest = json.loads((base / 'completion_evidence.json').read_text(encoding='utf-8'))
    for item in manifest['artifacts']:
        archived = (base / item['archive_name']).read_bytes()
        assert archived == replay.ROOT.joinpath(item['source_path']).read_bytes()
        assert hashlib.sha256(archived).hexdigest() == item['sha256']
    assert manifest['full_native_training_budget_completed'] is True
    assert manifest['independent_full500_native_replay_completed'] is False
    assert manifest['strict_TAB_result'] is False
    assert manifest['all_original_experiments_complete'] is False
