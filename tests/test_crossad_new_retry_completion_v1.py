"""Same-seed retry routing and preservation of complete139-case science."""
import copy

import pytest

from scripts.flow_matching import run_crossad_new_retry_completion_v1 as review

CONFIG = 'configs/runtime/fm_crossad_new_seed1103_retry_completion.v1.json'


def bindings():
    config = review.read(CONFIG)
    base = review.read(config['original_completion_configuration'])
    original = review.read(base['original_queue'])
    canonical = {j['id']: j for j in original['jobs']}
    return config, base, original, canonical


def test_three_same_seed_routes_preserve_every_original_scientific_field():
    config, base, original, canonical = bindings()
    assert len(config['retry_routes']) == 3
    assert {r['canonical_id'] for r in config['retry_routes']} == review.EXPECTED
    for route in config['retry_routes']:
        queue = review.read(route['native_queue'])
        job = next(j for j in queue['jobs'] if j['id'] == route['canonical_id'])
        review.exact_science(canonical[job['id']], job, original, queue)
        assert len(review.original.validate_scope(queue, config)) == 139
        assert job['seed'] == 1103
        assert job['train_parameters']['train_epochs'] == 20


@pytest.mark.parametrize('corruption', ['batch', 'other_seed', 'other_output', 'remove_slot'])
def test_no_retry_can_change_the_science_or_other138_slots(corruption):
    config, _, original, canonical = bindings(); route = config['retry_routes'][0]
    queue = copy.deepcopy(review.read(route['native_queue']))
    job = next(j for j in queue['jobs'] if j['id'] == route['canonical_id'])
    if corruption == 'batch': job['train_parameters']['batch_size'] = 64
    if corruption == 'other_seed': queue['jobs'][-1]['seed'] += 1
    if corruption == 'other_output': queue['jobs'][-1]['output_directory'] = 'different/output'
    if corruption == 'remove_slot': queue['jobs'].pop()
    with pytest.raises(ValueError, match='All139 scientific'):
        review.exact_science(canonical[route['canonical_id']], job, original, queue)


def test_all_original_replay_resource_and_numeric_gates_retained():
    config, base, _, _ = bindings()
    for route in config['retry_routes']:
        effective = review.effective_stage_settings(config, base, route)
        for key in review.original.GUARDS:
            assert effective[key] == base[key] == config[key]
    assert config['score_tolerance'] == review.original.SCORE_TOLERANCE
    assert config['metric_tolerance'] == review.original.METRIC_TOLERANCE
    assert config['required_original_jobs'] == 139


def test_unexpected_or_missing_registered_route_cannot_pass_full_verification():
    config = copy.deepcopy(review.read(CONFIG)); config['retry_routes'].pop()
    # Full original baseline checks run before rejecting the incomplete roster.
    with pytest.raises(ValueError, match='three actual retries'):
        review.verify(config, check_live=False)
