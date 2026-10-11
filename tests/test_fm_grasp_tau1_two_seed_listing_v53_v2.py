from copy import deepcopy

import pytest

from scripts.flow_matching import publish_grasp_tau1_two_seed_listing_v53_v2 as publication
from scripts.flow_matching import run_grasp_tau1_two_seed_publication_v53_v2 as guarded


def pair_rows():
    before = [dict(algorithm='unrelated', captured_utc='original', mean='1')]
    for protocol, _, names in publication.PROTOCOLS:
        for name in names:
            before.append(dict(algorithm='GRASP/tau_1', dataset='swan', protocol=protocol,
                metric=name, n='1', required_repeats='10', mean='.5', std='', se='',
                ci95_low='', ci95_high='', captured_utc='original', source='original',
                author_equivalence_certified='false', strict_TAB_result=''))
    after = deepcopy(before)
    for row in after[1:]:
        row.update(n='2', mean='.6', std='.01', se='.007', ci95_low='.51',
            ci95_high='.69', captured_utc='verified', source='native-proof')
    return before, after


def test_only_six_original_tau1_slots_advance_and_both_protocols_remain_separate():
    before, after = pair_rows()
    changed = publication.check_delta(before, after)
    assert len(changed) == 6
    publication.validate_publication_statistics(after, changed)
    after[0]['captured_utc'] = 'recaptured'
    with pytest.raises(ValueError):
        publication.check_delta(before, after)


@pytest.mark.parametrize('key,value', [('required_repeats', '2'), ('dataset', 'other'),
    ('protocol', 'merged_PA'), ('n', '10'), ('algorithm', 'GRASP/tau_0p5'),
    ('author_equivalence_certified', 'true'), ('strict_TAB_result', 'true')])
def test_cannot_change_original_scope_or_claim_paper_or_tab_alignment(key, value):
    before, after = pair_rows()
    after[1][key] = value
    with pytest.raises(ValueError):
        publication.check_delta(before, after)


@pytest.mark.parametrize('key,value', [('mean', 'nan'), ('std', '-.01'), ('se', '-.01'),
    ('ci95_low', '.7'), ('ci95_high', '.5'), ('mean', 'inf')])
def test_invalid_uncertainty_cannot_enter_publication(key, value):
    before, after = pair_rows()
    after[1][key] = value
    with pytest.raises(ValueError):
        publication.validate_publication_statistics(after, publication.check_delta(before, after))


def test_missing_metric_and_added_slots_are_rejected():
    before, after = pair_rows()
    after[-1] = deepcopy(before[-1])
    with pytest.raises(ValueError, match='Exactly all six'):
        publication.check_delta(before, after)
    with pytest.raises(ValueError, match='Every existing metric'):
        publication.check_delta(before, after[:-1])


@pytest.mark.parametrize('field,value', [('passed', False),
    ('all_original_forward_counts_and_recipe_budgets_verified', False),
    ('actual_completed_recipe_count', 0), ('configuration_sha256', 'changed')])
def test_unverified_or_truncated_native_recipe_proof_rejected(monkeypatch, field, value):
    canonical = 'grasp_protocol_v3__tau_1__swan__swan_sf__seed1104'
    result = [dict(canonical_id=canonical)]
    tolerance = dict(rtol=.0002, atol=.001)
    metric = dict(metric_atol=.0002)
    config = dict(completed_new_seed_proofs={'tau_1': 'proof.json'},
        completed_review_configuration='review.json', score_tolerance=tolerance,
        metric_tolerance=metric)
    proof = dict(passed=True, configuration='review.json', configuration_sha256='pin',
        completed_native_replay_configuration='producer.json', actual_completed_recipe_count=1,
        all_original_forward_counts_and_recipe_budgets_verified=True)
    proof[field] = value
    review = dict(completed_replay_configuration='producer.json', pending_full_recipe_count=0,
        full_results=result)
    producer = dict(full_results=result, score_tolerance=tolerance, metric_tolerance=metric)
    values = {'proof.json': proof, 'review.json': review, 'producer.json': producer}
    monkeypatch.setattr(publication, 'read', lambda p: values[p.name])
    monkeypatch.setattr(publication, 'sha', lambda p: 'pin')
    with pytest.raises(ValueError, match='exact completed independent'):
        publication.verify_new_seed_terminal(config)


@pytest.mark.parametrize('failure', [None, 'other_capture', 'gaussian_ci', 'seed_scope', 'claimed_TAB'])
def test_independent_full_table_review_rejects_collateral_changes_and_false_uncertainty(tmp_path, monkeypatch, failure):
    """Small fabricated values exercise publication gates, never benchmark performance."""
    import csv
    import json
    import math

    before, after = pair_rows()
    before = before[1:]
    after = after[1:]
    for index, row in enumerate(after):
        std = .2 / math.sqrt(2)
        half = math.tan(math.pi * (.975 - .5)) * .1
        row.update(mean='.6', std=str(std), se='.1', ci95_low=str(.6-half),
            ci95_high=str(.6+half), source='summary.json')
    headers = list(before[0]) + [f'preserved_{i}' for i in range(31-len(before[0]))]
    filler = {key: '' for key in headers}
    filler.update(algorithm='unrelated', mean='1', captured_utc='original')
    empty = dict(filler, mean='')
    before += [dict(filler) for _ in range(2125)] + [dict(empty) for _ in range(6495)]
    after += [dict(filler) for _ in range(2125)] + [dict(empty) for _ in range(6495)]
    if failure == 'other_capture':
        after[7]['captured_utc'] = 'recaptured'
    elif failure == 'gaussian_ci':
        after[0]['ci95_low'] = str(.6-1.96*.1)
    elif failure == 'seed_scope':
        after[0]['required_repeats'] = '2'
    elif failure == 'claimed_TAB':
        after[0]['strict_TAB_result'] = 'true'
    for folder, values in [('before', before), ('after', after)]:
        target = tmp_path/folder
        target.mkdir()
        with (target/'all_algorithm_dataset_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(values)
    for seed, value in [(1103, .5), (1104, .7)]:
        metrics = {field: {name: value for name in names} for _, field, names in publication.PROTOCOLS}
        (tmp_path/f'{seed}.json').write_text(json.dumps(dict(job=dict(seed=seed), metrics=[dict(metrics=metrics)])), encoding='utf-8')
    config = dict(base_publication='before', output_directory='after',
        original_seed_results=['1103.json', '1104.json'], cohort_summary_path='summary.json', original_scope={'papers': 45})
    monkeypatch.setattr(guarded, 'ROOT', tmp_path)
    if failure:
        with pytest.raises((ValueError, KeyError)):
            guarded.independent_publication_review(config)
        assert not (tmp_path/'after/independent_publication_review.json').exists()
    else:
        guarded.independent_publication_review(config)
        proof = json.loads((tmp_path/'after/independent_publication_review.json').read_text(encoding='utf-8'))
        assert proof['passed'] and proof['changed_rows'] == 6
        assert proof['required_original_seeds'] == 10 and not proof['all_original_experiments_complete']
