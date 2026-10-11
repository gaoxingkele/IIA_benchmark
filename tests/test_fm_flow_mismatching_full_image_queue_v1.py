import copy
from contextlib import contextmanager
import pytest

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read
from scripts.flow_matching.run_flow_mismatching_full_image_evaluation_queue_v1 import full_gpu_slot, validate_queue


def full_queue():
    return read(ROOT / 'configs/experiments/fm_flow_mismatching_full_image_evaluation_queue.v1.json')


def test_all65_original_table_cases_and_five_full_seeds_are_retained():
    queue = full_queue()
    assert validate_queue(queue)
    assert len(queue['cases']) == 325
    assert sum(len(x['paper_reported']) for x in queue['paper_targets']) == 520
    assert sum(len(x['paper_reported']) for x in queue['remaining_original_comparator_targets']) == 35
    assert len({x['checkpoint_epoch'] for x in queue['cases']}) == 20
    for remove_or_duplicate in ('remove', 'duplicate'):
        broken = copy.deepcopy(queue)
        if remove_or_duplicate == 'remove': broken['cases'].pop()
        else: broken['cases'][-1] = broken['cases'][0]
        with pytest.raises(ValueError): validate_queue(broken)


@pytest.mark.parametrize('field,value', [('times', 8), ('paths', 8), ('aggregation', 'average'),
                                        ('weighted', False), ('top_fraction', .1)])
def test_original_paper_score_parameters_cannot_be_replaced_with_development_defaults(field, value):
    broken = full_queue()
    broken['cases'][0]['score_parameters'][field] = value
    with pytest.raises(ValueError): validate_queue(broken)


def test_original_printed_target_and_full_epoch_source_cannot_be_changed():
    for field in ('paper', 'epoch', 'checkpoint', 'conditioning'):
        broken = full_queue()
        case = broken['cases'][0]
        if field == 'paper': case['paper_reported']['image_AUROC'] += .1
        if field == 'epoch': case['checkpoint_epoch'] = 50
        if field == 'checkpoint': case['checkpoint'] = 'development_smoke.pt'
        if field == 'conditioning': case['variant'] = 'pooled'
        with pytest.raises(ValueError): validate_queue(broken)


def test_insufficient_gpu_wait_releases_shared_slot_before_sleep(monkeypatch):
    owned = dict(value=False, entries=0, sleep=0)
    @contextmanager
    def slot(*args):
        owned.update(value=True, entries=owned['entries'] + 1)
        try: yield
        finally: owned['value'] = False
    def snapshot(settings):
        assert owned['value']
        return dict(gpu_free_bytes=0 if owned['entries'] == 1 else 100)
    def sleep(seconds):
        assert not owned['value'] and seconds == 10
        owned['sleep'] += 1
    monkeypatch.setattr('scripts.flow_matching.run_flow_mismatching_full_image_evaluation_queue_v1.time.sleep', sleep)
    runtime = dict(settings=dict(minimum_gpu_free_bytes=100), resource_lock='fixture_virtual.lock')
    with full_gpu_slot(dict(resource_slot=slot, gpu_snapshot=snapshot), runtime, lambda _: None):
        assert owned['value']
    assert not owned['value'] and owned['entries'] == 2 and owned['sleep'] == 1
