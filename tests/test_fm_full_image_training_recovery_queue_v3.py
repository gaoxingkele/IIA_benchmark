from copy import deepcopy
import pytest
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read
from scripts.flow_matching.run_full_image_training_recovery_queue_v3 import verify_registration


def registration():
    return read(ROOT/'configs/reproducibility/fm_full_image_training_execution_recovery.2026-10-11_v3.json')


def test_every_original_method_dataset_conditioning_and_seed_slot_is_registered():
    records=verify_registration(registration())
    assert len(records)==25
    assert sum(v=='FM' for v in records.values())==20
    assert sum(v=='WT' for v in records.values())==5


def test_no_missing_original_task_or_duplicate_slot_can_pass():
    config=deepcopy(registration())
    config['ordered_original_jobs']=config['ordered_original_jobs'][:-1]
    with pytest.raises(ValueError,match='entire original'):verify_registration(config)
    config=deepcopy(registration())
    config['ordered_original_jobs'][-1]=config['ordered_original_jobs'][0]
    with pytest.raises(ValueError,match='entire original'):verify_registration(config)


@pytest.mark.parametrize('key', ['minimum_gpu_free_bytes','minimum_commit_headroom_bytes','emergency_gpu_free_bytes','emergency_commit_headroom_bytes'])
def test_original_resource_floors_cannot_be_weakened(key):
    config=deepcopy(registration())
    config['settings'][key]-=1
    with pytest.raises(ValueError,match='original shared'):verify_registration(config)


def test_full_checkpoint_disk_budget_cannot_be_replaced_by_a_smoke_budget():
    config=deepcopy(registration())
    config['full_cohort_required_free_disk_bytes']=1024
    with pytest.raises(ValueError,match='conservative disk'):verify_registration(config)
