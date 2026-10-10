import json

import pytest

from scripts.flow_matching.register_strict_empty_output_recovery_v1 import build_case


def fixture(tmp_path):
    (tmp_path/'old').mkdir()
    job=dict(id='timesnet_smd_seed1106',output_directory='old',seed=1106,
        model_config='full_original_model',parameter_overrides=dict(device='cpu',seed=1106),
        frozen_files=[],window=100)
    original=dict(jobs=[job],settings=dict(epochs=20,batch_size=128),evaluation=dict(threshold='validation_p99'),python='python.exe')
    (tmp_path/'queue.json').write_text(json.dumps(original))
    return original,job


def test_only_attempt_identity_and_output_change_not_scientific_budget(tmp_path):
    original,job=fixture(tmp_path)
    case=build_case(tmp_path,'queue.json',original,job['id'],'new')
    assert {k:v for k,v in case['job'].items() if k not in ('id','output_directory')}=={
        k:v for k,v in job.items() if k not in ('id','output_directory')}
    assert case['settings']==original['settings'] and case['evaluation']==original['evaluation']
    assert case['original_job']==job and not list((tmp_path/'old').iterdir())


def test_actual_partial_checkpoint_is_preserved_and_not_misclassified_as_empty(tmp_path):
    original,job=fixture(tmp_path)
    (tmp_path/'old/checkpoint.pt').write_bytes(b'partial optimizer and RNG')
    with pytest.raises(ValueError,match='empty output'):
        build_case(tmp_path,'queue.json',original,job['id'],'new')
    assert (tmp_path/'old/checkpoint.pt').read_bytes()==b'partial optimizer and RNG'


def test_existing_recovery_directory_cannot_be_overwritten(tmp_path):
    original,job=fixture(tmp_path)
    (tmp_path/('new/jobs/'+job['id']+'__empty_output_recovery_v1')).mkdir(parents=True)
    with pytest.raises(FileExistsError):build_case(tmp_path,'queue.json',original,job['id'],'new')


def test_changed_original_frozen_source_is_rejected(tmp_path):
    original,job=fixture(tmp_path)
    (tmp_path/'model.py').write_text('changed_model()')
    job['frozen_files']=[dict(path='model.py',sha256='wrong')]
    with pytest.raises(ValueError,match='frozen full source'):
        build_case(tmp_path,'queue.json',original,job['id'],'new')
