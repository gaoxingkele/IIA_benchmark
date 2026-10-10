import json
import random

import h5py
import numpy as np
import pytest
import torch

from scripts.flow_matching import validate_saits_full_result_v1 as validator


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, 'ROOT', tmp_path)
    data = tmp_path/'data'; data.mkdir()
    with h5py.File(data/'datasets.h5', 'w') as stream:
        stream.create_dataset('test/indicating_mask', data=np.ones((3, 2, 1)))
    np.savez(data/'audit_arrays.npz', test_ids=np.arange(3))
    sha=validator.digest
    (data/'data_audit.json').write_text(json.dumps(dict(dataset_sha256=sha(data/'datasets.h5'),
        audit_arrays_sha256=sha(data/'audit_arrays.npz'))))
    author=tmp_path/'author.ini'; author.write_text('[training]\nepochs=10\n')
    config=dict(id='unit_integrity_only', diagnostic=False, dataset_root='data', output_root='output',
        author_config='author.ini',author_config_sha256=sha(author),dataset_sha256=sha(data/'datasets.h5'))
    path=tmp_path/'config.json'; path.write_text(json.dumps(config))
    output=tmp_path/'output'; output.mkdir()
    state=dict(config_sha256=sha(path), dataset_sha256=config['dataset_sha256'], finished=True,
        next_epoch=10, patience=30,optimizer={'state':{1:{'step':torch.tensor(10.)}}},
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=[])
    torch.save(state,output/'resume.pt')
    best=dict(model_state_dict={'weight':torch.ones(1)}, validation_mae=1.,training_step=10,epoch=10)
    torch.save(best,output/'best.pt')
    np.savez(output/'patient_statistics.npz',unit_ids=np.arange(3),group_ids=np.arange(3),
        statistics=np.array([[2.,2.,2.,2.]]*3))
    result=dict(status='completed',config=config,config_sha256=sha(path),dataset_sha256=config['dataset_sha256'],
        validation_best={k:v for k,v in best.items() if k!='model_state_dict'},target_count=6,
        metrics={'mae':1.,'rmse':1.,'mre':1.})
    (output/'result.json').write_text(json.dumps(result))
    return path,output,result


def test_full_saved_state_and_all_test_units_are_required(tmp_path, monkeypatch):
    path,output,_=fixture(tmp_path,monkeypatch)
    proof=validator.validate(path)
    assert proof['full_test_units']==3 and proof['held_out_coordinates']==6
    np.savez(output/'patient_statistics.npz',unit_ids=np.arange(2),group_ids=np.arange(2),
        statistics=np.array([[2.,2.,2.,2.]]*2))
    with pytest.raises(ValueError,match='unit coverage'):
        validator.validate(path)


def test_published_metric_must_match_full_group_statistics(tmp_path, monkeypatch):
    path,output,result=fixture(tmp_path,monkeypatch)
    result['metrics']['mae']=.01
    (output/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='independent metric'):
        validator.validate(path)


def test_unfinished_resume_cannot_count_as_full_training(tmp_path, monkeypatch):
    path,output,_=fixture(tmp_path,monkeypatch)
    state=torch.load(output/'resume.pt',weights_only=False);state['finished']=False
    torch.save(state,output/'resume.pt')
    with pytest.raises(ValueError,match='did not finish'):
        validator.validate(path)
