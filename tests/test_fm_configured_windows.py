import hashlib
import json
import numpy as np
import pytest

from scripts.flow_matching.run_configured_windows import run


def fixture(tmp_path):
    model={'id':'objective','entrypoint':'iia_benchmark.models.fm_objective_detectors:CFMObjectiveDetector',
           'parameters':{'window':4,'epochs':1,'hidden_size':8,'batch_size':4,'monte_carlo_samples':1,'score_times':[.5],'device':'cpu'}}
    path=tmp_path/'model.json';path.write_text(json.dumps(model))
    values=np.random.default_rng(1).normal(size=(4,4,2)).astype(np.float32)
    inputs=tmp_path/'windows.npz';np.savez(inputs,train=values,validation=values+1,test=values+2)
    return {'id':'preflight','purpose':'preflight','model_config':'model.json',
            'model_config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_npz':'windows.npz',
            'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'output_directory':'output'}


def test_scores_are_saved_with_frozen_inputs_and_idempotent_run(tmp_path):
    config=fixture(tmp_path);result=run(tmp_path,config)
    assert result['status']=='preflight_completed'
    assert result['score_shapes']['test']==[4,4]
    assert run(tmp_path,config)==result
    with pytest.raises(FileExistsError):run(tmp_path,{**config,'id':'another'})
    (tmp_path/'model.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):run(tmp_path,config)


def test_formal_run_requires_group_audit_and_preserves_existing_outputs(tmp_path):
    config=fixture(tmp_path)
    with pytest.raises(ValueError,match='grouped'):run(tmp_path,{**config,'purpose':'formal'})
    (tmp_path/'output').mkdir();(tmp_path/'output/user.txt').write_text('preserved')
    with pytest.raises(FileExistsError):run(tmp_path,config)
    assert (tmp_path/'output/user.txt').read_text()=='preserved'


def test_completed_score_corruption_is_detected_and_preserved(tmp_path):
    config=fixture(tmp_path);run(tmp_path,config)
    path=tmp_path/'output/raw_window_scores.npz';path.write_bytes(b'user altered artifact')
    with pytest.raises(ValueError,match='score artifact'):run(tmp_path,config)
    assert path.read_bytes()==b'user altered artifact'
