import copy
import json

import numpy as np
import pytest

from scripts.flow_matching import run_cfm_ts_original_job as worker


def prepared(tmp_path,monkeypatch):
    job={'id':'coverage','epochs':4,'expected_optimizer_updates':4,'test_trajectories':1,
         'test_points_per_trajectory':2,'observed_dimensions':1,'output_directory':'run','data_path':'data.npz'}
    directory=tmp_path/'run';directory.mkdir()
    truth=np.array([[[1.],[2.]]]); times=np.array([[.1,.2]]); ids=np.array([0])
    np.savez(tmp_path/'data.npz',test_states=truth,test_times=times,test_ids=ids)
    np.savez(tmp_path/'predictions.npz',predictions=truth,truth=truth,times=times,test_ids=ids)
    result={'status':'completed','job_sha256':worker.fingerprint(job),'epochs_completed':4,'optimizer_updates':4,
            'test_prediction_shape':[1,2,1],'diagnostic_smoke':False,'author_equivalence_certified':False,
            'training_losses':[1.,.9,.8,.7],'test_trajectory_ids':[0],'metrics':{'MSE':0.},
            'artifacts':[{'path':str(tmp_path/'predictions.npz'),'sha256':worker.sha(tmp_path/'predictions.npz')}]}
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    return job,result,directory/'result.json'


def test_full_prediction_metric_zero_is_measured(tmp_path,monkeypatch):
    job,result,path=prepared(tmp_path,monkeypatch);path.write_text(json.dumps(result))
    assert worker.complete(job)


@pytest.mark.parametrize('field,value', [('epochs_completed',3),('optimizer_updates',3),('training_losses',[1.]),
                                       ('test_prediction_shape',[1,1,1]),('metrics',{'MSE':.5})])
def test_partial_budget_coverage_or_wrong_mse_rejected(tmp_path,monkeypatch,field,value):
    job,result,path=prepared(tmp_path,monkeypatch); result[field]=value;path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        worker.complete(job)


def test_self_consistent_saved_artifact_cannot_replace_ground_truth(tmp_path,monkeypatch):
    job,result,path=prepared(tmp_path,monkeypatch)
    np.savez(tmp_path/'predictions.npz',predictions=[[[7.],[8.]]],truth=[[[7.],[8.]]],times=[[.1,.2]],test_ids=[0])
    result['artifacts'][0]['sha256']=worker.sha(tmp_path/'predictions.npz');path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='ground truth'):
        worker.complete(job)
