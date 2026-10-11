import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.flow_matching.audit_crossad_full_release_completion_v1 import release_only,channel_mean_energy


def test_release_evaluation_cannot_be_counted_as_fresh_training():
    job=dict(id='released',seed=2025,mode='release_checkpoint',author_recipe='release_checkpoint',train_parameters={'train_epochs':20})
    result=dict(id='released',seed=2025,mode='release_checkpoint',status='completed',
        experiment_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
        paper_equivalence=False,strict_TAB_result=False,
        details=dict(epochs_completed=0,training_forward_calls=0,training_seconds=0,
            declared_train_parameters=job['train_parameters'],algorithm_weights={'all_algorithm_weights_strictly_loaded':True}))
    release_only(job,result)
    for field in ['epochs_completed','training_forward_calls','training_seconds']:
        changed=copy.deepcopy(result);changed['details'][field]=1
        with pytest.raises(ValueError):release_only(job,changed)
    changed=copy.deepcopy(result);changed['mode']='fresh_train'
    with pytest.raises(ValueError):release_only(job,changed)


def test_multichannel_energy_keeps_all_timestamps_and_averages_channels():
    energy=np.array([[1.,3.],[5.,7.],[9.,11.]])
    assert np.array_equal(channel_mean_energy(energy,3),[2.,6.,10.])
    for wrong in [energy.reshape(-1),energy[:-1],np.array([[1.,3.],[5.,np.nan],[9.,11.]])]:
        with pytest.raises(ValueError):channel_mean_energy(wrong,3)
