import copy
import numpy as np
import pytest

from scripts.flow_matching.audit_crossad_full_native_completion_v1 import training_budget, complete_scores


def test_early_stop_requires_original_patience_and_every_original_training_batch():
    job = dict(train_parameters=dict(train_epochs=20, patience=3, batch_size=128),
        model_parameters=dict(seq_len=96))
    details = dict(declared_train_parameters=job['train_parameters'], epochs_completed=5,
        native_points=dict(train=46653), training_forward_calls=1820)
    trace = dict(validation_losses=[.04,.03,.031,.032,.033], epochs_finished=5,
        declared_max_epochs=20, training_forward_calls=1820)
    assert training_budget(job,details,trace)['best_validation_epoch']==2
    bad = copy.deepcopy(trace); bad['validation_losses'][-1]=.02
    with pytest.raises(ValueError):
        training_budget(job,details,bad)
    bad = copy.deepcopy(details); bad['training_forward_calls']-=1
    with pytest.raises(ValueError):
        training_budget(job,bad,trace)


def test_full_native_tail_and_raw_source_labels_must_not_be_dropped_or_reordered():
    labels=np.array([0,0,1,0,1],dtype=np.uint8)
    lengths=dict(train=1,validation=1,test=3)
    arrays=dict(train=np.ones(1),validation=np.ones(1),test=np.ones(3),
        labels=labels[2:],source_rows=np.arange(2,5))
    complete_scores(arrays,labels,2,lengths)
    for key,value in [('test',np.ones(2)),('test',np.array([1,1,np.nan])),
            ('labels',np.array([0,1,0])),('source_rows',np.array([2,3,3]))]:
        bad=copy.deepcopy(arrays); bad[key]=value
        with pytest.raises(ValueError):
            complete_scores(bad,labels,2,lengths)
