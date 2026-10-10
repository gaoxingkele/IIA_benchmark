import ast
import copy

import pytest

from scripts.flow_matching.audit_pi_completed_native_v2 import full_budget,readonly_audit


def test_readonly_pi_audit_preserves_calculation_skips_publication():
    tree=readonly_audit('def audit():\n    report={"ok":True}\n    target=danger()\n    publish(report)\n    return report\n')
    namespace={'danger':lambda:pytest.fail('Publication executed')}
    exec(compile(tree,'<test>','exec'),namespace)
    assert namespace['audit']()=={'ok':True}


def test_complete_original_pi_budget_and_short_epoch_rejected():
    job=dict(author_config=dict(data=dict(win_size=100,batch_size=256),training=dict(n_epochs=3)),
             data_audit=dict(expected_shapes=dict(train=[132481,25])))
    epochs=[dict(epoch=i,steps=518,early_stop=False) for i in range(1,4)]
    result=dict(native_train_points=132481,training_windows=132382,steps_per_epoch=518,epochs=epochs)
    assert full_budget(job,result,epochs)['optimizer_minibatches']==1554
    short=copy.deepcopy(result);short['epochs']=short['epochs'][:2]
    with pytest.raises(ValueError,match='early-stop'):full_budget(job,short,short['epochs'])
    short['epochs'][-1]['early_stop']=True
    assert full_budget(job,short,short['epochs'])['epochs_completed']==2
    shortened=copy.deepcopy(result);shortened['epochs'][0]['steps']=1
    with pytest.raises(ValueError,match='step budget'):full_budget(job,shortened,shortened['epochs'])


def test_original_pi_publication_boundary_change_rejected():
    with pytest.raises(ValueError,match='publication boundary'):
        readonly_audit('def audit():\n    report={}\n    unexpected(report)\n')
