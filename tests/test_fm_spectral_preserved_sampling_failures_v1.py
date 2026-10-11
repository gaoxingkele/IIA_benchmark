import copy

import pytest
import torch

from scripts.flow_matching.audit_spectral_preserved_sampling_failures_v1 import checkpoint_budget


def original():
    return dict(step=10000,model={'weights':torch.tensor([1.0])},ema={'weights':torch.tensor([1.0])},
        opt={'param_groups':[{'params':[0,1]}],
             'state':{0:{'step':torch.tensor(10000.),'exp_avg':torch.tensor([.1])},
                      1:{'step':torch.tensor(10000.),'exp_avg':torch.tensor([.2])}}})


@pytest.mark.parametrize('fault',['epoch','optimizer_step','missing_step','missing_state','duplicate_group',
    'extra_state','model','ema','optimizer','rng'])
def test_partial_or_nonfinite_checkpoint_cannot_be_promoted_to_full_native_training(fault):
    checkpoint=original()
    result=checkpoint_budget(checkpoint,10000,torch)
    assert result['complete_optimizer_update_budget_verified']
    assert not result['RNG_state_saved'] and not result['exact_post_training_sampling_RNG_resume_certified']
    bad=copy.deepcopy(checkpoint)
    if fault=='epoch': bad['step']=9900
    elif fault=='optimizer_step': bad['opt']['state'][1]['step']=torch.tensor(9900.)
    elif fault=='missing_step': del bad['opt']['state'][1]['step']
    elif fault=='missing_state': del bad['opt']['state'][1]
    elif fault=='duplicate_group': bad['opt']['param_groups'][0]['params'].append(1)
    elif fault=='extra_state': bad['opt']['state'][2]=copy.deepcopy(bad['opt']['state'][0])
    elif fault in ['model','ema']: bad[fault]['weights'][0]=float('nan')
    elif fault=='optimizer': bad['opt']['state'][0]['exp_avg'][0]=float('inf')
    else: bad['RNG_state']='requires new restoration audit'
    with pytest.raises(ValueError): checkpoint_budget(bad,10000,torch)
