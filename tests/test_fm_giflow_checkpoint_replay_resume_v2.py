import ast
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.flow_matching import audit_giflow_checkpoint_replay_resume_v2 as replay


def test_resume_preserves_original_sampler_and_full_batch_mean():
    source=(replay.ROOT/'experiments/runs/flow_matching_campaign/sources/giflow/original/main.py').read_text(encoding='utf-8')
    original=replay.native_test_function(source)
    resumed=replay.native_test_function(source,dict(mae=[1.,4.],mse=[2.,16.],mape=[3.,6.]))
    left,right=original.body[0].body,resumed.body[0].body
    assert ast.dump(left[0])==ast.dump(right[0])
    assert ast.dump(left[4])==ast.dump(right[4])
    assert [n.value.elts[0].value for n in right[1:4]]==[1.,2.,3.]
    assert 'num_steps = 20' in ast.unparse(resumed)
    assert 'np.mean(losses_mae)' in ast.unparse(resumed)
    assert 'optimizer' not in ast.unparse(resumed)


def test_resume_rejects_altered_author_sampler():
    source=(replay.ROOT/'experiments/runs/flow_matching_campaign/sources/giflow/original/main.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError):
        replay.native_test_function(source.replace('num_steps = 20','num_steps = 19'),dict(mae=[],mse=[],mape=[]))


@pytest.fixture
def prefix_case(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(replay,'ROOT',tmp_path)
    batches=[];registered=[];originals=[]
    for index,windows in enumerate([5,3]):
        truth=torch.arange(1,windows*2+1,dtype=torch.float32).reshape(windows,2,1)
        mask=torch.ones_like(truth,dtype=torch.bool)
        batch=SimpleNamespace(values=(None,None,truth,None,mask))
        batch.to=lambda device,b=batch:b
        batches.append(batch)
        values=dict(target=truth[mask].numpy(),prediction=(truth[mask]+index+1).numpy(),
            masked_flat_positions=np.flatnonzero(mask.numpy()),batch_shape=np.asarray(mask.shape))
        old=tmp_path/f'original_{index}.npz';cache=tmp_path/f'cache_{index}.npz'
        np.savez_compressed(old,**values);np.savez_compressed(cache,**values)
        originals.append(dict(path=str(old),sha256=replay.sha(old)))
        registered.append(dict(batch=index,path=str(cache),sha256=replay.sha(cache)))
    replay.write(tmp_path/'progress.json',dict(completed_batches=2,required_batches=3,forward_calls=40,window_evaluations=160))
    config=dict(preserved_prefix=dict(batch_count=2,progress_path='progress.json',batches=registered),
        prediction_tolerance=dict(rtol=2e-4,atol=1e-3))
    native=SimpleNamespace(preprocess_fm_test=lambda batch,*unused:batch.values)
    return config,batches,native,originals


def test_prefix_uses_every_actual_target_and_native_float32_batch_metrics(prefix_case):
    config,batches,native,originals=prefix_case
    remaining,metrics,records=replay.verified_prefix(config,batches+[object()],native,None,None,originals,dict(full_test_batches=3))
    assert metrics['mae']==[1.,2.] and metrics['mse']==[1.,4.]
    assert sum(r['windows'] for r in records)==8 and sum(r['targets'] for r in records)==16
    assert all(r['agreement']['bitwise_identical'] for r in records)
    assert len(list(remaining))==1


def test_prefix_cannot_reuse_changed_actual_target(prefix_case):
    config,batches,native,originals=prefix_case
    batches[0].values[2][0,0,0]+=1
    with pytest.raises(ValueError,match='identity differs'):
        replay.verified_prefix(config,batches,native,None,None,originals,dict(full_test_batches=3))


def test_prefix_cannot_hide_a_changed_cache_or_missing_native_forwards(prefix_case):
    config,batches,native,originals=prefix_case
    config['preserved_prefix']['batches'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='intact contiguous'):
        replay.verified_prefix(config,batches,native,None,None,originals,dict(full_test_batches=3))
    config['preserved_prefix']['batches'][0]['sha256']=replay.sha(config['preserved_prefix']['batches'][0]['path'])
    progress=replay.ROOT/'progress.json';value=replay.read(progress);value['forward_calls']-=1;replay.write(progress,value)
    with pytest.raises(ValueError,match='budget changed'):
        replay.verified_prefix(config,batches,native,None,None,originals,dict(full_test_batches=3))
