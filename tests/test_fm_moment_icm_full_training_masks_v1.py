import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scripts.flow_matching import audit_moment_icm_full_training_masks_v1 as p
from tests.test_fm_moment_icm_full_training_input_bindings_v1 import actual_registries


@pytest.mark.parametrize('observed',[0,1,7,8,9,15,16,248,256])
def test_original_patchwise_keep_guarantee_covers_short_and_partial_patches(observed):
    mask=np.zeros(256);mask[:observed]=1
    r=p.analyze_padding(mask)
    assert r['complete_observed_patches']==observed//8
    assert r['retained_patches_for_every_source_random_draw']==p.math.ceil((observed//8)*.7)
    assert r['guaranteed_retained_observations']==(observed>=8)


def test_fragmented_observations_cannot_be_mistaken_for_one_complete_patch():
    mask=np.zeros(256);mask[::8]=1
    r=p.analyze_padding(mask)
    assert r['observed_time_points']==32 and r['complete_observed_patches']==0
    assert not r['guaranteed_retained_observations']
    mask[1]=.5
    with pytest.raises(ValueError,match='binary'):p.analyze_padding(mask)


@pytest.mark.parametrize('family',['classification','long_forecast','monash','anomaly_original'])
def test_complete_real_source_masks_match_independent_all_training_case_calculation(family):
    sources,_=p.inputs.epoch.complete_sources()
    source=next(s for s in sources if s['family']==family)
    files=p.inputs.expected_files(source,actual_registries(),{})
    reader=p.data.BoundedArrays(cache_bytes=2**20,block_bytes=4096)
    try:r=p.inspect_source(source,dict(input_files_sha256=files),reader)
    finally:reader.close()
    directory=Path(source['directory'])/('TRAIN' if family=='classification' else '')
    n=source['training_base_cases_or_windows'];groups=source['channel_groups']
    if family in ('classification','monash'):
        masks=np.load(directory/'input_mask.npy',allow_pickle=False)
        if family=='monash':
            start,end=source['source_train_shuffle_positions']
            ids=np.load(directory/'source_shuffle_order.npy',allow_pickle=False)[start:end]
            masks=masks[ids]
        else:masks=masks[:n]
        assert masks.shape==(n,256)
        counts=(masks.reshape(n,32,8).sum(-1)==8).sum(-1)
        histogram={str(int(k)):int((counts==k).sum())*groups for k in np.unique(counts)}
    else:
        mask=np.load(directory/'input_mask_window.npy',allow_pickle=False)
        counts=np.array([(mask.reshape(32,8).sum(-1)==8).sum()])
        histogram={str(int(counts[0])):source['items']}
    assert r['full_patch_count_histogram_over_all_training_items']==histogram
    assert r['source_items']==source['items'] and r['minimum_complete_observed_patches']==counts.min()
    assert r['guaranteed_nonempty_RevIN_keep_for_every_original_random_draw']==bool((counts>0).all())


def test_monash_training_shuffle_and_incompatible_case_ids_are_never_dropped(tmp_path):
    masks=np.ones((3,256));masks[2]=0
    np.save(tmp_path/'input_mask.npy',masks);np.save(tmp_path/'source_shuffle_order.npy',np.array([2,0,1],dtype=np.int64))
    source=dict(directory=str(tmp_path),family='monash',dataset_key='monash:fixture',training_base_cases_or_windows=2,
        items=2,channel_groups=1,width=1,source_train_shuffle_positions=[1,3])
    record=dict(input_files_sha256={str(file):p.sha(file) for file in tmp_path.glob('*.npy')})
    reader=p.data.BoundedArrays()
    try:
        result=p.inspect_source(source,record,reader)
        assert result['source_items']==2 and result['incompatible_training_items']==0
        source['source_train_shuffle_positions']=[0,2]
        result=p.inspect_source(source,record,reader)
        assert result['source_items']==2 and result['incompatible_training_items']==1
        assert result['incompatible_cases_or_shared_masks']==[dict(training_base_case=0,cache_case=2,affected_items=1)]
        assert sum(result['full_patch_count_histogram_over_all_training_items'].values())==2
    finally:reader.close()


@pytest.mark.parametrize('change',['sources','items','mask_ratio','stride','lock','guard'])
def test_full_registration_rejects_scope_mask_or_resource_substitutions(change):
    c=p.registration()
    if change=='sources':c['source_references']=1764
    elif change=='items':c['epoch_items']=100
    elif change=='mask_ratio':c['mask_ratio']=.1
    elif change=='stride':c['patch_len']=1
    elif change=='lock':c['resource_lock']='bypass.lock'
    else:c['settings']['minimum_commit_headroom_bytes']=1
    with pytest.raises(ValueError):p.verify(c)


def test_cold_mask_controller_resources_do_not_import_torch_or_construct_models():
    code='''
import json,sys
from scripts.flow_matching import audit_moment_icm_full_training_masks_v1 as p
from scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1 import install_facades
c=p.registration();assert len(p.verify(c))==2086
assert 'torch' not in sys.modules
api=install_facades();snapshot=api['resource_snapshot']()
assert 'torch' not in sys.modules and callable(api['resource_slot'])
assert c['settings']['maximum_worker_rss_bytes']==3*2**30
print(json.dumps(dict(passed=True,torch_imported=False,snapshot=snapshot)))
'''
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=p.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['torch_imported'] is False
