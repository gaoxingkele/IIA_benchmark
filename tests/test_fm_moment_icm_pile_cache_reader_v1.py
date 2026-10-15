import json
from pathlib import Path
import os
import subprocess
import numpy as np
import pytest
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as reader
from scripts.flow_matching import prepare_moment_icm_full_epoch_plan_v1 as full
from scripts.flow_matching.moment_icm_channel_data_v1 import channel_groups,BatchReference


def test_bounded_byte_maps_preserve_cross_block_values_and_close_handles(tmp_path):
    path=tmp_path/'x.npy';a=np.arange(20000,dtype=np.float64);np.save(path,a)
    cache=reader.BoundedArrays(cache_bytes=8192,block_bytes=1024)
    for start,count in [(0,300),(293,400),(17001,2048),(4,1024)]:assert cache.flat(path,start,count).tobytes()==a[start:start+count].tobytes()
    assert cache.peak_mapped_bytes<=8192 and cache.mapped_bytes<=8192
    cache.close();assert cache.mapped_bytes==0 and not cache.maps


@pytest.mark.parametrize('family',['classification','long_forecast','monash'])
def test_actual_closed_cache_values_rows_and_channel_groups_at_all_endpoints(family):
    sources,_=full.complete_sources()
    candidates=[s for s in sources if s['family']==family]
    if family=='classification':s=next(s for s in candidates if s['native_channels']>1)
    elif family=='long_forecast':s=next(s for s in candidates if s['native_channels']==862)
    else:s=next(s for s in candidates if s['dataset_key'].endswith('m1_monthly_dataset'))
    directory=Path(s['directory'])/('TRAIN' if family=='classification' else '')
    p=reader.PileCacheReader([s],epoch_seed=13,cache_bytes=4*2**20)
    groups=channel_groups(s['native_channels'],13,s['dataset_key'].split(':',1)[1],'appendix_D_partition8_remainder_replacement')
    try:
        for index in sorted({0,s['items']//2,s['items']-1}):
            item=p.item(0,index);base,g=divmod(index,len(groups));channels=list(groups[g])
            if family=='classification':
                expected=np.load(directory/'values.npy',mmap_mode='r')[base,channels,:].copy()
                mask=np.load(directory/'input_mask.npy',mmap_mode='r')[base].copy()
                assert item['provenance']['original_instance_row']==np.load(directory/'original_instance_rows.npy',mmap_mode='r')[base]
            elif family=='long_forecast':
                expected=np.load(directory/'values_TC.npy',mmap_mode='r')[base:base+256,channels].T.copy()
                mask=np.load(directory/'input_mask_window.npy');assert np.array_equal(item['provenance']['raw_CSV_rows'],np.load(directory/'original_CSV_rows.npy',mmap_mode='r')[base:base+256])
            else:
                case=int(np.load(directory/'source_shuffle_order.npy',mmap_mode='r')[s['source_train_shuffle_positions'][0]+base])
                expected=np.load(directory/'values.npy',mmap_mode='r')[case].copy();mask=np.load(directory/'input_mask.npy',mmap_mode='r')[case].copy()
                assert item['provenance']['original_series_row']==np.load(directory/'source_series_rows.npy',mmap_mode='r')[case]
            assert item['values'].tobytes()==expected.tobytes() and item['input_mask'].tobytes()==mask.tobytes()
            assert np.array_equal(item['valid_targets'],np.broadcast_to(mask.astype(bool),expected.shape))
        assert p.arrays.peak_mapped_bytes<=4*2**20
    finally:p.close()


def test_original_anomaly_never_defaults_to_all_true_and_real_raw_binding_preserves_identity(tmp_path):
    sources,_=full.complete_sources();s=next(s for s in sources if s['family']=='anomaly_original')
    p=reader.PileCacheReader([s],epoch_seed=13)
    try:
        with pytest.raises(ValueError,match='raw observation target binding'):p.item(0,0)
    finally:p.close()
    binding=reader.audit_original_anomaly_targets(s,full.ROOT,tmp_path/'observed')
    p=reader.PileCacheReader([s],epoch_seed=13,original_anomaly_target_bindings={s['dataset_key']:binding})
    try:
        item=p.item(0,s['items']-1);base=s['items']-1
        expected=np.load(Path(s['directory'])/'values_T.npy',mmap_mode='r')[base:base+256]
        assert item['values'].tobytes()==expected.reshape(1,256).tobytes()
        assert binding['raw_observation_verified'] and not binding['anomaly_labels_used']
        batch=p.batch(BatchReference(1,((0,0),(0,base))))
        assert batch['valid_targets'].shape==(2,1,256) and np.isfinite(batch['values']).all()
    finally:p.close()


def test_real_finite_control_reader_retains_raw_observation_loss_mask_in_native_environment(tmp_path):
    code=r'''
from pathlib import Path
import sys
from scripts.flow_matching import prepare_moment_icm_anomaly_finite_control_v3 as p
from scripts.flow_matching import moment_icm_pile_cache_reader_v1 as r
c=p.read(p.ROOT/p.CONFIG);job=p.verify(c)[0];p.source.install_private_path(c)
import numpy as np,pandas as pd
raw,rows=p.source.producer.csv_identity(job['raw']);observed=np.isfinite(pd.read_csv(p.ROOT/job['raw']['path']).iloc[:,0].to_numpy())
cls=p.source.producer.source_anomaly_class(p.ROOT/c['source_root']);data=p.source.producer.source_dataset(cls,p.ROOT/job['raw']['path'],'test')
out=Path(sys.argv[1])/'fixture';p.export_split(data,job,rows,observed,out)
s=dict(dataset_key='anomaly:'+job['id'],family='anomaly_finite_control',split='train',width=1,items=len(data.data)-255,
 native_channels=1,channel_groups=1,training_base_cases_or_windows=len(data.data)-255,directory=str(out))
# Fixture uses this actual TEST-shaped source output solely to exercise tail NaNs;
# it is never a registered training binding or experiment performance result.
reader=r.PileCacheReader([s],epoch_seed=13,cache_bytes=65536,block_bytes=4096)
try:
 index=s['items']-1;item=reader.item(0,index);valid=np.load(out/'valid_targets_T.npy')[index:index+256]
 assert np.array_equal(item['valid_targets'],valid.reshape(1,256))
 assert np.isfinite(item['values']).all() and not valid.all()
 loss=p.reconstruction_loss_mask(np.zeros((1,256)),item['valid_targets'],item['input_mask'])
 assert np.array_equal(loss,item['valid_targets'])
 assert reader.arrays.peak_mapped_bytes<=65536
finally:reader.close()
assert 'torch' not in sys.modules
'''
    result=subprocess.run([str(full.ROOT/'.venv/moment-author-py310/Scripts/python.exe'),'-B','-X','utf8','-c',code,str(tmp_path)],cwd=full.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout[-1000:]+result.stderr[-4000:]
