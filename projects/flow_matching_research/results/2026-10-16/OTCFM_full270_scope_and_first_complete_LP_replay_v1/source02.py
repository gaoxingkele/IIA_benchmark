import h5py
import numpy as np
import pytest
from scripts.flow_matching.package_otcfm_released_EB_CITE_inputs_v1 import labels, package
from scripts.flow_matching.timegan_native_common_v1 import sha


def test_all_five_EB_timepoints_keep_their_original_order_without_invented_day_column(tmp_path):
    path = tmp_path / 'labels.h5'
    with h5py.File(path, 'w') as f:
        g = f.create_group('obs/sample_labels')
        g.create_dataset('categories', data=np.array(['Day 00-03','Day 06-09','Day 12-15','Day 18-21','Day 24-27'], dtype=h5py.string_dtype()))
        g.create_dataset('codes', data=[4,0,1,3,2])
    with h5py.File(path, 'r') as f:
        y, names = labels(f, 'sample_labels')
    assert y.tolist() == [4,0,1,3,2] and len(names) == 5 and names[4] == 'Day 24-27'


def test_cite_keeps_real_nonconsecutive_days_without_remapping_to_elapsed_days(tmp_path):
    path = tmp_path / 'labels.h5'
    with h5py.File(path,'w') as f: f.create_dataset('obs/day',data=[7,2,4,3])
    with h5py.File(path,'r') as f: y, names = labels(f,'day')
    assert y.tolist() == [7,2,4,3] and names == ['2','3','4','7']


def test_complete_PC_columns_and_all_source_cells_survive_packaging_without_raw_overwrite(tmp_path):
    source = tmp_path / 'source.h5'; pcs = np.arange(24, dtype=np.float32).reshape(4,6)
    with h5py.File(source,'w') as f:
        obs=f.create_group('obs'); obs.attrs['_index']='cell_id'
        obs.create_dataset('cell_id',data=np.array(['a','b','c','d'],dtype=h5py.string_dtype()))
        obs.create_dataset('day',data=[2,3,4,7]);f.create_dataset('obsm/X_pca',data=pcs)
    digest=sha(source)
    config=dict(output_directory=str(tmp_path/'packaged'),original_scope=dict(papers=45),inputs=[dict(id='fixture_not_benchmark',
        source=str(source),source_sha256=digest,label_key='day',cells=4,released_PC_dimensions=6,filename='input.npz',
        model_timepoint_counts={'2':1,'3':1,'4':1,'7':1},boundary='Packaging fixture only; no benchmark claim')])
    result=package(config)
    with np.load(result['records'][0]['path'],allow_pickle=False) as arrays:
        assert np.array_equal(arrays['pcs'],pcs) and arrays['pcs'].shape == (4,6)
    assert sha(source)==digest and result['model_training_updates']==0 and not result['new_benchmark_performance_claim']
    with pytest.raises(FileExistsError): package(config)
