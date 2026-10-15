import json
import numpy as np
import pytest
from scripts.flow_matching import verify_moment_icm_PILE_monash_v1 as s


def fixture(tmp_path):
    directory=tmp_path/'cache';directory.mkdir()
    lengths=np.array([8,300,256],dtype=np.int64);order=np.random.RandomState(13).choice(3,3,replace=False).astype(np.int64)
    masks=(np.arange(256)[None,:]>=np.maximum(0,256-lengths[:,None])).astype(np.float64)
    values=np.arange(768,dtype=np.float64).reshape(3,1,256);values[:,0,:]*=masks
    for name,array in [('values.npy',values),('input_mask.npy',masks),('source_series_rows.npy',np.arange(3,dtype=np.int64)),
        ('source_shuffle_order.npy',order),('source_series_lengths.npy',lengths)]:np.save(directory/name,array)
    raw=tmp_path/'example.tsf';raw.write_text('@attribute series_name string\n@frequency monthly\n@horizon 12\n@data\n'+
        '\n'.join(name+':'+','.join('1' for _ in range(n)) for name,n in zip(['A','B','C'],lengths))+'\n')
    with (directory/'series_metadata.jsonl').open('w') as f:
        for i,(name,n) in enumerate(zip(['A','B','C'],lengths)):
            f.write(json.dumps(dict(source_row=i,series_name=name,pre_resampling_length=int(n),raw_series_index=i,
                source_chunk_index=None,raw_time_start=0,raw_time_end_exclusive=int(n),original_special_tail_omitted=0))+'\n')
    train=lengths[order[:1]]
    stats=dict(actual_training_rows=1,training_temporal_min=int(train.min()),training_temporal_max=int(train.max()),
        training_temporal_total_points=int(train.sum()),training_rows_length_ge_window=int((train>=256).sum()))
    profile=dict(source_length_field=3,**stats)
    record=dict(series=3,channels=1,sequence_length=256,source_order_cache_with_exact_shuffle_index=True,
        files_sha256={p.name:s.sha(p) for p in directory.iterdir()},original_split_positions=dict(train=[0,1],val=[1,3],test=[3,3]),
        observed_source_training_length_metadata=stats,source_count_gate_selected=False,separate_temporal_control_selected=bool(train.max()>=256),
        dtype='float64',mask_dtype='float64')
    return directory,record,profile,raw


def test_full_raw_names_lengths_permutation_split_and_exact_padding_layout(tmp_path):
    d,r,p,raw=fixture(tmp_path);proof=s.verify_cache(d,r,p,raw,[])
    assert proof['signal_coordinates']==768 and proof['source_padding_coordinates']==248
    assert proof['arrays']==5 and proof['exact_seed13_shuffle_and_splits']


@pytest.mark.parametrize('kind',['shuffle','source_rows','mask','padding','metadata'])
def test_semantic_corruption_is_rejected_after_refreshing_hashes(tmp_path,kind):
    d,r,p,raw=fixture(tmp_path)
    s.verify_cache(d,r,p,raw,[])
    if kind=='shuffle':name='source_shuffle_order.npy';np.save(d/name,np.arange(3,dtype=np.int64))
    elif kind=='source_rows':name='source_series_rows.npy';np.save(d/name,np.array([1,0,2],dtype=np.int64))
    elif kind=='mask':
        name='input_mask.npy';a=np.load(d/name);a[0,0]=1;np.save(d/name,a)
    elif kind=='padding':
        name='values.npy';a=np.load(d/name);a[0,0,0]=5;np.save(d/name,a)
    else:
        name='series_metadata.jsonl';text=(d/name).read_text().replace('"series_name": "A"','"series_name": "wrong"');(d/name).write_text(text)
    r['files_sha256'][name]=s.sha(d/name)
    with pytest.raises(ValueError):s.verify_cache(d,r,p,raw,[])
