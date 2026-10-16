from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import h5py
import numpy as np
import pytest
from scripts.flow_matching import acquire_CITE13176_GEO_official_v1 as s


def test_all_eight_official_raw_processed_days_are_required():
    q=s.read(s.ROOT/s.CONFIG);s.validate_sources(q)
    for fault in ('missing_day','duplicate','historical_claim','fake_publisher_checksum'):
        c=deepcopy(q)
        if fault=='missing_day':c['sources']=c['sources'][:-1]
        elif fault=='duplicate':c['sources'][-1]=deepcopy(c['sources'][0])
        elif fault=='historical_claim':c['historical_Kaggle_or_paper_processed_identity_certified']=True
        else:c['sources'][0]['publisher_checksum_available']=True
        with pytest.raises(ValueError):s.validate_sources(c)


def test_compressed_numeric_strings_scalar_and_empty_payloads_all_read_and_fingerprinted():
    with tempfile.TemporaryDirectory(prefix='cite-full-leaf-') as directory:
        p=Path(directory)/'fixture.h5'
        with h5py.File(p,'w') as f:
            f.create_dataset('counts',data=np.arange(120).reshape(30,4),compression='gzip',fletcher32=True)
            f.create_dataset('scalar',data=7)
            f.create_dataset('empty',shape=(0,4),dtype='f8')
            f.create_dataset('barcodes',data=np.array(['AAA','BBB'],dtype=h5py.string_dtype()))
        r=s.stream_hdf5(p,p.stat().st_size,block_bytes=64)
        assert r['leaf_datasets']==4 and r['total_leaf_elements_read']==123 and r['numeric_nonfinite']==0
        assert r['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        assert next(x for x in r['arrays'] if x['name']=='counts')['blocks']>1
        assert not r['publisher_checksum_verified'] and r['every_HDF5_leaf_payload_read']
        with pytest.raises(ValueError):s.stream_hdf5(p,p.stat().st_size+1)
        truncated=Path(directory)/'partial.h5';truncated.write_bytes(p.read_bytes()[:24])
        with pytest.raises((ValueError,OSError)):s.stream_hdf5(truncated,truncated.stat().st_size)
