"""Acquire all four official donor13176 days, preserving raw and processed data."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG='configs/acquisition/fm_CITE13176_GEO_official.v1.json'


def validate_sources(config):
    items=config['sources']
    expected={(d,k) for d in [2,3,4,7] for k in ['raw_10x_filtered_counts','GEO2025_processed_multimodal']}
    if (len(items)!=8 or {(s['day'],s['kind']) for s in items}!=expected
            or len({s['filename'] for s in items})!=8 or config['donor']!='13176'
            or config['historical_Kaggle_or_paper_processed_identity_certified']
            or not config['raw_and_processed_kept_separate']):
        raise ValueError('All8raw/processed files for every original donor13176 day required')
    for item in items:
        if (not item['url'].startswith('https://ftp.ncbi.nlm.nih.gov/geo/') or item['expected_bytes']<=0
                or Path(item['filename']).name!=item['filename'] or item['publisher_checksum_available']):
            raise ValueError('Exact official GEO file URL/length and honest missing publisher checksum required')


def stream_hdf5(path,expected_bytes,block_bytes=16*1024*1024):
    """Read every HDF5 dataset, including compressed leaf payloads, in bounded blocks."""
    import h5py
    import numpy as np
    path=Path(path)
    if path.stat().st_size!=expected_bytes or not h5py.is_hdf5(path):
        raise ValueError('Complete official byte length and actual HDF5 payload required')
    records=[];hints={}
    with h5py.File(path,'r') as f:
        def inspect(name,dataset):
            if not isinstance(dataset,h5py.Dataset):return
            shape=dataset.shape;nonfinite=0;blocks=0;elements=0
            if shape is None:
                records.append(dict(name=name,shape=None,empty_HDF5_dataspace=True,blocks=0));return
            if not shape:
                arrays=[dataset[()]]
            else:
                row_bytes=max(1,int(np.prod(shape[1:],dtype=np.int64)))*max(8,dataset.dtype.itemsize)
                step=max(1,block_bytes//row_bytes)
                arrays=(dataset[i:i+step] for i in range(0,shape[0],step))
            for a in arrays:
                a=np.asarray(a);blocks+=1;elements+=a.size
                if np.issubdtype(a.dtype,np.number):nonfinite+=int(np.count_nonzero(~np.isfinite(a)))
            expected=int(np.prod(shape,dtype=np.int64)) if shape else 1
            if elements!=expected:raise ValueError('Every HDF5 leaf coordinate must actually be read')
            records.append(dict(name=name,shape=list(shape),dtype=str(dataset.dtype),elements_read=elements,
                numeric_nonfinite=nonfinite,blocks=blocks,fletcher32=bool(dataset.fletcher32),compression=dataset.compression))
        f.visititems(inspect)
        if 'matrix/shape' in f:
            hints['raw10x_features_cells']=list(map(int,f['matrix/shape'][()]))
        for name in ['mod/rna/obs/_index','mod/RNA/obs/_index','mod/rna/var/_index','mod/RNA/var/_index','obs/_index']:
            if name in f:hints[name]=list(f[name].shape)
    if not records:raise ValueError('Real data HDF5 leaves required')
    return dict(path=str(path),bytes=expected_bytes,sha256=sha(path),every_HDF5_leaf_payload_read=True,
        leaf_datasets=len(records),total_leaf_elements_read=sum(r.get('elements_read',0) for r in records),
        numeric_nonfinite=sum(r.get('numeric_nonfinite',0) for r in records),cohort_shape_hints=hints,
        arrays=records,publisher_checksum_available=False,publisher_checksum_verified=False,
        SHA256_is_local_integrity_fingerprint=True,historical_paper_processed_identity_certified=False)


def acquire(config):
    validate_sources(config)
    for pin in config['official_metadata']+config['code_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Pinned official metadata/acquisition code changed')
    base=Path(config['output_directory']);base.mkdir(parents=True,exist_ok=False)
    state=ROOT/config['state_root'];state.mkdir(parents=True,exist_ok=True)
    queue=state/'aria2c_inputs.txt'
    with queue.open('x',encoding='utf-8',newline='\n') as stream:
        for item in config['sources']:
            stream.write(item['url']+'\n  dir='+str(base)+'\n  out='+item['filename']+'\n')
    command=[config['aria2c'],'--all-proxy='+config['proxy'],'--input-file='+str(queue),
        '--max-concurrent-downloads=2','--max-connection-per-server=4','--split=4','--continue=true',
        '--allow-overwrite=false','--auto-file-renaming=false','--connect-timeout=20','--timeout=30',
        '--max-tries=5','--retry-wait=2','--console-log-level=warn','--summary-interval=30']
    with (state/'aria2c.stdout.log').open('xb') as out,(state/'aria2c.stderr.log').open('xb') as err:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        import psutil
        exclusive(state/'aria2c_launch.json',dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),command=command))
        code=child.wait()
    exclusive(state/'aria2c_exit.json',dict(exit_code=code))
    if code:raise RuntimeError('Preserve every partial download and failure;do not overwrite or silently restart')
    records=[]
    for item in config['sources']:
        proof=stream_hdf5(base/item['filename'],item['expected_bytes']);proof.update(day=item['day'],kind=item['kind'],url=item['url'])
        exclusive(base/(item['filename']+'.integrity.json'),proof);records.append(proof)
        print(item['filename']+' complete full-leaf HDF5 read / '+proof['sha256'],flush=True)
    report=dict(status='all8official_payloads_complete_and_fully_read',records=records,
        files=8,total_bytes=sum(r['bytes'] for r in records),donor='13176',days=[2,3,4,7],
        raw_and_processed_kept_separate=True,publisher_checksum_available=False,publisher_checksum_verified=False,
        original_source_files_overwritten=False,historical_Kaggle_or_paper_processed_identity_certified=False,
        configuration_sha256=sha(ROOT/CONFIG),original_scope=config['original_scope'],all_original_experiments_complete=False)
    exclusive(base/'all8_payload_integrity.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG);args=parser.parse_args()
    acquire(read(ROOT/args.config))


if __name__=='__main__':main()
