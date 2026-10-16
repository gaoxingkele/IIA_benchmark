"""Fetch the exact code-referenced public EB asset and verify its Git blob hash."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile
from scripts.flow_matching.timegan_native_common_v1 import ROOT,read
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG='configs/acquisition/fm_TrajectoryNet_EB100_official.v1.json'


def verify_blob(path,expected_bytes,expected_git_blob_sha1):
    path=Path(path)
    if path.stat().st_size!=expected_bytes:raise ValueError('Exact public Git blob length required')
    blob=hashlib.sha1(b'blob '+str(expected_bytes).encode('ascii')+b'\0');raw=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):blob.update(block);raw.update(block)
    if blob.hexdigest()!=expected_git_blob_sha1:raise ValueError('Official Git object hash mismatch; preserve the original file without overwrite')
    return dict(bytes=expected_bytes,git_blob_sha1=blob.hexdigest(),sha256=raw.hexdigest())


def acquire(config):
    base=Path(config['output_directory']);base.mkdir(parents=True,exist_ok=True)
    path=base/config['filename'];state=ROOT/config['state_root'];state.mkdir(parents=True,exist_ok=True)
    if (base/'verified_acquisition.json').exists():raise FileExistsError('Preserve existing verified official download')
    if not path.exists() or (base/(config['filename']+'.aria2')).exists():
        command=[config['aria2c'],'--all-proxy='+config['proxy'],'--continue=true','--allow-overwrite=false',
            '--auto-file-renaming=false','--max-connection-per-server=4','--split=4','--connect-timeout=20',
            '--timeout=30','--max-tries=5','--retry-wait=2','--dir='+str(base),'--out='+config['filename'],config['url']]
        with (state/'aria2c.stdout.log').open('xb') as out,(state/'aria2c.stderr.log').open('xb') as err:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    proof=verify_blob(path,config['expected_bytes'],config['expected_git_blob_sha1'])
    if not zipfile.is_zipfile(path):raise ValueError('The official artifact must be a complete NPZ ZIP payload')
    receipt=dict(status='verified',path=str(path),**proof,source_commit=config['source_commit'],
        official_GitHub_blob_integrity_verified=True,publisher_raw_sha256_available=False,
        raw_sha256_is_independently_computed=True,original_data_overwritten=False,
        citation=config['citation'],all_original_experiments_complete=False)
    exclusive(base/'verified_acquisition.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=CONFIG)
    args=parser.parse_args()
    import json
    print(json.dumps(acquire(read(ROOT/args.config))))


if __name__=='__main__':main()
