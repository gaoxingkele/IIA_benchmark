"""Fetch a pinned author archive and verify every raw blob against its Git tree."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import uuid
import zipfile

from scripts.data_acquisition.download_public_datasets import find_aria2

ROOT=Path(__file__).resolve().parents[2]


def blob_identity(content):
    return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()


def safe_relative(value):
    path=PurePosixPath(value)
    if (path.is_absolute() or not path.parts or any(p in ('..','.') or ':' in p or '\\' in p for p in path.parts)
            or value!=path.as_posix()):
        raise ValueError('Unsafe or noncanonical author asset path')
    return path


def verified_archive(archive,config):
    """Require the complete registered tree; never interpret author symlinks."""
    expected={a['upstream_path']:a for a in config['assets']}
    if len(expected)!=len(config['assets']):raise ValueError('Duplicate registered upstream path')
    prefix=config['archive_prefix']
    safe_relative(prefix)
    with zipfile.ZipFile(archive) as z:
        actual={}
        for member in z.infolist():
            name=safe_relative(member.filename.rstrip('/'))
            if name.parts[0]!=prefix:raise ValueError('Archive does not use pinned commit root')
            if member.is_dir():continue
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            if relative in actual:raise ValueError('Duplicate archive member')
            actual[relative]=member
        if set(actual)!=set(expected):raise ValueError('Complete registered author Git tree differs')
        records=[]
        for path,asset in expected.items():
            safe_relative(path)
            if asset['upstream_mode'] not in ('100644','100755','120000'):
                raise ValueError('Unsupported author Git mode')
            content=z.read(actual[path])
            if len(content)!=asset['expected_size'] or blob_identity(content)!=asset['git_blob_sha1']:
                raise ValueError('Author Git blob differs: '+path)
            records.append(dict(asset,size=len(content),sha256=hashlib.sha256(content).hexdigest(),
                preserved_relative_path=path+'.upstream_link.txt' if asset['upstream_mode']=='120000' else path,
                upstream_link_followed=False))
        return records


def preserve_assets(archive,config,records,root=ROOT):
    relative=Path(config['asset_root'])
    allowed=root/'data/public_datasets/flow_matching'
    target=root/relative
    if relative.is_absolute() or not target.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Author source destination escaped registered data storage')
    with zipfile.ZipFile(archive) as z:
        for item in records:
            dest=target/item['preserved_relative_path']
            if not dest.resolve().is_relative_to(target.resolve()):raise ValueError('Raw author blob destination escaped')
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest()!=item['sha256']:
                    raise FileExistsError('Preserve conflicting original asset: '+str(dest))
                item['existing_verified']=True
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                content=z.read(config['archive_prefix']+'/'+item['upstream_path'])
                with dest.open('xb') as f:f.write(content)
                item['existing_verified']=False
            item['resolved_path']=str(dest.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    config_path=ROOT/args.config;config=json.loads(config_path.read_text(encoding='utf-8'))
    expected_url='https://codeload.github.com/'+config['repository_slug']+'/zip/'+config['commit']
    if config['archive_url']!=expected_url:raise ValueError('Only pinned official author commit URL allowed')
    receipt_root=ROOT/config['receipt_directory']
    if receipt_root.exists():raise FileExistsError('Preserve every acquisition attempt receipt')
    receipt_root.mkdir(parents=True)
    archive=ROOT/config['archive_path'];archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        aria=find_aria2()
        if not aria:raise RuntimeError('Registered downloads require aria2c')
        attempt=archive.parent/'_download_attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        command=[aria,'--all-proxy='+config['proxy'],'--split=1','--max-connection-per-server=1','--max-tries=3',
            '--retry-wait=2','--connect-timeout=20','--timeout=60','--file-allocation=none','--allow-overwrite=false',
            '--auto-file-renaming=false','--summary-interval=0','--console-log-level=error','--dir='+str(attempt),
            '--out='+archive.name,config['archive_url']]
        with (attempt/'aria2.log').open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        candidate=attempt/archive.name;verified_archive(candidate,config)
        with candidate.open('rb') as src,archive.open('xb') as dst:shutil.copyfileobj(src,dst)
    records=verified_archive(archive,config);preserve_assets(archive,config,records)
    receipt=dict(captured_utc=datetime.now(timezone.utc).isoformat(),config_path=args.config,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),repository=config['repository_slug'],
        commit=config['commit'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_size=archive.stat().st_size,
        downloader='aria2c',proxy=config['proxy'],assets=records,all_registered_author_blobs_verified=True,
        method_native_training_complete=False,performance_result=False,boundary=config['boundary'])
    (receipt_root/'acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(dict(commit=config['commit'],verified_author_blobs=len(records),archive_bytes=archive.stat().st_size))


if __name__=='__main__':main()
