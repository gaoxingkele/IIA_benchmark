"""Acquire full original VisA assets; preserve raw bytes and audit every split image."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import tarfile
import zipfile

import psutil

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe original image archive path')
    if any(part.rstrip(' .')!=part or part.split('.')[0].upper() in {
            'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]} for part in p.parts):
        raise ValueError('Unsafe Windows image archive path')
    return p


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def download(config,entry,state,update):
    path=ROOT/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve every old or partial raw image download')
    command=[config['aria2c'],'--all-proxy='+config['proxy'],'--split=4','--max-connection-per-server=4',
        '--min-split-size=16M','--continue=false','--allow-overwrite=false','--auto-file-renaming=false',
        '--file-allocation=none','--max-tries=4','--retry-wait=5','--connect-timeout=20','--timeout=45',
        '--max-overall-download-limit=16M','--summary-interval=10','--dir='+str(path.parent),'--out='+path.name,entry['url']]
    with path.with_suffix(path.suffix+'.download.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        actual=psutil.Process(child.pid)
        update(dict(phase='downloading_full_original_asset',asset=entry['id'],active_pid=child.pid,
            active_create_time=actual.create_time(),actual_command=actual.cmdline()))
        code=child.wait()
    if code:raise RuntimeError('Original image acquisition failed; preserve partial bytes: '+entry['id'])
    if entry.get('expected_bytes') is not None and path.stat().st_size!=entry['expected_bytes']:
        raise ValueError('Complete official image archive byte length differs')
    receipt=dict(id=entry['id'],url=entry['url'],path=str(path),bytes=path.stat().st_size,sha256=digest(path),
                 server_etag=entry.get('server_etag'),publisher_checksum_available=False,
                 checksum_boundary='Local streamed SHA256 plus complete archive/image/split verification; S3 multipart ETag is an object-version marker, not the full file MD5.')
    state['downloads'].append(receipt);update(dict(active_pid=None))
    return path


def extract_full_tar(archive,target):
    target=target.resolve();target.mkdir(parents=True,exist_ok=False)
    files=[];seen=set()
    with tarfile.open(archive,mode='r|') as tar:
        for member in tar:
            relative=safe_relative(member.name)
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError('Only complete original regular image files and directories accepted')
            if str(relative)=='.' and member.isdir():continue
            if relative.as_posix() in seen:raise ValueError('Duplicate original tar member')
            seen.add(relative.as_posix());path=(target/str(relative)).resolve()
            if target not in path.parents:raise ValueError('Original image path escapes configured raw root')
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            stream=tar.extractfile(member)
            with path.open('xb') as output:shutil.copyfileobj(stream,output,1024**2)
            if path.stat().st_size!=member.size:raise ValueError('Truncated original tar image member')
            files.append(dict(path=relative.as_posix(),bytes=member.size,sha256=digest(path)))
    return files


def extract_author(archive,target,commit):
    target=target.resolve();target.mkdir(parents=True,exist_ok=False)
    files=[];seen=set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            full=safe_relative(member.filename)
            if full.parts[0]!='spot-diff-'+commit:raise ValueError('Exact frozen original author archive root required')
            if member.is_dir():continue
            relative=PurePosixPath(*full.parts[1:])
            if relative.as_posix() in seen:raise ValueError('Duplicate original author source member')
            seen.add(relative.as_posix());target_file=(target/str(relative)).resolve()
            if target not in target_file.parents:raise ValueError('Author source escapes configured root')
            payload=source.read(member)
            target_file.parent.mkdir(parents=True,exist_ok=True)
            with target_file.open('xb') as stream:stream.write(payload)
            files.append(dict(path=relative.as_posix(),bytes=len(payload),sha256=digest(target_file)))
    return files


def validate_split_rows(rows,available_images,available_files):
    expected={'object','split','label','image','mask'}
    if not rows or set(rows[0])!=expected:raise ValueError('Exact original VisA one-class split columns required')
    seen=set();counts=Counter();categories=set()
    for row in rows:
        relative=safe_relative(row['image']).as_posix()
        if relative in seen or relative not in available_images:raise ValueError('Unique original image and disjoint split membership required')
        seen.add(relative);categories.add(row['object'])
        if relative.split('/')[0]!=row['object']:raise ValueError('Original category/image identity differs')
        if row['split'] not in ('train','test') or row['label'] not in ('normal','anomaly'):
            raise ValueError('Original one-class split/label required')
        if row['split']=='train' and row['label']!='normal':raise ValueError('Anomalies cannot enter the original one-class training split')
        if row['label']=='anomaly' and (not row['mask'] or safe_relative(row['mask']).as_posix() not in available_files):
            raise ValueError('All original anomalous image masks required')
        counts[(row['object'],row['split'],row['label'])]+=1
    if seen!=available_images:raise ValueError('Every original VisA image must belong to its native split')
    return counts,categories


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for receipt in config['source_receipts']:
        if digest(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen full image acquisition binding changed')
    root=(ROOT/config['asset_root']).resolve()
    if root.exists():raise FileExistsError('Fresh configured image asset root required; never overwrite raw data')
    if shutil.disk_usage(root.parent).free<config['minimum_free_disk_bytes']:raise OSError('Insufficient configured image asset space')
    root.mkdir(parents=True)
    state_root=ROOT/config['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),phase='starting',active_pid=None,downloads=[],
               captured_utc=datetime.now(timezone.utc).isoformat(),complete=False)
    def update(values):
        state.update(values);state['captured_utc']=datetime.now(timezone.utc).isoformat();json_write(state_root/'status.json',state)
    try:
        paths={}
        for entry in config['sources']:paths[entry['id']]=download(config,entry,state,update)
        update(dict(phase='auditing_and_extracting_every_original_archive_member',active_pid=None))
        original=ROOT/config['author_directory'];source_files=extract_author(paths['visa_author_source'],original,config['author_commit'])
        raw=ROOT/config['raw_directory'];raw_files=extract_full_tar(paths['visa_raw'],raw)
        # Archive releases may wrap a single VisA/ prefix. Bind the actual root
        # explicitly in the receipt rather than renaming any original files.
        roots=[raw]+[p for p in raw.iterdir() if p.is_dir()]
        candidates=[p for p in roots if (p/'candle/Data/Images').is_dir()]
        if len(candidates)!=1:raise ValueError('Exactly one complete original VisA root required')
        data=candidates[0]
        images={p.relative_to(data).as_posix() for p in data.glob('*/Data/Images/*/*') if p.is_file()}
        files={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
        split=original/'split_csv/1cls.csv'
        with split.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        counts,categories=validate_split_rows(rows,images,files)
        from PIL import Image
        image_manifest=[]
        for row in rows:
            path=data/row['image']
            with Image.open(path) as image:
                geometry=image.size;mode=image.mode;image.verify()
            with Image.open(path) as image:image.load()
            if row['mask']:
                mask=data/row['mask']
                if not mask.is_file():raise ValueError('Original mask missing')
                with Image.open(mask) as image:
                    if image.size!=geometry:raise ValueError('Original pixel mask geometry differs')
                    image.verify()
                with Image.open(mask) as image:image.load()
            image_manifest.append(dict(row,width=geometry[0],height=geometry[1],mode=mode,
                                      image_sha256=digest(path),mask_sha256=digest(data/row['mask']) if row['mask'] else None))
        if len(rows)!=10821 or len(categories)!=12 or sum(r['label']=='normal' for r in rows)!=9621 or sum(r['label']=='anomaly' for r in rows)!=1200:
            raise ValueError('Every original published VisA category/image/label count required')
        report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=args.config,
            configuration_sha256=digest(ROOT/args.config),complete_raw_archive_sha256=digest(paths['visa_raw']),
            exact_raw_archive_bytes=paths['visa_raw'].stat().st_size,complete_author_archive_sha256=digest(paths['visa_author_source']),
            author_repository=config['author_repository'],author_commit=config['author_commit'],native_raw_root=str(data),
            original_split_csv=str(split),original_split_sha256=digest(split),original_images=len(images),categories=sorted(categories),
            normal_images=9621,anomalous_images=1200,all_images_and_masks_decoded_and_verified=True,
            grouped_original_train_test_image_identities_disjoint=True,raw_files=raw_files,author_files=source_files,
            native_split_image_manifest=image_manifest,per_category_split_counts=[dict(category=c,split=s,label=l,images=n) for (c,s,l),n in sorted(counts.items())],
            source_receipts=config['source_receipts'],publisher_full_file_checksum_available=False,
            preprocessing_performed=False,benchmark_performance=False,full_Flow_Mismatching_training_completed=False,
            boundary=config['boundary'])
        output=ROOT/config['output_directory'];output.mkdir(parents=True,exist_ok=False)
        json_write(output/'full_original_visa_data_audit.json',report)
        update(dict(phase='completed_full_original_VisA_raw_data_and_split_audit',complete=True,report=str(output/'full_original_visa_data_audit.json')))
        print(json.dumps(dict(passed=True,original_images=len(images),original_categories=len(categories))))
    except BaseException as error:
        update(dict(phase='failed_preserved',complete=False,error=str(error)))
        raise


if __name__=='__main__':main()
