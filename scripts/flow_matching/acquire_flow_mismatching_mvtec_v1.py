"""Download original complete MVTec AD with the maintained Anomalib checksum."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile

import psutil

from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import ROOT,digest,safe_relative,json_write,download


def native_identity(relative,available):
    p=safe_relative(relative)
    if len(p.parts)!=4 or p.parts[1] not in ('train','test') or p.suffix.lower()!='.png':
        raise ValueError('Exact original MVTec image identity required')
    category,split,defect=p.parts[:3]
    if split=='train' and defect!='good':raise ValueError('Original MVTec training images must be defect-free')
    anomaly=defect!='good'
    mask=PurePosixPath(category,'ground_truth',defect,p.stem+'_mask.png').as_posix() if anomaly else ''
    if anomaly and mask not in available:raise ValueError('Every native MVTec anomalous test mask required')
    return dict(category=category,split=split,defect_type=defect,label=int(anomaly),image=p.as_posix(),mask=mask)


def extract(archive,target):
    target=target.resolve();target.mkdir(parents=True,exist_ok=False);seen=set();files=[]
    with tarfile.open(archive,mode='r|*') as tar:
        for member in tar:
            relative=safe_relative(member.name)
            if not (member.isdir() or member.isfile()) or member.issym() or member.islnk():
                raise ValueError('Original MVTec regular files/directories required')
            if str(relative)=='.' and member.isdir():continue
            if str(relative) in seen:raise ValueError('Duplicate original MVTec archive member')
            seen.add(str(relative));path=(target/str(relative)).resolve()
            if target not in path.parents:raise ValueError('Original archive escapes configured raw root')
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output:shutil.copyfileobj(tar.extractfile(member),output,1024**2)
            if path.stat().st_size!=member.size:raise ValueError('Original MVTec member is truncated')
            files.append(dict(path=relative.as_posix(),bytes=member.size,sha256=digest(path)))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for receipt in config['source_receipts']:
        if digest(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen original MVTec acquisition binding changed')
    root=(ROOT/config['asset_root']).resolve()
    if root.exists():raise FileExistsError('Fresh configured raw asset root required; preserve all original downloads')
    if shutil.disk_usage(root.parent).free<config['minimum_free_disk_bytes']:raise OSError('Insufficient full original archive space')
    root.mkdir(parents=True)
    state_root=ROOT/config['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),phase='starting',active_pid=None,downloads=[],complete=False)
    def update(values):
        state.update(values);state['captured_utc']=datetime.now(timezone.utc).isoformat();json_write(state_root/'status.json',state)
    try:
        archive=download(config,config['source'],state,update)
        if digest(archive)!=config['source']['expected_sha256']:
            raise ValueError('Exact complete maintained Anomalib MVTec SHA256 differs')
        update(dict(phase='auditing_complete_original_mvtec_archive',active_pid=None,
                    complete_original_archive_sha256_verified=True))
        raw=ROOT/config['raw_directory'];raw_files=extract(archive,raw)
        roots=[raw]+[p for p in raw.iterdir() if p.is_dir()]
        candidates=[p for p in roots if (p/'bottle/train/good').is_dir()]
        if len(candidates)!=1:raise ValueError('Exactly one complete original MVTec dataset root required')
        data=candidates[0];available={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
        images=sorted(p.relative_to(data).as_posix() for p in data.glob('*/train/*/*.png'))+sorted(p.relative_to(data).as_posix() for p in data.glob('*/test/*/*.png'))
        if len(images)!=config['expected_images'] or len(set(images))!=len(images):
            raise ValueError('All original published MVTec images required exactly once')
        from PIL import Image
        records=[];counts=Counter()
        for relative in images:
            record=native_identity(relative,available);path=data/relative
            with Image.open(path) as image:geometry=image.size;mode=image.mode;image.verify()
            with Image.open(path) as image:image.load()
            if record['mask']:
                mask=data/record['mask']
                with Image.open(mask) as image:
                    if image.size!=geometry:raise ValueError('Original MVTec pixel mask geometry differs')
                    image.verify()
                with Image.open(mask) as image:image.load()
            record.update(width=geometry[0],height=geometry[1],mode=mode,image_sha256=digest(path),
                          mask_sha256=digest(data/record['mask']) if record['mask'] else None)
            records.append(record);counts[(record['category'],record['split'],record['defect_type'])]+=1
        if set(r['category'] for r in records)!=set(config['expected_categories']):
            raise ValueError('Every original MVTec category required')
        masks={r['mask'] for r in records if r['mask']}
        if masks!={p for p in available if '/ground_truth/' in p and p.endswith('.png')}:
            raise ValueError('Every native original MVTec anomaly mask must align with exactly one test image')
        report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=args.config,
            configuration_sha256=digest(ROOT/args.config),complete_original_archive_sha256=digest(archive),
            maintained_anomalib_full_file_checksum_verified=True,checksum_reference=config['source']['checksum_reference'],
            original_archive_bytes=archive.stat().st_size,raw_root=str(data),original_images=len(records),
            original_categories=sorted(set(r['category'] for r in records)),original_anomaly_masks=len(masks),
            original_train_images=sum(r['split']=='train' for r in records),original_test_images=sum(r['split']=='test' for r in records),
            all_original_images_and_masks_decoded_and_verified=True,original_train_test_image_paths_disjoint=True,
            all_original_training_images_defect_free=True,raw_archive_member_manifest=raw_files,native_image_manifest=records,
            per_category_split_defect_counts=[dict(category=c,split=s,defect=d,images=n) for (c,s,d),n in sorted(counts.items())],
            source_receipts=config['source_receipts'],preprocessing_performed=False,benchmark_performance=False,
            full_Flow_Mismatching_training_completed=False,boundary=config['boundary'])
        output=ROOT/config['output_directory'];output.mkdir(parents=True,exist_ok=False)
        json_write(output/'full_original_mvtec_data_audit.json',report)
        update(dict(phase='completed_full_original_MVTec_raw_data_and_split_audit',complete=True,report=str(output/'full_original_mvtec_data_audit.json')))
        print(json.dumps(dict(passed=True,original_images=len(records),original_categories=len(config['expected_categories']))))
    except BaseException as error:
        update(dict(phase='failed_preserved',complete=False,error=str(error)))
        raise


if __name__=='__main__':main()
