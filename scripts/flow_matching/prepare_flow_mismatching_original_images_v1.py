"""Audit-gated, read-only raw image preprocessing into fresh derived arrays.

128px RGB, bilinear image resizing, nearest mask resizing, and pooled per-RGB
population normalization are explicitly LOCAL frozen choices where author
details are unavailable. This procedure cannot certify paper fidelity.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import psutil

from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import digest, json_write, safe_relative

ROOT=Path(__file__).resolve().parents[2]


def unified_rows(audit, dataset):
    if not audit.get('passed'):
        raise ValueError('A successful full raw acquisition audit is required')
    if dataset=='VisA':
        if not audit.get('all_images_and_masks_decoded_and_verified'):
            raise ValueError('Every original VisA image and mask must be verified')
        rows=[dict(r,category=r['object'],label={'normal':0,'anomaly':1}[r['label']])
              for r in audit['native_split_image_manifest']]
    elif dataset=='MVTec-AD':
        if not audit.get('all_original_images_and_masks_decoded_and_verified') or not audit.get('maintained_anomalib_full_file_checksum_verified'):
            raise ValueError('Complete MVTec decode and full archive checksum required')
        rows=[dict(r) for r in audit['native_image_manifest']]
    else:raise ValueError('Only the original registered image datasets are supported')
    paths=[r['image'] for r in rows]
    if len(paths)!=len(set(paths)):
        raise ValueError('Native original image identity must be unique')
    for row in rows:
        safe_relative(row['image'])
        if row['mask']:safe_relative(row['mask'])
        if row['split'] not in ('train','test') or row['label'] not in (0,1):
            raise ValueError('Native split and binary label required')
        if row['split']=='train' and row['label']!=0:
            raise ValueError('Anomaly cannot enter normal-only training normalization')
        if bool(row['mask'])!=bool(row['label']):
            raise ValueError('Every anomaly requires its native mask; normal masks are implicit zero')
    return rows


def source_path(raw_root, relative):
    path=(raw_root/str(safe_relative(relative))).resolve()
    if not path.is_relative_to(raw_root.resolve()) or not path.is_file():
        raise ValueError('Verified raw member is absent or resolves outside original source root')
    return path


def resized_rgb(raw_root, row, resolution):
    path=source_path(raw_root,row['image'])
    if digest(path)!=row['image_sha256']:
        raise ValueError('Raw image SHA256 differs from completed original audit')
    with Image.open(path) as image:
        array=np.asarray(image.convert('RGB').resize((resolution,resolution),Image.Resampling.BILINEAR),dtype=np.float64)/255.
    return array.transpose(2,0,1)


def training_statistics(raw_root, rows, resolution):
    sums=np.zeros(3,dtype=np.float64);squares=sums.copy();pixels=0;images=0
    for row in rows:
        if row['split']!='train':continue
        if row['label']!=0:raise ValueError('Statistics may only use normal training images')
        x=resized_rgb(raw_root,row,resolution)
        sums+=x.sum(axis=(1,2));squares+=(x*x).sum(axis=(1,2));pixels+=resolution**2;images+=1
    if not pixels:raise ValueError('Original normal training images are required')
    mean=sums/pixels;variance=np.maximum(squares/pixels-mean*mean,0)
    std=np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Nondegenerate finite RGB training statistics required')
    return dict(mean=mean.tolist(),std=std.tolist(),normal_training_images=images,pixels_per_channel=pixels,
                estimator='population variance over all resized normal training pixels per RGB channel',
                test_images_used=0)


def prepare(root, config, update):
    audit_path=root/config['full_raw_audit']
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    rows=unified_rows(audit,config['dataset']);categories=sorted(set(r['category'] for r in rows))
    counts={s:sum(r['split']==s for r in rows) for s in ['train','test']}
    if len(rows)!=config['expected_images'] or len(categories)!=config['expected_categories'] or counts!=config['expected_native_splits']:
        raise ValueError('Every original image/category and exact native train/test split is required')
    raw=Path(audit[config['raw_root_field']]).resolve()
    target=(root/config['derived_directory']).resolve()
    if target.is_relative_to(raw) or raw.is_relative_to(target):
        raise ValueError('Derived destination must be separate from preserved raw source')
    if target.exists():raise FileExistsError('Preserve any prior derived attempt; do not overwrite')
    if shutil.disk_usage(target.parent if target.parent.exists() else root/config['asset_root']).free<config['minimum_free_disk_bytes']:
        raise ValueError('Original full derived arrays require registered free disk capacity')
    target.mkdir(parents=True,exist_ok=False)
    update(dict(phase='normal_training_only_statistics',raw_root=str(raw),derived_directory=str(target)))
    statistics=training_statistics(raw,rows,config['resolution'])
    mean=np.array(statistics['mean'])[:,None,None];std=np.array(statistics['std'])[:,None,None]
    artifacts=[];manifest=[];category_index={c:i for i,c in enumerate(categories)}
    for split in ['train','test']:
        subset=[r for r in rows if r['split']==split];count=len(subset);size=config['resolution']
        paths={'images':target/(split+'_images.npy'),'masks':target/(split+'_masks.npy'),
               'labels':target/(split+'_labels.npy'),'categories':target/(split+'_categories.npy')}
        arrays={'images':np.lib.format.open_memmap(paths['images'],mode='w+',dtype=np.float32,shape=(count,3,size,size)),
                'masks':np.lib.format.open_memmap(paths['masks'],mode='w+',dtype=np.uint8,shape=(count,size,size)),
                'labels':np.lib.format.open_memmap(paths['labels'],mode='w+',dtype=np.int64,shape=(count,)),
                'categories':np.lib.format.open_memmap(paths['categories'],mode='w+',dtype=np.int64,shape=(count,))}
        for i,row in enumerate(subset):
            x=(resized_rgb(raw,row,size)-mean)/std
            if not np.isfinite(x).all():raise ValueError('Normalized complete image must be finite')
            arrays['images'][i]=x
            mask=np.zeros((size,size),dtype=np.uint8)
            if row['mask']:
                path=source_path(raw,row['mask'])
                if digest(path)!=row['mask_sha256']:raise ValueError('Original raw mask SHA256 differs')
                with Image.open(path) as image:
                    mask=(np.asarray(image.convert('L').resize((size,size),Image.Resampling.NEAREST))>0).astype(np.uint8)
            arrays['masks'][i]=mask;arrays['labels'][i]=row['label'];arrays['categories'][i]=category_index[row['category']]
            manifest.append(dict(row,derived_split=split,derived_row=i,category_index=category_index[row['category']]))
            if i%100==0:update(dict(phase='writing_full_native_'+split,completed_images=i,total_images=count))
        for array in arrays.values():array.flush()
        del array,arrays
        for name,path in paths.items():
            reopened=np.load(path,mmap_mode='r')
            if len(reopened)!=count:raise ValueError('Full derived array count changed')
            artifacts.append(dict(split=split,field=name,path=str(path),shape=list(reopened.shape),dtype=str(reopened.dtype),bytes=path.stat().st_size,sha256=digest(path)))
            del reopened
    result=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),dataset=config['dataset'],
        complete_original_images=len(rows),original_categories=categories,category_index=category_index,native_splits=counts,
        full_raw_audit_path=str(audit_path),full_raw_audit_sha256=digest(audit_path),normalization=statistics,
        all_original_raw_images_and_masks_SHA_reverified=True,raw_files_written_or_overwritten=0,
        artifacts=artifacts,native_to_derived_image_manifest=manifest,preprocessing_config=config,
        full_original_training_executed=False,paper_preprocessing_equivalence_certified=False,strict_TAB_equivalent=False,
        boundary='Complete derived local image preprocessing, with disclosed author-unpinned interpolation/normalization/mask choices; no measured model performance or exact paper fidelity claim.')
    json_write(target/'full_derived_image_preprocessing.json',result)
    return target/'full_derived_image_preprocessing.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    state_root=ROOT/config['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),configuration=args.config,
        configuration_sha256=digest(ROOT/args.config),complete=False)
    def update(values):
        state.update(values);state['captured_utc']=datetime.now(timezone.utc).isoformat();json_write(state_root/'status.json',state)
    try:
        for receipt in config['source_receipts']:
            if digest(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen preprocessing source changed')
        while not (ROOT/config['full_raw_audit']).is_file():
            acquisition=json.loads((ROOT/config['acquisition_state']).read_text(encoding='utf-8'))
            if 'failed' in acquisition['phase']:raise ValueError('Original acquisition failed; preserved raw attempt requires audit')
            process=psutil.Process(acquisition['pid'])
            if abs(process.create_time()-acquisition['create_time'])>.001:
                raise ValueError('Original acquisition controller identity changed')
            update(dict(phase='waiting_for_completed_full_raw_audit',actual_original_acquisition_pid=process.pid,
                        actual_original_acquisition_create_time=process.create_time(),preprocessing_started=False))
            time.sleep(config['poll_seconds'])
        while psutil.virtual_memory().available<config['minimum_free_physical_bytes']:
            update(dict(phase='waiting_for_registered_preprocessing_memory',preprocessing_started=False));time.sleep(config['poll_seconds'])
        result=prepare(ROOT,config,update)
        update(dict(phase='completed_full_original_derived_image_preprocessing',report=str(result),complete=True,
                    preprocessing_started=True,full_original_training_executed=False,paper_equivalence_certified=False))
        print(json.dumps(dict(passed=True,report=str(result))))
    except Exception as error:
        update(dict(phase='failed_preserved_preprocessing_attempt',error=repr(error),complete=False))
        raise


if __name__=='__main__':main()
