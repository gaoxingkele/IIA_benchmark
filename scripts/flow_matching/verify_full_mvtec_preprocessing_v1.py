"""Verify every complete derived MVTec-AD row against its preserved raw image/mask."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.flow_matching.verify_visa_archive_and_split_v1 import file_sha

ROOT=Path(__file__).resolve().parents[2]


def check_row(image,mask,label,category,expected_image,expected_mask,expected_label,expected_category):
    if not np.isfinite(image).all() or not np.array_equal(image,expected_image):
        raise ValueError('Complete derived image differs from frozen raw transformation')
    if not np.array_equal(mask,expected_mask) or not np.isin(mask,[0,1]).all():
        raise ValueError('Complete derived mask differs from original anomaly mask')
    if int(label)!=expected_label or int(category)!=expected_category:
        raise ValueError('Complete native label/category binding differs')


def check_normalized_moments(sums,squares,pixels,tolerance):
    mean=sums/pixels;variance=squares/pixels-mean**2
    if not np.all(np.abs(mean)<=tolerance['absolute_mean']) or not np.all(np.abs(variance-1)<=tolerance['absolute_variance']):
        raise ValueError('Complete normal-training-only normalized moments differ')
    return dict(mean=mean.tolist(),population_variance=variance.tolist(),pixels_per_channel=pixels)


def verify(config):
    raw_audit=json.loads((ROOT/config['raw_audit']).read_text(encoding='utf-8'))
    independent=json.loads((ROOT/config['independent_raw_audit']).read_text(encoding='utf-8'))
    if not raw_audit['passed'] or not independent['passed'] or independent['full_raw_audit_sha256']!=file_sha(ROOT/config['raw_audit']):
        raise ValueError('Independent complete raw archive audit is required')
    prep_config=json.loads((ROOT/config['preprocessing_configuration']).read_text(encoding='utf-8'))
    for receipt in prep_config['source_receipts']:
        if file_sha(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Frozen preprocessing source changed')
    derived_root=(ROOT/prep_config['derived_directory']).resolve()
    report_path=derived_root/'full_derived_image_preprocessing.json'
    derived=json.loads(report_path.read_text(encoding='utf-8'))
    if not derived['passed'] or derived['full_raw_audit_sha256']!=file_sha(ROOT/config['raw_audit']) or derived['preprocessing_config']!=prep_config:
        raise ValueError('Complete derived report/configuration/raw binding differs')
    original=raw_audit['native_image_manifest']
    converted=[dict(r) for r in original]
    categories={c:i for i,c in enumerate(sorted({r['category'] for r in original}))}
    expected_manifest=[]
    for split in ['train','test']:
        for i,row in enumerate(r for r in converted if r['split']==split):
            expected_manifest.append(dict(row,derived_split=split,derived_row=i,category_index=categories[row['category']]))
    if expected_manifest!=derived['native_to_derived_image_manifest'] or derived['category_index']!=categories:
        raise ValueError('All original image identities/split order/category indices differ')
    normalization=derived['normalization']
    if normalization['test_images_used']!=0 or normalization['normal_training_images']!=3629:
        raise ValueError('Training-only normalization boundary differs')
    mean=np.asarray(normalization['mean'],dtype=np.float64)[:,None,None]
    std=np.asarray(normalization['std'],dtype=np.float64)[:,None,None]
    if mean.shape!=(3,1,1) or std.shape!=(3,1,1) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Frozen complete RGB statistics invalid')
    raw=Path(raw_audit[prep_config['raw_root_field']]).resolve();resolution=prep_config['resolution']
    sums=np.zeros(3);squares=np.zeros(3);pixels=0;verified=0;anomalies=0
    receipts=[]
    expected_artifacts={(split,field) for split in ['train','test'] for field in ['images','masks','labels','categories']}
    if len(derived['artifacts'])!=8 or {(r['split'],r['field']) for r in derived['artifacts']}!=expected_artifacts:
        raise ValueError('All eight full native arrays are required')
    for split in ['train','test']:
        rows=[r for r in expected_manifest if r['split']==split]
        arrays={}
        for receipt in (r for r in derived['artifacts'] if r['split']==split):
            path=(derived_root/(split+'_'+receipt['field']+'.npy')).resolve()
            if path!=Path(receipt['path']).resolve() or path.stat().st_size!=receipt['bytes'] or file_sha(path)!=receipt['sha256']:
                raise ValueError('Full derived array SHA256/size/path changed')
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if list(array.shape)!=receipt['shape'] or str(array.dtype)!=receipt['dtype'] or len(array)!=len(rows):
                raise ValueError('Full derived array shape or dtype differs')
            arrays[receipt['field']]=array;receipts.append(receipt)
        for i,row in enumerate(rows):
            image_path=(raw/row['image']).resolve()
            if not image_path.is_relative_to(raw) or file_sha(image_path)!=row['image_sha256']:
                raise ValueError('Preserved complete raw image changed')
            with Image.open(image_path) as image:
                values=np.asarray(image.convert('RGB').resize((resolution,resolution),Image.Resampling.BILINEAR),dtype=np.float64).transpose(2,0,1)/255.
            expected_image=((values-mean)/std).astype(np.float32)
            expected_mask=np.zeros((resolution,resolution),dtype=np.uint8)
            if row['mask']:
                mask_path=(raw/row['mask']).resolve()
                if not mask_path.is_relative_to(raw) or file_sha(mask_path)!=row['mask_sha256']:
                    raise ValueError('Preserved complete raw mask changed')
                with Image.open(mask_path) as mask:
                    expected_mask=(np.asarray(mask.convert('L').resize((resolution,resolution),Image.Resampling.NEAREST))>0).astype(np.uint8)
                anomalies+=1
            check_row(arrays['images'][i],arrays['masks'][i],arrays['labels'][i],arrays['categories'][i],
                expected_image,expected_mask,row['label'],row['category_index'])
            if split=='train':
                values=arrays['images'][i].astype(np.float64)
                sums+=values.sum(axis=(1,2));squares+=(values**2).sum(axis=(1,2));pixels+=resolution**2
            verified+=1
        del arrays
        print(json.dumps(dict(complete_verified_split=split,images=len(rows))),flush=True)
    moments=check_normalized_moments(sums,squares,pixels,config['registered_normalized_moment_tolerance'])
    if verified!=5354 or anomalies!=1258 or normalization['pixels_per_channel']!=pixels:
        raise ValueError('Complete original image/mask coverage differs')
    return dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        complete_original_images_individually_raw_to_derived_verified=verified,complete_original_masks_verified=anomalies,
        native_splits=derived['native_splits'],original_categories=len(categories),all_eight_array_hashes_and_shapes_verified=True,
        derived_report_sha256=file_sha(report_path),preprocessing_config_sha256=file_sha(ROOT/config['preprocessing_configuration']),
        full_raw_audit_sha256=file_sha(ROOT/config['raw_audit']),independent_raw_audit_sha256=file_sha(ROOT/config['independent_raw_audit']),
        all_native_labels_categories_masks_and_split_order_match=True,normalized_train_only_moments=moments,
        moment_tolerance=config['registered_normalized_moment_tolerance'],all_train_test_image_transformations_exact_float32_equal=True,
        complete_original_test_data_used_for_normalization=False,raw_files_written_or_overwritten=0,
        artifact_receipts=receipts,paper_preprocessing_equivalence_certified=False,full_original_training_executed=False,
        strict_TAB_equivalence_certified=False,new_benchmark_result_rows=0,
        boundary='Full verification of registered LOCAL 128px preprocessing, including every original row and training-only moments. Author-unspecified choices prevent original-paper preprocessing certification. No training or performance results.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'));output=ROOT/config['output_report']
    if output.exists():raise FileExistsError('Preserve prior full derived verification')
    report=verify(config);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,full_original_images=5354,full_original_anomaly_masks=1258)))


if __name__=='__main__':main()
