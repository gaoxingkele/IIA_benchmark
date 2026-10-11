"""Read-only independent full archive hash, extracted files and native identities."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.flow_matching.verify_visa_archive_and_split_v1 import file_sha,bind_member

ROOT=Path(__file__).resolve().parents[2]


def identity_checks(rows,available):
    seen=set();masks=set();counts=Counter();by_sha=defaultdict(list)
    for row in rows:
        image=row['image'];p=Path(image);parts=p.parts
        if len(parts)!=4 or parts[1] not in ('train','test') or p.suffix!='.png' or image in seen:
            raise ValueError('Native original image identity is invalid or duplicated')
        seen.add(image);category,split,defect,_=parts;label=int(defect!='good')
        expected_mask=f'{category}/ground_truth/{defect}/{p.stem}_mask.png' if label else ''
        if split=='train' and label:raise ValueError('Original training must be normal-only')
        if (row['category'],row['split'],row['defect_type'],row['label'],row['mask'])!=(category,split,defect,label,expected_mask):
            raise ValueError('Native image directory, label and mask binding differs')
        if image not in available or (expected_mask and expected_mask not in available):
            raise ValueError('Every native image/mask is required')
        if expected_mask in masks:raise ValueError('Native anomaly mask is reused')
        if expected_mask:masks.add(expected_mask)
        counts[(category,split,defect)]+=1
        by_sha[row['image_sha256']].append(dict(image=image,split=split,category=category,label=label))
    actual_images={s for s in available if len(Path(s).parts)==4 and Path(s).parts[1] in ('train','test') and s.endswith('.png')}
    actual_masks={s for s in available if '/ground_truth/' in s and s.endswith('.png')}
    if seen!=actual_images or masks!=actual_masks:raise ValueError('Full native original image/mask coverage differs')
    duplicates=[dict(sha256=sha,images=items) for sha,items in sorted(by_sha.items()) if len(items)>1]
    cross=[d for d in duplicates if len({r['split'] for r in d['images']})>1]
    return dict(per_category_split_defect_counts=[dict(category=c,split=s,defect=d,images=n) for (c,s,d),n in sorted(counts.items())],
        duplicate_encoded_file_content_groups=duplicates,cross_split_identical_encoded_file_content_groups=cross,
        cross_split_identical_encoded_file_content_images=sum(len(d['images']) for d in cross))


def verify(config):
    acquisition_path=ROOT/config['acquisition_configuration']
    acquisition=json.loads(acquisition_path.read_text(encoding='utf-8'))
    audit_path=ROOT/config['full_raw_audit'];audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or not audit['all_original_images_and_masks_decoded_and_verified']:
        raise ValueError('Complete original image/mask decode required')
    if audit['configuration_sha256']!=file_sha(acquisition_path):raise ValueError('Frozen acquisition changed')
    for receipt in acquisition['source_receipts']:
        if file_sha(ROOT/receipt['path'])!=receipt['sha256']:raise ValueError('Original source receipt changed')
    archive=ROOT/acquisition['source']['path'];archive_sha=file_sha(archive)
    if archive_sha!=acquisition['source']['expected_sha256'] or archive_sha!=audit['complete_original_archive_sha256']:
        raise ValueError('Complete maintained Anomalib archive checksum differs')
    if archive.stat().st_size!=acquisition['source']['expected_bytes'] or archive.stat().st_size!=audit['original_archive_bytes']:
        raise ValueError('Complete original archive byte length differs')
    raw=(ROOT/acquisition['raw_directory']).resolve();native=Path(audit['raw_root']).resolve()
    records=audit['raw_archive_member_manifest'];expected={r['path']:r for r in records}
    if len(expected)!=len(records):raise ValueError('Duplicate original raw archive member')
    actual={p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}
    if actual!=set(expected):raise ValueError('Complete extracted raw file coverage differs')
    for record in records:
        _,path=bind_member(record['path'],raw)
        if path.stat().st_size!=record['bytes'] or file_sha(path)!=record['sha256']:
            raise ValueError('Original extracted raw file content differs')
    if not native.is_relative_to(raw):raise ValueError('Native data root escapes extraction')
    available={p.relative_to(native).as_posix() for p in native.rglob('*') if p.is_file()}
    rows=audit['native_image_manifest'];identities=identity_checks(rows,available)
    if identities['per_category_split_defect_counts']!=audit['per_category_split_defect_counts']:
        raise ValueError('All original per-category split/defect counts differ')
    if len(rows)!=5354 or len({r['category'] for r in rows})!=15 or sum(bool(r['mask']) for r in rows)!=1258:
        raise ValueError('Complete original image/category/mask counts differ')
    splits={s:sum(r['split']==s for r in rows) for s in ['train','test']}
    if splits!={'train':3629,'test':1725}:raise ValueError('Full original native split counts differ')
    for row in rows:
        for field,sha_field in [('image','image_sha256'),('mask','mask_sha256')]:
            if not row[field]:continue
            relative=(native/row[field]).relative_to(raw).as_posix()
            if expected[relative]['sha256']!=row[sha_field]:raise ValueError('Original image/mask manifest hash differs')
    return dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),
        complete_original_archive_sha256=archive_sha,maintained_anomalib_archive_checksum_independently_verified=True,
        full_raw_audit_sha256=file_sha(audit_path),all_original_extracted_file_hashes_sizes_and_coverage_reverified=True,
        independently_reverified_original_raw_files=len(records),original_images=len(rows),original_categories=15,
        original_anomaly_masks=1258,native_splits=splits,resolved_raw_directory=str(native),
        all_original_image_masks_labels_categories_and_native_splits_bound=True,raw_files_written_or_overwritten=0,
        independent_PIL_decode_repeated=False,original_full_PIL_decode_and_mask_geometry_passed=True,
        decoded_pixel_or_near_duplicate_similarity_certified=False,
        full_original_training_executed=False,new_benchmark_result_rows=0,strict_TAB_equivalence_certified=False,**identities)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'));output=ROOT/config['output_report']
    if output.exists():raise FileExistsError('Preserve prior complete verification')
    report=verify(config);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}))


if __name__=='__main__':main()
