"""Verify every pinned VisA author Git blob and the full original split metadata."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import io
import json
import subprocess
import zipfile

from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import ROOT,digest,safe_relative,json_write
from scripts.flow_matching.acquire_adopted_gan_sources_v1 import verify_blob


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    for item in config['source_receipts']:
        if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen image asset registration source changed')
    acquisition=json.loads((ROOT/config['visa_acquisition']).read_text(encoding='utf-8'))
    repo=ROOT/config['author_git_identity']
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip()
    if commit!=acquisition['author_commit']:raise ValueError('Frozen original VisA author commit differs')
    git('fsck','--strict','--full')
    entries={}
    for line in git('ls-tree','-r','-l',commit).decode().splitlines():
        identity,path=line.split('\t',1);mode,kind,sha,size=identity.split()
        if kind!='blob' or mode not in ('100644','100755'):raise ValueError('Original regular Git blobs required')
        entries[path]=dict(mode=mode,sha=sha,size=int(size))
    archive=ROOT/next(a['path'] for a in acquisition['sources'] if a['id']=='visa_author_source')
    receipts=[];found=set();split=None
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name=safe_relative(member.filename)
            if name.parts[0]!='spot-diff-'+commit:raise ValueError('Frozen original commit archive root differs')
            if member.is_dir():continue
            path='/'.join(name.parts[1:])
            if path in found or path not in entries:raise ValueError('Exact complete original author archive required')
            found.add(path);payload=zipped.read(member);entry=entries[path]
            sha256=verify_blob(payload,entry)
            if payload!=git('cat-file','blob',entry['sha']):raise ValueError('Original author archive/Git blob bytes differ')
            receipts.append(dict(path=path,git_blob_sha1=entry['sha'],sha256=sha256,bytes=entry['size']))
            if path=='split_csv/1cls.csv':split=payload
    if found!=set(entries) or split is None:raise ValueError('All frozen author files and full original split required')
    rows=list(csv.DictReader(io.StringIO(split.decode('utf-8-sig'))));seen=set();counts=Counter()
    if len(rows)!=10821 or set(rows[0])!={'object','split','label','image','mask'}:raise ValueError('Full original one-class split schema required')
    for row in rows:
        image=safe_relative(row['image']).as_posix()
        if image in seen or image.split('/')[0]!=row['object']:raise ValueError('Unique original category/image split identities required')
        seen.add(image)
        if row['split'] not in ('train','test') or row['label'] not in ('normal','anomaly'):raise ValueError('Original split labels required')
        if row['split']=='train' and row['label']!='normal':raise ValueError('Original one-class training must be normal-only')
        if row['label']=='anomaly' and not row['mask']:raise ValueError('All original anomaly mask identities required')
        if row['mask']:safe_relative(row['mask'])
        counts[(row['object'],row['split'],row['label'])]+=1
    if len({r['object'] for r in rows})!=12 or sum(r['label']=='normal' for r in rows)!=9621 or sum(r['label']=='anomaly' for r in rows)!=1200:
        raise ValueError('Complete original category/label inventory required')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),passed=True,configuration=args.config,
        configuration_sha256=digest(ROOT/args.config),author_repository=acquisition['author_repository'],author_commit=commit,
        original_author_archive_sha256=digest(archive),all_author_git_blobs_and_zip_crc_verified=True,
        every_original_author_file_verified=receipts,complete_original_one_class_split_metadata_verified=True,
        original_image_identities=len(rows),original_categories=12,original_normal_image_identities=9621,original_anomaly_image_identities=1200,
        all_original_train_test_image_identities_unique=True,
        per_category_split_counts=[dict(category=c,split=s,label=l,images=n) for (c,s,l),n in sorted(counts.items())],
        raw_original_images_present_and_decoded=False,full_original_data_downloads_complete=False,
        full_Flow_Mismatching_training_completed=False,benchmark_performance=False,
        boundary='Complete pinned original author source and native one-class CSV identity audit only. All10821 original image identities across12categories accounted for and all Git blobs verified. Raw image/mask downloads and full decode audits remain running; no fabricated image presence, completed preprocessing or training/performance claim.')
    target=ROOT/config['output_directory'];target.mkdir(parents=True,exist_ok=False)
    json_write(target/'original_author_and_split_metadata_audit.json',report)
    print(json.dumps(dict(original_author_files=len(receipts),original_image_identities=len(rows),raw_data_complete=False)))


if __name__=='__main__':main()
