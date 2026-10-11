"""Preserve and verify original FlowGrad source plus public method papers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import zipfile

import psutil
from pypdf import PdfReader

from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import ROOT,digest,download,json_write,safe_relative


def extract_verified_author(archive,target,git_root,commit):
    target.mkdir(parents=True,exist_ok=False);receipts=[];seen=set()
    expected_root='FlowGrad-'+commit
    tree=subprocess.check_output(['git','-C',str(git_root),'ls-tree','-r',commit]).decode('utf-8')
    blobs={}
    for line in tree.splitlines():
        head,name=line.split('\t',1);mode,kind,identity=head.split()
        if kind!='blob' or mode not in ('100644','100755'):
            raise ValueError('Only original regular author Git files supported')
        blobs[name]=identity
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Original author ZIP CRC failure')
        for member in z.infolist():
            name=safe_relative(member.filename)
            if name.parts[0]!=expected_root:raise ValueError('Original immutable author archive root differs')
            if member.is_dir():continue
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            if relative in seen or relative not in blobs:raise ValueError('Duplicate or non-original Git file in author archive')
            payload=z.read(member);blob_sha=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            git_bytes=subprocess.check_output(['git','-C',str(git_root),'cat-file','blob',blobs[relative]])
            if blob_sha!=blobs[relative] or payload!=git_bytes:raise ValueError('Original complete ZIP/Git blob byte mismatch')
            path=(target/relative).resolve()
            if not path.is_relative_to(target.resolve()):raise ValueError('Unsafe author source destination')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(payload)
            receipts.append(dict(path=relative,bytes=len(payload),sha256=digest(path),verified_git_blob=blob_sha));seen.add(relative)
    if seen!=set(blobs):raise ValueError('Every original author Git file must be present in full archive')
    return receipts


def verify_pdf(path):
    if not path.read_bytes()[:5]==b'%PDF-':raise ValueError('Public paper download is not a PDF')
    reader=PdfReader(path,strict=True)
    text=[p.extract_text() or '' for p in reader.pages]
    if not text or any(not t.strip() for t in text):raise ValueError('Every original paper page must parse nonempty content')
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,pages=len(text),all_pages_parse=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    state_root=ROOT/cfg['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),complete=False,downloads=[])
    def update(value):
        state.update(value);state['captured_utc']=datetime.now(timezone.utc).isoformat();json_write(state_root/'status.json',state)
    try:
        paths={}
        for source in cfg['sources']:paths[source['id']]=download(cfg,source,state,update)
        git_root=ROOT/cfg['asset_root']/'flowgrad_git_identity'
        if git_root.exists():raise FileExistsError('Preserve any earlier author Git checkout')
        update(dict(phase='verifying_full_original_author_git_tree'))
        subprocess.run(['git','clone','--no-checkout','--depth','1',cfg['flowgrad_author_repository'],str(git_root)],check=True)
        head=subprocess.check_output(['git','-C',str(git_root),'rev-parse','HEAD']).decode().strip()
        if head!=cfg['flowgrad_author_commit']:raise ValueError('Author HEAD moved; preserve downloaded frozen source and fetch immutable revision separately')
        subprocess.run(['git','-C',str(git_root),'fsck','--full'],check=True)
        receipts=extract_verified_author(paths['flowgrad_author_zip'],ROOT/cfg['flowgrad_author_code_root'],git_root,head)
        pdfs=[verify_pdf(path) for name,path in paths.items() if name!='flowgrad_author_zip']
        out=ROOT/cfg['output_directory'];out.mkdir(parents=True,exist_ok=False)
        result=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
                    configuration_sha256=digest(ROOT/args.config),downloads=state['downloads'],
                    flowgrad_author_commit=head,flowgrad_author_git_fsck_passed=True,all_original_author_git_files_verified=len(receipts),
                    author_original_root=str((ROOT/cfg['flowgrad_author_code_root']).resolve()),author_files=receipts,original_papers=pdfs,
                    original_FlowGrad_author_code_available=True,Flow_Mismatching_original_adaptation_verified=False,
                    benchmark_performance=False,new_training_seed_slots=0,
                    boundary='Full original FlowGrad immutable author source and original public papers acquired; original D-Flow+FlowGrad anomaly adaptation, trained checkpoint and performance remain unverified.')
        json_write(out/'full_original_guidance_material_audit.json',result)
        update(dict(phase='completed_original_guidance_material_audit',complete=True,active_pid=None,report=str(out/'full_original_guidance_material_audit.json')))
        print(json.dumps(dict(passed=True,author_files=len(receipts),papers=len(pdfs))))
    except Exception as error:
        update(dict(phase='failed_preserved_guidance_acquisition_attempt',complete=False,error=repr(error)))
        raise


if __name__=='__main__':main()
