"""Acquire immutable related optimizer sources; never claim target-author identity."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import PurePosixPath
import subprocess
import zipfile

import psutil

from scripts.flow_matching.acquire_flow_mismatching_native_guidance_sources_v1 import verify_pdf
from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import ROOT,digest,download,json_write,safe_relative


def verified_extract(archive,target,git_root,commit,repository_name):
    target.mkdir(parents=True,exist_ok=False)
    tree=subprocess.check_output(['git','-C',str(git_root),'ls-tree','-r',commit]).decode('utf-8')
    blobs={}
    for line in tree.splitlines():
        head,name=line.split('\t',1);mode,kind,identity=head.split()
        if kind!='blob' or mode not in ('100644','100755'):raise ValueError('Only original regular Git files accepted')
        blobs[name]=identity
    receipts=[];seen=set()
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:raise ValueError('Complete optimizer ZIP CRC failed')
        for member in source.infolist():
            name=safe_relative(member.filename)
            if name.parts[0]!=repository_name+'-'+commit:raise ValueError('Immutable optimizer repository wrapper differs')
            if member.is_dir():continue
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            if relative in seen or relative not in blobs:raise ValueError('Duplicate or unregistered original Git member')
            payload=source.read(member)
            blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if blob!=blobs[relative] or payload!=subprocess.check_output(['git','-C',str(git_root),'cat-file','blob',blobs[relative]]):
                raise ValueError('Complete optimizer ZIP/Git bytes differ')
            path=(target/relative).resolve()
            if not path.is_relative_to(target.resolve()):raise ValueError('Unsafe original optimizer member')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(payload)
            receipts.append(dict(path=relative,bytes=len(payload),sha256=digest(path),verified_git_blob=blob));seen.add(relative)
    if seen!=set(blobs):raise ValueError('Incomplete original optimizer Git tree')
    return receipts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    state_root=ROOT/cfg['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),complete=False,downloads=[])
    def update(value):
        state.update(value);state['captured_utc']=datetime.now(timezone.utc).isoformat();json_write(state_root/'status.json',state)
    try:
        downloaded={s['id']:download(cfg,s,state,update) for s in cfg['sources']}
        repositories=[]
        for repo in cfg['repositories']:
            update(dict(phase='verifying_complete_original_optimizer_git_tree',repository=repo['url']))
            git_root=ROOT/repo['git_root']
            if git_root.exists():raise FileExistsError('Preserve prior optimizer Git attempt')
            subprocess.run(['git','init','--bare',str(git_root)],check=True)
            subprocess.run(['git','-C',str(git_root),'fetch','--depth','1',repo['url'],repo['commit']],check=True)
            subprocess.run(['git','-C',str(git_root),'fsck','--full'],check=True)
            files=verified_extract(downloaded[repo['source_id']],ROOT/repo['code_root'],git_root,repo['commit'],repo['name'])
            repositories.append(dict(repository=repo['url'],commit=repo['commit'],git_fsck_passed=True,
                original_code_root=str((ROOT/repo['code_root']).resolve()),all_original_git_files_verified=len(files),files=files))
        papers=[verify_pdf(downloaded[name]) for name in cfg['paper_source_ids']]
        report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
            configuration_sha256=digest(ROOT/args.config),downloads=state['downloads'],repositories=repositories,original_papers=papers,
            Flow_Mismatching_original_optimizer_identity_verified=False,
            full_original_training_executed=False,new_original_training_seeds=0,benchmark_performance=False,
            boundary=cfg['boundary'])
        output=ROOT/cfg['output_directory'];output.mkdir(parents=True,exist_ok=False)
        json_write(output/'full_immutable_optimizer_source_audit.json',report)
        update(dict(phase='completed_all_optimizer_source_and_paper_audits',complete=True,active_pid=None))
        print(json.dumps(dict(passed=True,repositories=len(repositories),original_papers=len(papers))))
    except Exception as error:
        update(dict(phase='failed_preserved_optimizer_acquisition_attempt',complete=False,error=repr(error)));raise


if __name__=='__main__':main()
