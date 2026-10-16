"""Verify the complete pinned author repository and acquire exact replay materials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG='configs/acquisition/fm_TrajectoryNet_author_checkpoint_materials.v1.json'


def acquire(c):
    original=ROOT/c['source_root']
    sources=[]
    for entry in c['expected_original_blobs']:
        p=original/entry['path'];content=p.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if len(content)!=entry['size'] or blob!=entry['sha']:
            raise ValueError('Pinned original author file mismatch: '+entry['path'])
        sources.append(dict(path=entry['path'],bytes=len(content),git_blob_sha1=blob,sha256=sha(p)))
    out=Path(c['output_directory']);out.mkdir(parents=True,exist_ok=False)
    state=ROOT/c['state_root'];state.mkdir(parents=True,exist_ok=True);transfers={}
    for kind in ['wheel','paper']:
        item=c[kind];target=out/item['filename']
        cmd=[c['aria2c'],'--all-proxy='+c['proxy'],'--allow-overwrite=false','--auto-file-renaming=false',
             '--continue=false','--max-tries=3','--connect-timeout=20','--timeout=30','--retry-wait=2',
             '--dir='+str(out),'--out='+item['filename'],item['url']]
        with (state/(kind+'.aria2c.stdout.log')).open('xb') as stdout,(state/(kind+'.aria2c.stderr.log')).open('xb') as stderr:
            subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,check=True,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if kind=='wheel':
            if sha(target)!=item['sha256'] or target.stat().st_size!=item['bytes']:
                raise ValueError('Official PyPI checksum and complete wheel length required')
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None:raise ValueError('Wheel CRC failure')
        transfers[kind]=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),
                             publisher_sha256_available=kind=='wheel')
    from pypdf import PdfReader
    pdf=PdfReader(out/c['paper']['filename'],strict=True)
    if len(pdf.pages)<10 or 'TrajectoryNet' not in pdf.pages[0].extract_text():
        raise ValueError('Complete intended original TrajectoryNet paper required')
    overlay=Path(c['overlay_directory']);overlay.mkdir(parents=True,exist_ok=False)
    target=overlay/'site-packages'
    cmd=[c['python'],'-B','-m','pip','install','--no-deps','--no-compile','--no-index',
         '--disable-pip-version-check','--target='+str(target),str(out/c['wheel']['filename'])]
    with (state/'pip.stdout.log').open('xb') as stdout,(state/'pip.stderr.log').open('xb') as stderr:
        subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    installed=[dict(path=p.relative_to(target).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(target.rglob('*')) if p.is_file()]
    proof=dict(passed=True,source_commit=c['source_commit'],source_tree=c['source_tree'],
               original_source_files=len(sources),all_original_Git_blobs_verified=True,
               source_files=sources,transfers=transfers,paper_pages=len(pdf.pages),
               isolated_import_directory=str(target),installed_files=installed,
               base_environment_modified=False,torchdiffeq_runtime_version='0.0.1',
               native_checkpoint_replay_complete=False,historical_paper_equivalence_certified=False,
               configuration_sha256=sha(ROOT/CONFIG),all_original_experiments_complete=False)
    exclusive(out/'verified_acquisition.json',proof)
    exclusive(state/'verified_acquisition.json',proof)
    return proof


if __name__=='__main__':
    p=acquire(read(ROOT/CONFIG))
    print(json.dumps({k:p[k] for k in ['passed','original_source_files','paper_pages','torchdiffeq_runtime_version','base_environment_modified']}))
