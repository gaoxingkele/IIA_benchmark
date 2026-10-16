"""Acquire official versioned TeX, verify archive, extract text without execution."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG='configs/acquisition/fm_TrajectoryNet_arxiv_source.v1.json'
TEXT_SUFFIXES={'.tex','.bib','.sty','.cls','.bbl','.txt','.csv','.json','.yaml','.yml','.sh','.py','.md','.rst','.cfg','.ini'}


def safe_member(name):
    value=PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe original archive member path')
    return value


def inspect_archive(path, target, c):
    if path.stat().st_size>c['maximum_compressed_bytes']:
        raise ValueError('Source archive exceeds registered bound')
    with path.open('rb') as stream:
        is_gzip=stream.read(2)==b'\x1f\x8b'
    if is_gzip:
        total=0
        with gzip.open(path,'rb') as stream:
            while chunk:=stream.read(2**20):
                total+=len(chunk)
                if total>c['maximum_uncompressed_bytes']:
                    raise ValueError('Expanded source archive exceeds registered bound')
    members=[];total=0;names=set()
    with tarfile.open(path,'r:*') as archive:
        for member in archive:
            relative=safe_member(member.name)
            if member.isdir():continue
            if not member.isfile():raise ValueError('Never extract links,devices,or other special members')
            if relative.as_posix() in names:raise ValueError('Never overwrite duplicate archive members')
            names.add(relative.as_posix());total+=member.size
            if total>c['maximum_uncompressed_bytes'] or len(names)>c['maximum_members']:
                raise ValueError('Original archive size/count bound exceeded')
            content=archive.extractfile(member).read()
            if len(content)!=member.size:raise ValueError('Truncated original source member')
            record=dict(path=relative.as_posix(),bytes=member.size,sha256=hashlib.sha256(content).hexdigest(),extracted=False)
            if relative.suffix.lower() in TEXT_SUFFIXES:
                destination=target/relative.as_posix()
                if not destination.resolve().is_relative_to(target.resolve()):
                    raise ValueError('Member must stay inside fresh original source directory')
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:stream.write(content)
                if sha(destination)!=record['sha256']:raise ValueError('Extracted member checksum differs')
                record['extracted']=True
            members.append(record)
    if not any(r['path'].endswith('.tex') and r['extracted'] for r in members):
        raise ValueError('Require actual original TeX source, not an error or PDF payload')
    return members


def acquire(c):
    root=Path(c['output_directory']);root.mkdir(parents=True,exist_ok=False)
    state=ROOT/c['state_root'];state.mkdir(parents=True,exist_ok=False)
    attempts=[]
    for index,url in enumerate(c['source_links']):
        out=root/f'official_endpoint_{index+1}';out.mkdir()
        archive=out/'original_source.tar'
        cmd=[c['aria2c'],'--all-proxy='+c['proxy'],'--allow-overwrite=false','--auto-file-renaming=false',
            '--continue=false','--max-tries=1','--connect-timeout=20','--timeout=30',
            '--max-download-limit=8M','--dir='+str(out),'--out='+archive.name,url]
        with (state/f'endpoint{index+1}.stdout.log').open('xb') as stdout,(state/f'endpoint{index+1}.stderr.log').open('xb') as stderr:
            result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        attempt=dict(url=url,exit_code=result.returncode,local_path=str(archive))
        attempts.append(attempt)
        if result.returncode!=0:continue
        digest=sha(archive)
        try:members=inspect_archive(archive,out/'text_source',c)
        except (ValueError,tarfile.TarError,OSError,EOFError) as error:
            attempt.update(validation_error=str(error),bytes=archive.stat().st_size,sha256=digest)
            continue
        if sha(archive)!=digest:raise ValueError('Original archive changed during complete member verification')
        proof=dict(passed=True,arxiv_id=c['arxiv_id'],citation=c['citation'],source_url=url,
            archive=str(archive),archive_sha256=digest,archive_bytes=archive.stat().st_size,
            complete_member_count=len(members),members=members,official_download_attempts=attempts,
            text_directory=str(out/'text_source'),publisher_checksum_available=False,
            member_integrity='complete archive traversal and SHA256; extracted text reread and compared',
            source_execution_performed=False,native_models_completed=0,
            configuration_sha256=sha(ROOT/CONFIG),all_original_experiments_complete=False)
        exclusive(root/'verified_acquisition.json',proof);exclusive(state/'verified_acquisition.json',proof)
        print(json.dumps({k:proof[k] for k in ['passed','arxiv_id','archive_bytes','complete_member_count','text_directory']}));return
    exclusive(state/'failed_attempts.json',dict(attempts=attempts,preserved=True,all_original_experiments_complete=False))
    raise RuntimeError('Official TeX source acquisition failed; all attempts preserved')


if __name__=='__main__':acquire(read(ROOT/CONFIG))
