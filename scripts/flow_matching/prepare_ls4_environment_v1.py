"""Acquire frozen author dependencies with aria2 and install isolated Python offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--install',action='store_true')
    args=parser.parse_args();config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    environment=json.loads((ROOT/config['environment_config']).read_text(encoding='utf-8'))
    base=ROOT/config['state_root'];base.mkdir(parents=True,exist_ok=True)
    sources=config['sources']
    if args.install:
        for source in sources:
            if sha(ROOT/source['path'])!=source['checksum'].split(':',1)[1]:
                raise ValueError('Registered package SHA256 differs')
        python=ROOT/environment['python']
        wheels=ROOT/Path(sources[0]['path']).parent
        requirements=base/'offline_requirements.txt'
        requirements.write_text('\n'.join(str((ROOT/source['path']).resolve()) for source in sources)+'\n',encoding='utf-8')
        wheel=next(ROOT/source['path'] for source in sources if source['name'].lower()=='wheel')
        commands=[[str(python),'-m','pip','install','--no-index','--no-deps',str(wheel)],
            [str(python),'-m','pip','install','--no-index','--no-build-isolation','--find-links',str(wheels),'-r',str(requirements)],
            [str(python),'-m','pip','check']]
        log=base/'install.log'
        if log.exists():
            raise FileExistsError('Keep prior installation evidence')
        with log.open('xb') as stream:
            for command in commands:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        write(base/'installation_receipt.json',dict(passed=True,commands=commands,log_sha256=sha(log),
            configuration_sha256=sha(ROOT/args.config),captured_utc=datetime.now(timezone.utc).isoformat()))
        print('Offline exact dependency installation and pip check passed');return
    from scripts.data_acquisition.download_public_datasets import download_file
    log=base/'downloads.log'
    if log.exists():
        raise FileExistsError('Keep original dependency download evidence')
    progress=dict(configuration_sha256=sha(ROOT/args.config),completed=[],required=len(sources))
    def acquire(source):
        path=ROOT/source['path']
        if path.exists() and sha(path)==source['checksum'].split(':',1)[1]:
            return dict(id=source['id'],path=source['path'],sha256=sha(path),bytes=path.stat().st_size,backend='preexisting_verified')
        download_file(source)
        if sha(path)!=source['checksum'].split(':',1)[1]:
            raise ValueError('Publisher checksum differs')
        return dict(id=source['id'],path=source['path'],sha256=sha(path),bytes=path.stat().st_size,backend='aria2c')
    with log.open('wb') as stream:
        # Native aria2 stdout/stderr and Python logging share one UTF8 byte log.
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(acquire,source) for source in sources]
            for future in as_completed(futures):
                progress['completed'].append(future.result());write(base/'download_progress.json',progress)
    progress.update(passed=True,publisher_checksums_verified=True,captured_utc=datetime.now(timezone.utc).isoformat())
    write(base/'download_receipt.json',progress)


if __name__=='__main__':
    main()
