"""Acquire checksum-pinned SHCL wheels and install into a new isolated environment."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.flow_matching.timegan_native_common_v1 import ROOT,read,sha
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def run(command,log):
    with log.open('xb') as stream:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        code=child.wait()
    if code: raise RuntimeError('Environment command failed; preserved log: '+str(log))


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument('--config',required=True)
    args=parser.parse_args(); config=read(ROOT/args.config)
    for pin in config['source_receipts']:
        if sha(ROOT/pin['path'])!=pin['sha256']: raise ValueError('Frozen environment acquisition changed')
    if sha(config['base_python'])!=config['base_python_sha256']: raise ValueError('Python base changed')
    base=Path(config['root']); base.mkdir(parents=True,exist_ok=False)
    wheels=base/'wheels'; wheels.mkdir(); logs=base/'logs'; logs.mkdir(); records=[]
    for index,wheel in enumerate(config['wheels']):
        command=[config['aria2c'],'--all-proxy='+config['proxy'],'--check-integrity=true',
            '--checksum=sha-256='+wheel['sha256'],'--split=8','--max-connection-per-server=8',
            '--file-allocation=none','--auto-file-renaming=false','--allow-overwrite=false',
            '--max-tries=3','--retry-wait=3','--connect-timeout=20','--timeout=30',
            '--console-log-level=warn','--summary-interval=0','--dir='+str(wheels),'--out='+wheel['filename'],wheel['url']]
        run(command,logs/('download_'+str(index).zfill(2)+'.log'))
        path=wheels/wheel['filename']
        if sha(path)!=wheel['sha256'] or (wheel.get('expected_size') and path.stat().st_size!=wheel['expected_size']):
            raise ValueError('Complete publisher wheel checksum/size mismatch')
        records.append(dict(wheel,actual_bytes=path.stat().st_size,publisher_SHA256_verified=True))
        exclusive(base/('wheel_'+str(index).zfill(2)+'.json'),records[-1])
        print(json.dumps(dict(downloaded=len(records),required=len(config['wheels']),package=wheel['package'])),flush=True)
    run([config['base_python'],'-m','venv',str(base/'venv')],logs/'create_venv.log')
    python=config['python']
    run([python,'-m','pip','install','--no-index','--no-deps',*[str(wheels/w['filename']) for w in config['wheels']]],logs/'offline_install.log')
    run([python,'-m','pip','check'],logs/'pip_check.log')
    snapshot=subprocess.check_output([python,'-m','pip','list','--format=json'],encoding='utf-8')
    packages=json.loads(snapshot)
    installed={row['name'].lower().replace('_','-'):row['version'] for row in packages}
    for wheel in config['wheels']:
        if installed[wheel['package']]!=wheel['version']: raise ValueError('Installed package version mismatch')
    report=dict(captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
        configuration_sha256=sha(ROOT/args.config),python=python,python_sha256=sha(python),
        actual_python_version=subprocess.check_output([python,'--version'],encoding='utf-8').strip(),
        installed_packages=packages,wheels=records,all_publisher_SHA256_verified=True,
        offline_install_complete=True,pip_check_passed=True,existing_environments_modified=False,
        NumPy_compatibility_correction_explicit=True,original_paper_environment_equivalence=False,
        native_tensor_import_forward_backward_verified=False,all_original_experiments_complete=False)
    exclusive(base/'environment_acquisition_report.json',report)
    print(json.dumps(dict(environment_ready=True,torch=installed['torch'],numpy=installed['numpy'])),flush=True)


if __name__=='__main__':main()
