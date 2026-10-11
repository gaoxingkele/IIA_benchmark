"""Verified PyPI wheel in a fresh isolated target; no shared environment changes."""
import argparse
from datetime import datetime,timezone
import os
import subprocess
import psutil
from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT,read,sha
from scripts.flow_matching.acquire_flow_mismatching_visa_v1 import download,json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=read(ROOT/args.config)
    if sha(cfg['python'])!=cfg['python_sha256']:raise ValueError('Frozen component Python differs')
    target=ROOT/cfg['isolated_target']
    if target.exists():raise FileExistsError('Preserve previous isolated component dependency target')
    state_root=ROOT/cfg['state_root'];state_root.mkdir(parents=True,exist_ok=False)
    process=psutil.Process();state=dict(pid=process.pid,create_time=process.create_time(),complete=False,downloads=[])
    def update(value):state.update(value,captured_utc=datetime.now(timezone.utc).isoformat());json_write(state_root/'status.json',state)
    try:
        path=download(cfg,cfg['source'],state,update)
        if sha(path)!=cfg['source']['publisher_sha256']:raise ValueError('PyPI publisher wheel SHA256 differs')
        command=[cfg['python'],'-X','utf8','-m','pip','install','--no-deps','--no-index','--no-compile','--target',str(target.resolve()),str(path.resolve())]
        update(dict(phase='installing_isolated_verified_wheel',command=command))
        subprocess.run(command,cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        report=dict(passed=True,captured_utc=datetime.now(timezone.utc).isoformat(),configuration=args.config,
            configuration_sha256=sha(ROOT/args.config),publisher_wheel_sha256_verified=True,
            isolated_target=str(target.resolve()),command=command,
            shared_python_site_packages_modified=False,shared_numpy_torch_versions_modified=False,
            native_benchmark_performance=False,download=state['downloads'][0],
            original_full_training_environment_certified=False)
        json_write(ROOT/cfg['output_directory']/'isolated_opencv_dependency_installation.json',report)
        update(dict(phase='completed_isolated_native_dependency_installation',complete=True,active_pid=None))
    except Exception as error:
        update(dict(phase='failed_isolated_dependency_attempt_preserved',complete=False,error=repr(error)));raise


if __name__=='__main__':main()
