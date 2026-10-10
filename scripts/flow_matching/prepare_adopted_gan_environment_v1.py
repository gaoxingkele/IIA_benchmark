"""Copy the frozen native environment, install checksum-pinned legacy dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    source = Path(config['source_environment']).resolve()
    destination = Path(config['destination_environment']).resolve()
    if source == destination or destination.exists() or destination.parent != Path(config['wheel_directory']).parent.resolve():
        raise ValueError('Fresh separate registered environment required')
    source_lock_hash = sha(ROOT / config['source_environment_lock'])
    package_directory = Path(config['wheel_directory'])
    package_directory.mkdir(parents=True, exist_ok=False)
    def download(item):
        target = package_directory / item['filename']
        command = [config['aria2c'], '--all-proxy=' + config['proxy'], '--auto-file-renaming=false',
                   '--allow-overwrite=false', '--continue=false', '--file-allocation=none',
                   '--max-connection-per-server=2', '--split=2', '--max-tries=2', '--retry-wait=3',
                   '--connect-timeout=20', '--timeout=30', '--summary-interval=0',
                   '--checksum=sha-256=' + item['sha256'], '--dir=' + str(package_directory),
                   '--out=' + item['filename'], item['url']]
        with target.with_suffix(target.suffix + '.download.log').open('xb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            raise ValueError('Official PyPI package size/checksum differs')
        return str(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, item) for item in config['packages']]
        wheels = [f.result() for f in futures]
    shutil.copytree(source, destination)
    executable = destination / 'python.exe'
    environment = dict(os.environ, PYTHONPATH='', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
    probe = subprocess.run([str(executable), '-X', 'utf8', '-c',
        'import sys,json; print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,path=sys.path)))'],
        env=environment, capture_output=True, text=True, check=True)
    identity = json.loads(probe.stdout)
    if Path(identity['executable']).resolve() != executable.resolve() or any(str(source).lower() in p.lower() for p in identity['path']):
        raise ValueError('Copied native runtime must not reference the live source environment')
    install = subprocess.run([str(executable), '-X', 'utf8', '-m', 'pip', '--isolated', '--disable-pip-version-check',
        'install', '--no-index', '--no-deps', '--force-reinstall', *wheels],
        env=environment, capture_output=True, text=True, check=True)
    (destination.parent / 'installation.stdout.log').write_text(install.stdout, encoding='utf-8')
    (destination.parent / 'installation.stderr.log').write_text(install.stderr, encoding='utf-8')
    freeze = subprocess.run([str(executable), '-X', 'utf8', '-m', 'pip', '--isolated', 'freeze', '--all'],
                            env=environment, capture_output=True, text=True, check=True)
    if sha(ROOT / config['source_environment_lock']) != source_lock_hash:
        raise ValueError('Live source environment lock changed')
    report = dict(configuration=args.config, configuration_sha256=sha(ROOT / args.config),
        source_environment_lock_sha256=source_lock_hash, source_environment_unchanged=True,
        destination_identity=identity, official_package_checksums_verified=config['packages'],
        frozen_packages=freeze.stdout.splitlines(), local_compatibility_environment=True,
        original_historical_environment_equivalence=False, full_native_training_completed=False)
    write(ROOT / config['output_receipt'], report)
    print(dict(isolated_environment=str(destination), packages_added_or_replaced=len(wheels), source_environment_unchanged=True))


if __name__ == '__main__':
    main()
