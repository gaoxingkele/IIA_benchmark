"""Download the original author's recommended Windows wheels and verify SHA256."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.run_tsad_queue import exclusive_lock

CONFIG = 'configs/acquisition/fm_otcfm_author_py310_torch_wheels.v1.json'


def verify_asset(asset):
    path = Path(asset['path'])
    if path.stat().st_size != asset['expected_bytes'] or sha(path) != asset['checksum'].split(':', 1)[1]:
        raise ValueError('Preserve an incomplete or checksum-mismatching author wheel: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default=CONFIG)
    config_path = parser.parse_args().config; c = read(ROOT / config_path)
    pin = c['author_requirements']
    if sha(ROOT / pin['path']) != pin['sha256']:
        raise ValueError('Original author environment instructions changed')
    base = ROOT / c['state_root']; base.mkdir(parents=True, exist_ok=True)
    lock = exclusive_lock(base / 'download.lock')
    if lock is None: raise RuntimeError('These exact author wheels are already being downloaded')
    try:
        for asset in c['sources']:
            path = Path(asset['path']); path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size == asset['expected_bytes']:
                verify_asset(asset)
            else:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                command = [c['aria2'], '--all-proxy=' + c['proxy'], '--continue=true',
                    '--auto-file-renaming=false', '--allow-overwrite=false', '--file-allocation=none',
                    '--check-integrity=true', '--checksum=sha-256=' + asset['checksum'].split(':', 1)[1],
                    '--max-connection-per-server=4', '--split=4', '--min-split-size=16M',
                    '--connect-timeout=30', '--timeout=60', '--max-tries=3', '--retry-wait=3',
                    '--summary-interval=10', '--console-log-level=warn',
                    '--dir=' + str(path.parent), '--out=' + path.name, asset['url']]
                with (base / (asset['id'] + '.' + stamp + '.download.log')).open('xb') as log:
                    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode: raise RuntimeError('Preserved failed author-wheel download: ' + asset['id'])
                verify_asset(asset)
            receipt = path.with_suffix(path.suffix + '.verified.json')
            if not receipt.exists():
                receipt.write_text(json.dumps(dict(verified=True, bytes=path.stat().st_size,
                    sha256=sha(path), official_index=asset['index_url'], source_url=asset['url']), indent=2) + '\n', encoding='utf-8')
            print(json.dumps(dict(verified=True, package=asset['id'], bytes=asset['expected_bytes'])), flush=True)
        output = base / 'verified_author_torch_wheels.json'
        if output.exists(): raise FileExistsError('Preserve the previous wheel acquisition receipt')
        output.write_text(json.dumps(dict(passed=True, configuration=config_path, configuration_sha256=sha(ROOT / config_path),
            sources=c['sources'], exact_author_torch='1.12.1+cu113', exact_author_torchvision='0.13.1+cu113',
            all_bytes_and_checksums_verified=True, any_experiment_or_performance_completed=False), indent=2) + '\n', encoding='utf-8')
    finally:
        lock.close()


if __name__ == '__main__': main()
