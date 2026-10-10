"""Acquire pinned upstream USHCN assets without following upstream symlinks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from scripts.data_acquisition.download_public_datasets import find_aria2

ROOT = Path(__file__).resolve().parents[2]


def verify_blob(path, asset):
    size = path.stat().st_size
    if size != asset['expected_size']:
        raise ValueError('Upstream asset size differs')
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    content = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            blob.update(block)
            content.update(block)
    if blob.hexdigest() != asset['git_blob_sha1']:
        raise ValueError('Upstream Git blob identity differs')
    return dict(size=size, git_blob_sha1=blob.hexdigest(), sha256=content.hexdigest())


def validate_asset(asset, commit, root=ROOT):
    relative = Path(asset['path'])
    allowed = root / 'data/public_datasets/flow_matching/ls4_native_ushcn_v1'
    target = root / relative
    if relative.is_absolute() or not target.resolve().is_relative_to(allowed.resolve()):
        raise ValueError('Asset destination escaped registered data directory')
    expected = f"https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/{commit}/{asset['upstream_path']}"
    if asset['url'] != expected:
        raise ValueError('Download URL is not the pinned upstream asset')
    if asset['upstream_mode'] not in ('100644', '120000'):
        raise ValueError('Only regular blobs and preserved link text are supported')
    if asset['upstream_mode'] == '120000' and not target.name.endswith('.upstream_link.txt'):
        raise ValueError('Upstream symlink must be preserved as inert text')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    receipt_dir = ROOT / config['receipt_directory']
    if receipt_dir.exists():
        raise FileExistsError('Preserve every acquisition receipt')
    targets = [validate_asset(a, config['commit']) for a in config['assets']]
    if len(set(p.resolve() for p in targets)) != len(targets):
        raise ValueError('Duplicate acquisition destinations')
    aria = find_aria2()
    if not aria:
        raise RuntimeError('Registered acquisition requires aria2c')
    receipt_dir.mkdir(parents=True)
    receipts = []
    for asset, target in zip(config['assets'], targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.exists()
        if not reused:
            attempt = target.parent / '_download_attempts' / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            command = [aria, f"--all-proxy={config['proxy']}", '--split=1',
                       '--max-connection-per-server=1', '--max-tries=2',
                       '--retry-wait=2', '--connect-timeout=20', '--timeout=40',
                       '--file-allocation=none', '--allow-overwrite=false',
                       '--auto-file-renaming=false', '--console-log-level=error',
                       '--summary-interval=0', '--dir=' + str(attempt),
                       '--out=' + target.name, asset['url']]
            with (attempt / 'aria2.log').open('wb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=120,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            candidate = attempt / target.name
            verify_blob(candidate, asset)
            with candidate.open('rb') as src, target.open('xb') as dest:
                shutil.copyfileobj(src, dest)
        evidence = verify_blob(target, asset)
        receipts.append(dict(asset | evidence, existing_verified=reused,
                             resolved_path=str(target.resolve()),
                             upstream_link_followed=False))
        print('verified', asset['upstream_path'], evidence['size'], flush=True)
        (receipt_dir / 'progress.json').write_text(
            json.dumps(dict(completed=len(receipts), required=len(targets)), indent=2)+'\n',
            encoding='utf-8')
    receipt = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                   config_path=args.config,
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   downloader='aria2c', proxy=config['proxy'],
                   repository=config['repository'], commit=config['commit'],
                   assets=receipts, all_registered_assets_verified=True,
                   historical_author_fold_arrays_available=False,
                   boundary=config['boundary'])
    (receipt_dir / 'acquisition_receipt.json').write_text(
        json.dumps(receipt, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
