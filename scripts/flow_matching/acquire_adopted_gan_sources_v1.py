"""Acquire pinned author repositories with aria2 and verify every Git blob."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe author archive member')
    return path


def verify_blob(payload, entry):
    if len(payload) != entry['size']:
        raise ValueError('Pinned Git blob byte size differs')
    digest = hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()
    if digest != entry['sha']:
        raise ValueError('Pinned Git blob checksum differs')
    return hashlib.sha256(payload).hexdigest()


def download(config, url, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve original or partial registered downloads')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [config['aria2c'], '--all-proxy=' + config['proxy'], '--auto-file-renaming=false',
               '--allow-overwrite=false', '--continue=false', '--file-allocation=none',
               '--max-connection-per-server=2', '--split=2', '--max-tries=2', '--retry-wait=3',
               '--connect-timeout=20', '--timeout=30', '--summary-interval=0',
               '--dir=' + str(path.parent), '--out=' + path.name, url]
    with path.with_suffix(path.suffix + '.download.log').open('xb') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    return dict(url=url, path=str(path), bytes=path.stat().st_size, sha256=sha(path), command=command)


def extract_verified(asset, registered_root):
    destination = Path(asset['original_directory']).resolve()
    if registered_root.resolve() not in destination.parents or destination.exists():
        raise ValueError('Fresh original directory inside registered asset root required')
    expected = {e['path']: e for e in asset['expected_git_entries']}
    if len(expected) != len(asset['expected_git_entries']):
        raise ValueError('Unique pinned Git paths required')
    receipts = []
    with zipfile.ZipFile(asset['archive_path']) as archive:
        members = [e for e in archive.infolist() if not e.is_dir()]
        stripped = {}
        for member in members:
            path = safe_member(member.filename)
            if len(path.parts) < 2:
                raise ValueError('Pinned repository archive root required')
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in stripped:
                raise ValueError('Duplicate archive member')
            stripped[relative] = member
        if set(stripped) != set(expected):
            raise ValueError('Every pinned author Git blob must be present, no extra files')
        destination.mkdir(parents=True)
        for relative, entry in expected.items():
            if entry['mode'] not in ('100644', '100755'):
                raise ValueError('Only pinned regular Git files accepted')
            payload = archive.read(stripped[relative])
            digest = verify_blob(payload, entry)
            target = (destination / relative).resolve()
            if destination not in target.parents:
                raise ValueError('Resolved archive path escapes registered original directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
            if sha(target) != digest:
                raise ValueError('Extracted pinned author bytes differ')
            receipts.append(dict(path=relative, bytes=len(payload), github_git_blob_sha1=entry['sha'],
                                 sha256=digest, original_git_mode=entry['mode']))
    return dict(id=asset['id'], repository=asset['repository'], commit=asset['commit'],
                tree_sha=asset['tree_sha'], original_directory=str(destination),
                all_git_blobs_verified=True, files=len(receipts), total_bytes=sum(r['bytes'] for r in receipts),
                file_receipts=receipts, archive_sha256=sha(asset['archive_path']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(ROOT / args.config)
    output = ROOT / config['output_directory']
    if output.exists():
        raise FileExistsError('Preserve every acquisition attempt')
    jobs = [(a['archive_url'], a['archive_path']) for a in config['assets']]
    jobs += [(a['pdf_url'], a['pdf_path']) for a in config['assets']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, config, url, path) for url, path in jobs]
        results = [f.result() for f in futures]
    original = [extract_verified(a, Path(config['asset_root'])) for a in config['assets']]
    from pypdf import PdfReader
    papers = []
    for asset in config['assets']:
        path = Path(asset['pdf_path'])
        if not path.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Original open PDF returned non-PDF content')
        pages = len(PdfReader(path).pages)
        if not pages:
            raise ValueError('Complete parseable original open paper required')
        papers.append(dict(id=asset['id'], arxiv_id=asset['arxiv_id'], url=asset['pdf_url'],
                           path=str(path), pages=pages, sha256=sha(path), full_text_status='open_pdf',
                           download_status='downloaded'))
    output.mkdir(parents=True)
    proof = dict(configuration=args.config, configuration_sha256=sha(ROOT / args.config),
                 original_repositories=original, original_papers=papers, downloads=results,
                 all_original_git_files_and_open_pdfs_verified=True, benchmark_performance=False,
                 full_native_experiments_completed=False, all_original_experiments_complete=False)
    write(output / 'acquisition_receipt.json', proof)
    write(output / 'validation.json', dict(passed=True, outputs={'acquisition_receipt.json': sha(output / 'acquisition_receipt.json')}))
    print(dict(original_repositories=len(original), original_git_files=sum(r['files'] for r in original),
               open_pdfs=len(papers), benchmark_performance=False))


if __name__ == '__main__':
    main()
