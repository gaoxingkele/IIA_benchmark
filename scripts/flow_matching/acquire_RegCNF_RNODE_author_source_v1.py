"""Acquire pinned Finlay author sources and verify every original Git blob."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG = 'configs/acquisition/fm_RegCNF_RNODE_author_source.v1.json'


def verified_archive(path, tree, prefix):
    if tree.get('truncated') is not False:
        raise ValueError('A complete recursive author Git tree is required')
    expected = {n['path']: n for n in tree['tree'] if n['type'] == 'blob'}
    if any(n['type'] not in ('tree', 'blob') for n in tree['tree']):
        raise ValueError('Submodules need separate acquisition, never omit them')
    records = []
    seen = set()
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Original source ZIP CRC failed')
        for member in archive.infolist():
            # Python normalizes Windows backslashes in .filename; inspect the
            # original central-directory name before any extraction.
            name = member.orig_filename
            parts = PurePosixPath(name).parts
            if ('\\' in name or ':' in name or not parts or parts[0] != prefix
                    or '..' in parts or PurePosixPath(name).is_absolute()
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe source archive member')
            if member.is_dir():
                continue
            relative = PurePosixPath(*parts[1:]).as_posix()
            if relative not in expected or relative in seen:
                raise ValueError('Duplicate or unregistered source archive member')
            content = archive.read(member)
            node = expected[relative]
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if len(content) != node['size'] or blob != node['sha']:
                raise ValueError('Pinned official source Git blob mismatch: ' + relative)
            seen.add(relative)
            records.append(dict(path=relative, bytes=len(content), git_blob_sha1=blob,
                                sha256=hashlib.sha256(content).hexdigest()))
    if seen != set(expected):
        raise ValueError('Incomplete pinned author source archive')
    return sorted(records, key=lambda n: n['path'])


def acquire(c):
    state = ROOT / c['state_root']
    output = Path(c['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'verified_acquisition.json').exists():
        raise FileExistsError('Preserve existing acquisition and its original receipt')
    tree_path = ROOT / c['git_tree_receipt']
    if sha(tree_path) != c['git_tree_receipt_sha256']:
        raise ValueError('Pinned author Git tree metadata changed')
    tree = read(tree_path)
    if tree['sha'] != c['source_tree']:
        raise ValueError('Pinned author root tree mismatch')
    transfers = {}
    for kind in ('archive', 'paper'):
        item = c[kind]
        target = output / item['filename']
        if not target.exists() or Path(str(target) + '.aria2').exists():
            cmd = [c['aria2c'], '--all-proxy=' + c['proxy'], '--continue=true',
                   '--allow-overwrite=false', '--auto-file-renaming=false',
                   '--max-connection-per-server=4', '--split=4', '--connect-timeout=20',
                   '--timeout=30', '--max-tries=4', '--retry-wait=2',
                   '--dir=' + str(output), '--out=' + item['filename'], item['url']]
            with (state / (kind + '.aria2c.stdout.log')).open('xb') as out, \
                    (state / (kind + '.aria2c.stderr.log')).open('xb') as err:
                subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                               check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if Path(str(target) + '.aria2').exists():
            raise ValueError('Partial download is not complete')
        transfers[kind] = dict(path=str(target), bytes=target.stat().st_size,
                               sha256=sha(target), publisher_sha256_available=False)
    archive = output / c['archive']['filename']
    prefix = 'ffjord-rnode-' + c['source_commit']
    records = verified_archive(archive, tree, prefix)
    from pypdf import PdfReader
    paper = PdfReader(output / c['paper']['filename'], strict=True)
    page_text = paper.pages[0].extract_text().lower()
    if len(paper.pages) < 5 or 'finlay' not in page_text or 'regularization' not in page_text:
        raise ValueError('Complete intended Finlay PDF required')
    original = output / 'original'
    original.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for row in records:
            target = original / row['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(z.read(prefix + '/' + row['path']))
            if sha(target) != row['sha256']:
                raise ValueError('Original extraction hash mismatch')
    receipt = dict(passed=True, source_repository=c['source_repository'], source_commit=c['source_commit'],
                   source_tree=c['source_tree'], complete_zip_crc_verified=True,
                   every_original_git_blob_verified=True, original_source_directory=str(original),
                   complete_source_files=len(records), source_files=records, transfers=transfers,
                   paper_pages=len(paper.pages), paper=c['paper'], configuration_sha256=sha(ROOT / CONFIG),
                   official_Git_object_integrity_verified=True, publisher_raw_sha256_available=False,
                   native_training_complete=False, historical_OTCFM_RegCNF_equivalence_certified=False,
                   all_original_experiments_complete=False, boundary=c['boundary'])
    exclusive(output / 'verified_acquisition.json', receipt)
    exclusive(state / 'verified_acquisition.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=CONFIG)
    a = p.parse_args()
    receipt = acquire(read(ROOT / a.config))
    print(json.dumps({k: receipt[k] for k in ('passed', 'source_commit', 'complete_source_files', 'paper_pages')}))


if __name__ == '__main__':
    main()
