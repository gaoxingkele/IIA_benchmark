"""Download author-published MEGA shares via its anonymous public API.

Requires requests and pycryptodome. Verifies MEGA's plaintext MAC, then SHA256.
No accounts, access-control bypass, quota evasion, or existing-file overwrites.
Protocol references: https://github.com/meganz/webclient and
https://github.com/odwyersoftware/mega.py/blob/master/src/mega/mega.py
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import subprocess
import sys
from urllib.parse import urlparse

import requests
from Crypto.Cipher import AES
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def decode64(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def words(raw):
    return struct.unpack('>' + str(len(raw) // 4) + 'I', raw)


def file_parts(raw):
    value = words(raw)
    return struct.pack('>4I', *(value[i] ^ value[i + 4] for i in range(4))), value[4:6], value[6:8]


def attributes(raw, key):
    plain = AES.new(key, AES.MODE_CBC, bytes(16)).decrypt(decode64(raw)).rstrip(b'\0')
    if not plain.startswith(b'MEGA'):
        raise ValueError('Invalid decrypted MEGA attributes')
    return json.loads(plain[4:])


def chunks(size):
    offset, count = 0, 1
    while offset < size:
        length = min(count * 131072, size - offset)
        yield length
        offset += length
        count = min(count + 1, 8)


class PublicMega:
    def __init__(self, proxy, resume=False):
        self.session = requests.Session()
        self.session.proxies = {'http': proxy, 'https': proxy}
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5,
                                                                  allowed_methods=frozenset(['GET', 'POST']),
                                                                  status_forcelist=[500, 502, 503, 504])))
        self.resume = resume
        self.proxy = proxy
        self.range_local = threading.local()

    def range_bytes(self, url, offset, length):
        if not hasattr(self.range_local, 'session'):
            self.range_local.session = requests.Session()
            self.range_local.session.proxies = {'http': self.proxy, 'https': self.proxy}
            self.range_local.session.mount('https://', HTTPAdapter(max_retries=Retry(
                total=2, backoff_factor=0.5, allowed_methods=['GET'],
                status_forcelist=[500, 502, 503, 504])))
        response = self.range_local.session.get(url, headers={'Range': f'bytes={offset}-{offset + length - 1}'},
                                                timeout=(25, 90))
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise IOError('Server did not honor public byte range; incomplete transfer retained')
        if len(response.content) != length:
            raise IOError('Public byte range returned incorrect length')
        return response.content

    def api(self, command, folder=None):
        params = {'id': time.time_ns() % 2147483647}
        if folder:
            params['n'] = folder
        response = self.session.post('https://g.api.mega.co.nz/cs', params=params,
                                     json=[command], timeout=(20, 60))
        response.raise_for_status()
        value = response.json()[0]
        if isinstance(value, int):
            raise RuntimeError(f'MEGA public API error {value}; access/quota gate retained')
        return value

    def enumerate(self, source):
        parsed = urlparse(source['url'])
        kind, handle = parsed.path.strip('/').split('/')[:2]
        key = decode64(parsed.fragment)
        if kind == 'file':
            info = self.api({'a': 'g', 'g': 1, 'p': handle})
            aeskey, _, _ = file_parts(key)
            name = attributes(info['at'], aeskey)['n']
            return [dict(handle=handle, key=key.hex(), name=name, size=info['s'], folder=None)], []
        listing = self.api({'a': 'f', 'c': 1, 'r': 1, 'ca': 1}, folder=handle)
        nodes = {}
        for node in listing['f']:
            decoded = AES.new(key, AES.MODE_ECB).decrypt(decode64(node['k'].split(':')[-1]))
            attr_key = file_parts(decoded)[0] if node['t'] == 0 else decoded
            nodes[node['h']] = dict(node, plaintext_name=attributes(node['a'], attr_key)['n'],
                                   decoded_key=decoded.hex())
        files, directories = [], []
        for node in nodes.values():
            ancestors, parent = [], node['p']
            while parent in nodes:
                parent_node = nodes[parent]
                if parent_node['p'] in nodes:
                    ancestors.append(parent_node['plaintext_name'])
                parent = parent_node['p']
            name = '/'.join(list(reversed(ancestors)) + [node['plaintext_name']])
            if node['t'] == 0:
                files.append(dict(handle=node['h'], key=node['decoded_key'], name=name,
                                  size=node['s'], folder=handle))
            else:
                directories.append(name)
        return files, directories

    def download(self, source, item):
        base = Path(source['path']).resolve()
        relative = Path(item['name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe public-share filename')
        target = (base / relative).resolve()
        target.relative_to(base)
        result = dict(source_id=source['id'], source_url=source['url'],
                      evidence_url=source['evidence_url'], paper_ids=source['paper_ids'],
                      path=str(target), relative_path=target.relative_to(Path.cwd()).as_posix(),
                      expected_bytes=item['size'], sha256=None, mega_mac_verified=False)
        if target.exists():
            try:
                if target.stat().st_size != item['size']:
                    raise ValueError('Existing size differs from public metadata; existing file preserved')
                result.update(status='existing_not_overwritten', bytes=target.stat().st_size,
                              sha256=verify_plain_file(target, bytes.fromhex(item['key'])), mega_mac_verified=True,
                              note='Existing raw file preserved; native MAC reverified.')
            except Exception as exc:
                result.update(status='existing_verification_failed', error=str(exc))
            return result
        temp = target.with_name(target.name + '.mega.part')
        if temp.exists() and not self.resume:
            result.update(status='partial_preserved', error='Existing partial retained; no overwrite.')
            return result
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            key, iv, meta = file_parts(bytes.fromhex(item['key']))
            cmd = {'a': 'g', 'g': 1, 'n' if item['folder'] else 'p': item['handle']}
            info = self.api(cmd, item['folder'])
            if not info.get('g'):
                raise RuntimeError('Public file no longer accessible; access gate retained')
            aggregate = AES.new(key, AES.MODE_CBC, bytes(16))
            chunk_iv = struct.pack('>4I', iv[0], iv[1], iv[0], iv[1])
            sha = hashlib.sha256()
            received = 0
            lengths = list(chunks(item['size']))
            final_mac = bytes(16)
            if temp.exists():
                existing = temp.stat().st_size
                with temp.open('rb') as old:
                    for length in lengths:
                        if received == existing:
                            break
                        plain = old.read(length)
                        if len(plain) != length:
                            raise ValueError('Partial does not end at a verified download chunk boundary')
                        sha.update(plain)
                        received += length
                        padded = plain + b'\0' * (-len(plain) % 16)
                        final_mac = aggregate.encrypt(AES.new(key, AES.MODE_CBC, chunk_iv).encrypt(padded)[-16:])
                if received != existing or existing > item['size']:
                    raise ValueError('Partial length invalid; existing partial preserved')
            counter = struct.pack('>4I', iv[0], iv[1], 0, 0)
            decrypt = AES.new(key, AES.MODE_CTR, nonce=b'', initial_value=int.from_bytes(counter, 'big') + received // 16)
            initial_offset = received
            pending, offset = [], 0
            for length in lengths:
                if offset >= initial_offset:
                    pending.append((offset, length))
                offset += length
            # Four ordinary public range requests, like a normal transfer client.
            # Decryption and authentication remain ordered and bounded in memory.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ranges, temp.open('ab' if temp.exists() else 'xb') as dest:
                queue = {}
                for index in range(min(4, len(pending))):
                    queue[index] = ranges.submit(self.range_bytes, info['g'], *pending[index])
                for index, (_, length) in enumerate(pending):
                    encrypted = queue.pop(index).result()
                    if index + 4 < len(pending):
                        queue[index + 4] = ranges.submit(self.range_bytes, info['g'], *pending[index + 4])
                    if len(encrypted) != length:
                        raise IOError(f'Truncated download: {received + len(encrypted)}/{item["size"]}')
                    plain = decrypt.decrypt(encrypted)
                    dest.write(plain)
                    sha.update(plain)
                    received += len(plain)
                    padded = plain + b'\0' * (-len(plain) % 16)
                    chunk_mac = AES.new(key, AES.MODE_CBC, chunk_iv).encrypt(padded)[-16:]
                    final_mac = aggregate.encrypt(chunk_mac)
            value = words(final_mac)
            if (value[0] ^ value[1], value[2] ^ value[3]) != meta:
                raise ValueError('MEGA native plaintext MAC mismatch; partial quarantined')
            if target.exists():
                raise FileExistsError('Target appeared during download; existing file preserved')
            # Windows rename refuses replacement; raw data remains untouched.
            temp.rename(target)
            result.update(status='downloaded', bytes=received, sha256=sha.hexdigest(), mega_mac_verified=True)
        except Exception as exc:
            result.update(status='failed', error=str(exc), partial_path=str(temp) if temp.exists() else None)
        print(json.dumps({k: result.get(k) for k in ['source_id', 'relative_path', 'status', 'bytes', 'error']}), flush=True)
        return result


def hash_file(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def verify_plain_file(path, raw_key):
    key, iv, meta = file_parts(raw_key)
    aggregate = AES.new(key, AES.MODE_CBC, bytes(16))
    chunk_iv = struct.pack('>4I', iv[0], iv[1], iv[0], iv[1])
    sha, final = hashlib.sha256(), bytes(16)
    with path.open('rb') as stream:
        for length in chunks(path.stat().st_size):
            plain = stream.read(length)
            sha.update(plain)
            padded = plain + b'\0' * (-len(plain) % 16)
            final = aggregate.encrypt(AES.new(key, AES.MODE_CBC, chunk_iv).encrypt(padded)[-16:])
    value = words(final)
    if (value[0] ^ value[1], value[2] ^ value[3]) != meta:
        raise ValueError('Existing public file native MAC mismatch; existing file preserved')
    return sha.hexdigest()


def save_manifest(output, value):
    """Merge verified artifacts and replace JSON atomically for report readers."""
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf8'))
        current = {f['relative_path']: f for f in value['files']}
        for old in previous.get('files', []):
            if old.get('mega_mac_verified') and old.get('sha256'):
                latest = current.get(old['relative_path'])
                if not latest or not latest.get('mega_mac_verified'):
                    current[old['relative_path']] = old
        value['files'] = list(current.values())
    temp = output.with_name(output.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(temp, output)


def wait_for_pids(pids):
    if os.name != 'nt':
        raise RuntimeError('--wait-pids is a Windows process watcher')
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    for pid in pids:
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                while kernel.WaitForSingleObject(handle, 10000) == 258:
                    pass
            finally:
                kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/acquisition/flow_matching_mega_sources.json')
    parser.add_argument('--manifest', default='papers/literature/flow_matching/mega_download_manifest.json')
    parser.add_argument('--proxy', default='http://127.0.0.1:17890')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='Append to this tool\'s plaintext .mega.part files; full native MAC still required')
    parser.add_argument('--wait-pids', nargs='*', type=int, default=[])
    parser.add_argument('--rounds', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    if args.wait_pids:
        print(f'Waiting for authorized transfers {args.wait_pids}', flush=True)
        wait_for_pids(args.wait_pids)
    if args.rounds == 2:
        command = [sys.executable, str(Path(__file__).resolve()), '--config', args.config,
                   '--manifest', args.manifest, '--proxy', args.proxy, '--workers', str(args.workers), '--resume']
        for iteration in range(2):
            print(f'Durable recovery round {iteration + 1}/2', flush=True)
            subprocess.run(command, check=False)
            if Path(args.manifest).exists():
                result = json.loads(Path(args.manifest).read_text(encoding='utf8'))
                transient_source_failure = any(s['status'] == 'failed' and '-16' not in s.get('error', '')
                                               and '-17' not in s.get('error', '') for s in result.get('sources', []))
                failed_file = any(f['status'] not in ['downloaded', 'existing_not_overwritten'] for f in result.get('files', []))
                if not transient_source_failure and not failed_file:
                    break
        return
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    manifest = dict(schema_version=1, checked_at='2026-10-01', sources=[], files=[],
                    protocol_evidence=config['protocol_evidence'], verification='MEGA native MAC plus local SHA256')
    jobs = []
    for source in config['sources']:
        entry = dict(source_id=source['id'], url=source['url'], evidence_url=source['evidence_url'])
        try:
            files, dirs = PublicMega(args.proxy).enumerate(source)
            entry.update(status='enumerated', file_count=len(files), expected_bytes=sum(f['size'] for f in files), directories=dirs)
            entry['missing_required_directories'] = [directory for directory in source.get('expected_nonempty_directories', [])
                                                     if not any(f['name'].startswith(directory + '/') for f in files)]
            jobs.extend((source, item) for item in files)
        except Exception as exc:
            entry.update(status='failed', error=str(exc))
        manifest['sources'].append(entry)
        print(json.dumps(entry), flush=True)
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A folder node and an individual share may expose exactly the same
    # authenticated file key. Retain both source associations, transfer once.
    unique, aliases = {}, {}
    for source, item in jobs:
        identity = (item['key'], item['size'])
        if identity in unique:
            canonical = unique[identity][0]['id']
            aliases.setdefault(canonical, []).append(source['id'])
        else:
            unique[identity] = (source, item)
    jobs = list(unique.values())
    manifest['duplicate_source_aliases'] = aliases
    save_manifest(output, manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(PublicMega(args.proxy, args.resume).download, source, item) for source, item in jobs]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            record['also_from_source_ids'] = aliases.get(record['source_id'], [])
            manifest['files'].append(record)
            save_manifest(output, manifest)
    manifest['counts'] = {state: sum(f['status'] == state for f in manifest['files'])
                          for state in sorted({f['status'] for f in manifest['files']})}
    manifest['failure_gate'] = (any(f['status'] not in ['downloaded', 'existing_not_overwritten'] for f in manifest['files'])
                                or any(s['status'] != 'enumerated' or s.get('missing_required_directories') for s in manifest['sources']))
    manifest['complete'] = not manifest['failure_gate']
    manifest['coverage_boundary'] = 'Complete refers only to publicly enumerated files; empty or removed author-share entries are not recovered.'
    save_manifest(output, manifest)
    print(json.dumps(manifest['counts']), flush=True)


if __name__ == '__main__':
    main()
