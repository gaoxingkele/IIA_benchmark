"""Copy raw data to new storage with SHA256 verification before any cutover.

This command never removes source data. Cutover and source cleanup are separate
operations after the stable-tree audit passes. State permits interrupted copies
to resume without overwriting unrelated destination files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolved(path):
    value = path.resolve()
    spelling = str(value)
    # Concurrent Windows directory creation can make resolve() retain the
    # extended-length prefix. It names the same path, not an escaped root.
    if os.name == 'nt' and spelling.startswith('\\\\?\\'):
        spelling = spelling[4:]
        if spelling.startswith('UNC\\'):
            spelling = '\\\\' + spelling[4:]
        value = Path(spelling)
    return value


def inside(root, relative):
    target = (root / relative).absolute()
    if not resolved(target).is_relative_to(resolved(root)):
        raise ValueError('Data path escapes migration root')
    return target


def signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def copy_verified(source, destination, relative, previous=None, progress=None):
    original = inside(source, relative)
    target = inside(destination, relative)
    before = signature(original)
    if previous and tuple(previous['source_signature']) == before and target.is_file():
        if signature(target) == tuple(previous['destination_signature']):
            return {**previous, 'reused_verified_copy': True}
    if target.exists():
        source_hash, target_hash = sha256(original), sha256(target)
        if before != signature(original):
            raise ValueError('Source changed while hashing: ' + relative)
        if source_hash != target_hash:
            # Neither an old source version nor unrelated existing data may be
            # overwritten silently. Retain both and leave cutover unfinished.
            raise FileExistsError('Destination differs; both originals preserved: ' + relative)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.iia_copying_' + uuid.uuid4().hex)
        if not resolved(temporary).is_relative_to(resolved(destination)):
            raise ValueError('Temporary path escapes migration root')
        digest = hashlib.sha256()
        with original.open('rb') as reader, temporary.open('xb') as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                writer.write(block)
                digest.update(block)
                if progress:
                    progress(len(block))
            writer.flush()
            os.fsync(writer.fileno())
        if before != signature(original):
            raise ValueError('Source changed during copy; temporary retained: ' + relative)
        source_hash = digest.hexdigest()
        target_hash = sha256(temporary)
        if source_hash != target_hash or temporary.stat().st_size != before[0]:
            raise ValueError('Copied data checksum mismatch; source preserved: ' + relative)
        shutil.copystat(original, temporary)
        # A destination appearing concurrently must not be replaced.
        if target.exists():
            raise FileExistsError('Destination appeared during migration: ' + relative)
        temporary.rename(target)
    return {'path': relative, 'bytes': before[0], 'sha256': source_hash,
            'source_signature': list(before), 'destination_signature': list(signature(target)),
            'verified_at': datetime.now(timezone.utc).isoformat(), 'reused_verified_copy': False}


def inventory(root):
    records = {}
    def fail_scan(error):
        raise error

    for directory, folders, files in os.walk(root, followlinks=False, onerror=fail_scan):
        for name in folders + files:
            path = Path(directory) / name
            stat = path.stat(follow_symlinks=False)
            if path.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
                raise ValueError('Reparse point requires separate migration: ' + str(path))
        for name in files:
            path = Path(directory) / name
            records[path.relative_to(root).as_posix()] = signature(path)
    return records


def audit_stable_tree(source, destination, records):
    live = inventory(source)
    missing, changed = [], []
    for relative, current in live.items():
        item = records.get(relative)
        target = inside(destination, relative)
        if item is None or not target.is_file():
            missing.append(relative)
        elif current != tuple(item['source_signature']) or signature(target) != tuple(item['destination_signature']):
            changed.append(relative)
    extra = sorted(set(records) - set(live))
    pending = [path.relative_to(destination).as_posix()
               for path in destination.rglob('*.iia_copying_*')
               if path.relative_to(destination).as_posix() not in live]
    return {'complete': not (missing or changed or extra or pending), 'file_count': len(live),
            'bytes': sum(x[0] for x in live.values()), 'missing': missing,
            'changed': changed, 'source_files_removed_during_copy': extra,
            'uncommitted_copy_files': pending,
            'sha256_verified_files': sum(p in records for p in live)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    source, destination, state = (resolved(p) for p in (args.source, args.destination, args.state))
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and destination must be separate, non-nested directories')
    if not source.is_dir():
        raise FileNotFoundError(source)
    state.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    records_path = state / 'verified_files.jsonl'
    report_path = state / 'status.json'
    records = {}
    if records_path.exists():
        for line in records_path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            records[record['path']] = record
    live = inventory(source)
    required = sum(size for relative, (size, _) in live.items() if not inside(destination, relative).exists())
    if shutil.disk_usage(destination).free < required + 2 * 1024**3:
        raise ValueError('Insufficient free destination space for verified copy')
    stats = {'source': str(source), 'destination': str(destination), 'total_files': len(live),
             'total_bytes': sum(x[0] for x in live.values()), 'copied_bytes_this_run': 0,
             'verified_files': 0, 'verified_bytes': 0, 'started_at': datetime.now(timezone.utc).isoformat()}
    lock = threading.Lock()
    stopped = threading.Event()

    def save(status):
        with lock:
            snapshot = {**stats, 'status': status, 'updated_at': datetime.now(timezone.utc).isoformat()}
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2)+'\n', encoding='utf-8')
        temporary.replace(report_path)
        print(json.dumps(snapshot), flush=True)

    def progress(size):
        with lock:
            stats['copied_bytes_this_run'] += size

    def heartbeat():
        while not stopped.wait(30):
            save('copying_and_sha256_verifying')

    errors = []
    if not args.audit_only:
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        save('copying_and_sha256_verifying')
        try:
            with records_path.open('a', encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(copy_verified, source, destination, relative, records.get(relative), progress): relative
                           for relative in sorted(live, key=lambda p: live[p][0], reverse=True)}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        records[record['path']] = record
                        log.write(json.dumps(record)+'\n')
                        log.flush()
                        with lock:
                            stats['verified_files'] += 1
                            stats['verified_bytes'] += record['bytes']
                    except Exception as error:
                        errors.append({'path': futures[future], 'error': str(error)})
                        print(json.dumps(errors[-1]), flush=True)
        finally:
            stopped.set()
            thread.join()
    # Empty directories also belong to the preserved data tree.
    directories = [Path(directory) for directory, _, _ in os.walk(source)]
    for directory in directories:
        (destination / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
    for directory in reversed(directories):
        shutil.copystat(directory, destination / directory.relative_to(source))
    audit = audit_stable_tree(source, destination, records)
    result = {**stats, 'status': 'verified_ready_for_cutover' if audit['complete'] and not errors else 'incomplete',
              'audit': audit, 'errors': errors, 'source_removed': False,
              'completed_at': datetime.now(timezone.utc).isoformat()}
    report_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    return int(not audit['complete'] or bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
