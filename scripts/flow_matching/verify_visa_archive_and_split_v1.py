"""Read-only independent original archive/extraction/split verification."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def stream_sha(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def bind_member(name, root):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Archive member escapes the registered extraction')
    path = (root / relative.as_posix()).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Archive member resolves outside registered extraction')
    return relative.as_posix(), path


def verify_tar(archive, root, records):
    expected = {r['path']: r for r in records}
    if len(expected) != len(records):
        raise ValueError('Duplicate raw audit identity')
    seen = set()
    with tarfile.open(archive, mode='r|') as source:
        for member in source:
            name, path = bind_member(member.name, root)
            if member.isdir():
                continue
            if not member.isfile() or name in seen or name not in expected:
                raise ValueError('Unexpected or duplicate original archive member')
            seen.add(name)
            record = expected[name]
            if member.size != record['bytes'] or path.stat().st_size != member.size:
                raise ValueError('Original archive/extraction size mismatch')
            with source.extractfile(member) as stream:
                archived_sha = stream_sha(stream)
            if archived_sha != record['sha256'] or file_sha(path) != archived_sha:
                raise ValueError('Original archive/extraction content mismatch')
    if seen != set(expected):
        raise ValueError('Incomplete original archive coverage')
    return len(seen)


def split_audit(rows):
    identities = [row['image'] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Native image identities overlap')
    counts = Counter((r['object'], r['split'], r['label']) for r in rows)
    by_sha = defaultdict(list)
    for row in rows:
        if row['split'] not in ('train', 'test') or row['label'] not in ('normal', 'anomaly'):
            raise ValueError('Unknown original split or label')
        if row['split'] == 'train' and row['label'] != 'normal':
            raise ValueError('Original normal-only training contains anomaly')
        if bool(row['mask']) != (row['label'] == 'anomaly'):
            raise ValueError('Native anomaly mask binding differs')
        by_sha[row['image_sha256']].append(dict(image=row['image'], split=row['split'],
            category=row['object'], label=row['label']))
    duplicates = [dict(sha256=sha, images=items) for sha, items in sorted(by_sha.items()) if len(items) > 1]
    cross_split = [group for group in duplicates if len({r['split'] for r in group['images']}) > 1]
    return dict(per_category_split_counts=[dict(category=c, split=s, label=l, images=n)
        for (c,s,l),n in sorted(counts.items())], duplicate_file_content_groups=duplicates,
        cross_split_identical_file_content_groups=cross_split,
        cross_split_identical_file_content_images=sum(len(g['images']) for g in cross_split),
        duplicate_check_boundary='SHA256 of full original encoded image files; decoded-pixel and near-duplicate similarity not certified.')


def verify(config, root=ROOT):
    audit_path = root / config['full_raw_audit']
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or not audit['all_images_and_masks_decoded_and_verified']:
        raise ValueError('Complete successful original image/mask audit required')
    if file_sha(root / audit['configuration']) != audit['configuration_sha256']:
        raise ValueError('Original acquisition configuration changed')
    for receipt in audit['source_receipts']:
        if file_sha(root / receipt['path']) != receipt['sha256']:
            raise ValueError('Original acquisition source receipt changed')
    archive = root / config['original_archive']
    if archive.stat().st_size != config['expected_archive_bytes']:
        raise ValueError('Full original archive bytes differ')
    sha = file_sha(archive)
    if sha != config['expected_local_archive_sha256'] or sha != audit['complete_raw_archive_sha256']:
        raise ValueError('Full original archive SHA256 changed')
    raw = (root / config['original_extraction']).resolve()
    count = verify_tar(archive, raw, audit['raw_files'])
    actual = {p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}
    if actual != {r['path'] for r in audit['raw_files']}:
        raise ValueError('Unregistered extracted raw file')
    author = (root / config['original_author_directory']).resolve()
    author_zip = root / config['original_author_archive']
    if file_sha(author_zip) != audit['complete_author_archive_sha256']:
        raise ValueError('Original author archive differs')
    expected = {r['path']: r for r in audit['author_files']}; seen = set()
    with zipfile.ZipFile(author_zip) as source:
        if source.testzip() is not None:
            raise ValueError('Original author ZIP CRC failed')
        for member in source.infolist():
            if member.is_dir(): continue
            parts = PurePosixPath(member.filename).parts
            if parts[0] != 'spot-diff-' + config['author_commit']:
                raise ValueError('Original author commit wrapper differs')
            name, path = bind_member('/'.join(parts[1:]), author)
            if name in seen or name not in expected:
                raise ValueError('Unexpected original author file')
            seen.add(name); record = expected[name]
            with source.open(member) as stream: archived_sha = stream_sha(stream)
            if member.file_size != record['bytes'] or archived_sha != record['sha256'] or file_sha(path) != archived_sha:
                raise ValueError('Original author ZIP/extraction mismatch')
    if seen != set(expected): raise ValueError('Incomplete original author archive')
    split = author / config['split_csv']
    if file_sha(split) != audit['original_split_sha256']:
        raise ValueError('Original one-class split CSV changed')
    with split.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source); fields = reader.fieldnames; csv_rows = list(reader)
    rows = audit['native_split_image_manifest']
    if csv_rows != [{k: r[k] for k in fields} for r in rows]:
        raise ValueError('Native original split order or values changed')
    if len(rows) != config['expected_images'] or len({r['object'] for r in rows}) != config['expected_categories']:
        raise ValueError('Full original image/category coverage differs')
    split_checks = split_audit(rows)
    if split_checks['per_category_split_counts'] != audit['per_category_split_counts']:
        raise ValueError('Native per-category counts differ')
    raw_records = {r['path']: r for r in audit['raw_files']}
    for row in rows:
        if raw_records[row['image']]['sha256'] != row['image_sha256']:
            raise ValueError('Original image manifest differs from archive')
        if row['mask'] and raw_records[row['mask']]['sha256'] != row['mask_sha256']:
            raise ValueError('Original mask manifest differs from archive')
    return dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
        original_archive_sha256=sha, publisher_full_file_checksum_available=False,
        original_archive_bytes=archive.stat().st_size, independently_verified_raw_files=count,
        independently_verified_author_files=len(seen), original_images=len(rows),
        anomalous_masks=sum(bool(r['mask']) for r in rows),
        full_raw_audit_sha256=file_sha(audit_path), original_one_class_split_sha256=file_sha(split),
        resolved_raw_directory=str(raw), resolved_author_directory=str(author),
        all_original_archive_members_match_extracted_bytes=True, native_original_split_preserved=True,
        raw_files_written_or_overwritten=0, full_original_training_executed=False,
        new_benchmark_result_rows=0, strict_TAB_equivalence_certified=False, **split_checks)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config', required=True); args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    output=ROOT/config['output_report']
    if output.exists(): raise FileExistsError('Preserve prior complete audit')
    report=verify(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}))


if __name__=='__main__': main()
