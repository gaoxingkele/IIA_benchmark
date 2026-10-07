"""Import user-provided PDF bundles without overwriting sources or other editions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import unicodedata
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]


def normal(text):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKC', text).lower())


def contained(root, value):
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(str(value)).is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe repository-relative path')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path escapes repository')
    return target


def check_member(item):
    name = item.filename.replace('\\', '/')
    relative = PurePosixPath(name)
    if relative.is_absolute() or PureWindowsPath(name).is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe archive member')
    if stat.S_ISLNK(item.external_attr >> 16):
        raise ValueError('Archive symlinks are not supported')
    if not item.is_dir() and (relative.suffix.lower() != '.pdf' or item.file_size > 512 * 1024**2):
        raise ValueError('Unexpected or oversized archive member')


def import_archives(root, config):
    records = []
    for archive in config['archives']:
        source = contained(root, archive['path'])
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != archive['sha256']:
            raise ValueError('Archive SHA256 differs from registration')
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            infos = bundle.infolist()
            for info in infos:
                check_member(info)
            names = [i.filename for i in infos if not i.is_dir()]
            expected = {m['name']: m for m in archive['members']}
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError('Archive member set differs from registration')
            for name in names:
                member = expected[name]
                if not re.fullmatch(r'[a-z][a-z0-9_]*', member['paper_id']):
                    raise ValueError('Unsafe paper identifier')
                raw = bundle.read(name)  # zipfile verifies CRC before returning.
                digest = hashlib.sha256(raw).hexdigest()
                if digest != member['sha256']:
                    raise ValueError('PDF SHA256 differs from registration')
                reader = PdfReader(io.BytesIO(raw))
                selected = member.get('selected_pdf_pages') or list(range(1, len(reader.pages)+1))
                if not selected or len(set(selected)) != len(selected) or any(p < 1 or p > len(reader.pages) for p in selected):
                    raise ValueError('Invalid selected PDF pages')
                first = normal(reader.pages[selected[0]-1].extract_text() or '')
                if any(normal(token) not in first for token in member['title_tokens']):
                    raise ValueError('PDF title does not match registered paper')
                path = contained(root, str(Path(config['output_directory']) / member['paper_id'] / (digest+'.pdf')))
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise FileExistsError('Existing local PDF differs; preserved')
                else:
                    with path.open('xb') as handle:
                        handle.write(raw)
                records.append({'id': member['paper_id'], 'paper_id': member['paper_id'],
                                'archive': archive['path'], 'archive_sha256': archive['sha256'],
                                'archive_member': name, 'path': path.relative_to(root).as_posix(),
                                'status': 'verified_pdf', 'format': 'pdf', 'bytes': len(raw),
                                'sha256': digest, 'pages': len(reader.pages),
                                'selected_pdf_pages': selected, 'title_verified': True,
                                'source': 'user_supplied_archive', 'zip_crc_verified': True,
                                'boundary': 'Source identity/integrity verified; no experiment executed.'})
    return {'schema_version': 1, 'records': records,
            'summary': {'archive_count': len(config['archives']), 'pdf_entries': len(records),
                        'unique_file_hashes': len({r['sha256'] for r in records}),
                        'unique_papers': len({r['paper_id'] for r in records})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/acquisition/fm_user_supplied_papers.v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    report = import_archives(ROOT, config)
    output = contained(ROOT, config['manifest'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
