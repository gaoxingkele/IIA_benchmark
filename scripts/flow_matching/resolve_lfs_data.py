"""Resolve registered public Git LFS objects to new, checksum-verified paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from curl_cffi import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'configs/acquisition/flow_matching_fm_data_sources.json'


def main():
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    originals = list(registry['sources'])
    records, additions = [], []
    for source in originals:
        pointer = ROOT / source['path']
        if not pointer.is_file() or pointer.stat().st_size > 256:
            continue
        text = pointer.read_text(encoding='utf-8', errors='replace')
        if not text.startswith('version https://git-lfs.github.com/spec/v1'):
            continue
        expected = re.search(r'oid sha256:([a-f0-9]{64})', text).group(1)
        size = int(re.search(r'size (\d+)', text).group(1))
        url = source['url'].replace('https://raw.githubusercontent.com/', 'https://media.githubusercontent.com/media/')
        relative = Path(source['path']).relative_to('data/public_datasets/flow_matching/cfmi')
        target = ROOT / 'data/public_datasets/flow_matching/cfmi/lfs_resolved' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            payload = target.read_bytes()
        else:
            response = requests.get(url, proxy='http://127.0.0.1:17890', impersonate='chrome', timeout=90)
            response.raise_for_status()
            payload = response.content
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'LFS object identity mismatch: {source["id"]}')
        if not target.exists():
            target.write_bytes(payload)
        with np.load(target, allow_pickle=False) as array:
            shapes = {key: list(array[key].shape) for key in array.files}
        sid = source['id'] + '_lfs_resolved'
        path = target.relative_to(ROOT).as_posix()
        additions.append({**source, 'id': sid, 'url': url, 'path': path, 'size_bytes': size,
                          'checksum': 'sha256:' + expected, 'notes': 'Verified Git LFS object; original pointer preserved at its original path.'})
        records.append({'id': sid, 'path': path, 'status': 'available', 'bytes': size, 'sha256': expected,
                        'integrity_basis': 'Git LFS expected SHA256 and object size', 'array_shapes': shapes})
        print(json.dumps(records[-1]), flush=True)
    ids = {source['id'] for source in registry['sources']}
    registry['sources'].extend(source for source in additions if source['id'] not in ids)
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    path = ROOT / 'papers/literature/flow_matching/lfs_resolved_manifest.json'
    path.write_text(json.dumps({'records': records}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
