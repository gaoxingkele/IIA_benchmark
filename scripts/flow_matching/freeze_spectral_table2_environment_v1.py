"""Freeze the isolated Table2 overlay without importing numerical libraries."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acquisition/spectral_table2_environment_additions.v1.json')
    parser.add_argument('--output', default='configs/reproducibility/spectral_table2_author_py310.lock.json')
    args = parser.parse_args()
    registry = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    parent = ROOT / registry['parent_environment_lock']
    inherited = json.loads(parent.read_text(encoding='utf-8'))
    for package, expected in inherited['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Inherited package differs: ' + package)
    if sys.version != inherited['python'] or Path(sys.prefix).resolve() != (ROOT / registry['isolated_environment']).resolve():
        raise ValueError('Isolated Python environment differs')
    for item in registry['sources']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Registered overlay wheel differs')
    if importlib.metadata.version('tf-slim') != '1.1.0':
        raise ValueError('TF-Slim version differs')
    pth = Path(sys.prefix) / 'Lib/site-packages/frozen_table1_parent.pth'
    parent_packages = ROOT / '.venv/spectral-author-py310/Lib/site-packages'
    if Path(pth.read_text(encoding='utf-8').strip()).resolve() != parent_packages.resolve():
        raise ValueError('Frozen inherited package path differs')
    result = dict(python=sys.version, packages=dict(inherited['packages'], **{'tf-slim': '1.1.0'}),
        inherited_environment_lock=registry['parent_environment_lock'], inherited_environment_sha256=sha(parent),
        parent_site_packages=str(parent_packages), overlay_python=str(Path(sys.prefix) / 'Scripts/python.exe'),
        overlay_pth=str(pth), overlay_pth_sha256=sha(pth),
        source_pythonpath=[str(ROOT), str(ROOT / 'src')],
        addition_registry=args.config, addition_registry_sha256=sha(ROOT / args.config),
        parent_environment_unchanged=True, wheel_receipts=registry['sources'])
    destination = ROOT / args.output
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != result:
            raise ValueError('Existing environment lock differs; preserve it')
    else:
        destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(effective_packages=len(result['packages']), parent_versions_identical=True,
        numerical_libraries_imported=any(p in sys.modules for p in ('numpy','torch','tensorflow')), lock=args.output)))


if __name__ == '__main__':
    main()
