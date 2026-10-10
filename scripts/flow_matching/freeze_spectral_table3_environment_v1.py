"""Freeze the isolated long-series overlay; never alter live parent environments."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha

ROOT = Path(__file__).resolve().parents[2]


def verify_versions(expected, actual, bootstrap_exceptions):
    if set(bootstrap_exceptions) - {'pip'}:
        raise ValueError('Only the explicitly registered pip bootstrap may differ')
    for package, version in expected.items():
        accepted = bootstrap_exceptions.get(package, version)
        if actual.get(package) != accepted:
            raise ValueError('Inherited package differs: ' + package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acquisition/spectral_table3_environment_additions.v1.json')
    parser.add_argument('--output', default='configs/reproducibility/spectral_table3_author_py310.lock.json')
    args = parser.parse_args()
    registry_path = ROOT / args.config
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    parent = ROOT / registry['parent_environment_lock']
    if sha(parent) != registry['parent_environment_sha256']:
        raise ValueError('Frozen parent environment lock differs')
    inherited = json.loads(parent.read_text(encoding='utf-8'))
    if sys.version != inherited['python'] or Path(sys.prefix).resolve() != (ROOT / registry['isolated_environment']).resolve():
        raise ValueError('Isolated Python identity differs')
    actual = {p: importlib.metadata.version(p) for p in inherited['packages']}
    verify_versions(inherited['packages'], actual, registry['bootstrap_exceptions'])
    audio = importlib.metadata.version('torchaudio')
    if audio != '2.8.0+cu126' or actual['torch'] != '2.8.0+cu126':
        raise ValueError('Registered matching official torch/audio binaries required')
    for item in registry['sources']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Registered overlay wheel differs')
    pth = Path(sys.prefix) / 'Lib/site-packages/frozen_table2_parent.pth'
    expected_paths = [ROOT / p for p in registry['parent_site_packages']]
    if [Path(p).resolve() for p in pth.read_text(encoding='utf-8').splitlines()] != [p.resolve() for p in expected_paths]:
        raise ValueError('Frozen inherited package paths differ')
    parent_pth = Path(inherited['overlay_pth'])
    if sha(parent_pth) != inherited['overlay_pth_sha256']:
        raise ValueError('Live parent path binding differs')
    result = dict(python=sys.version, packages=dict(actual, torchaudio=audio),
        inherited_environment_lock=registry['parent_environment_lock'], inherited_environment_sha256=sha(parent),
        parent_site_packages=[str(p) for p in expected_paths],
        overlay_python=str(Path(sys.prefix) / 'Scripts/python.exe'),
        overlay_pth=str(pth), overlay_pth_sha256=sha(pth),
        addition_registry=args.config, addition_registry_sha256=sha(registry_path),
        bootstrap_exceptions={p: dict(inherited=inherited['packages'][p], effective=v)
                              for p, v in registry['bootstrap_exceptions'].items()},
        source_pythonpath=[str(ROOT), str(ROOT / 'src')],
        parent_environment_unchanged=True, inherited_numerical_versions_identical=True,
        wheel_receipts=registry['sources'])
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != result:
            raise ValueError('Preserve existing environment lock')
    else:
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(effective_packages=len(result['packages']),
        inherited_numerical_versions_identical=True,
        numerical_libraries_imported=any(p in sys.modules for p in ('numpy', 'torch', 'tensorflow')),
        lock=args.output)))


if __name__ == '__main__':
    main()
