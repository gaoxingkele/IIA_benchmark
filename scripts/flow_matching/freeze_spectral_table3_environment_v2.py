"""Freeze a separate config-parser overlay without changing numerical packages."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from scripts.flow_matching.freeze_spectral_table3_environment_v1 import ROOT, verify_versions
from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha


def verify_additions(inherited, actual, additions):
    if set(additions) != {'omegaconf', 'antlr4-python3-runtime'} or set(additions) & set(inherited):
        raise ValueError('Only the missing isolated config-parser dependencies may be added')
    verify_versions(inherited, actual, {})
    for package, version in additions.items():
        if actual.get(package) != version:
            raise ValueError('Frozen parser dependency differs: ' + package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    registry_path = ROOT / args.config
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    parent = ROOT / registry['parent_environment_lock']
    if sha(parent) != registry['parent_environment_sha256']:
        raise ValueError('Frozen inherited environment changed')
    inherited = json.loads(parent.read_text(encoding='utf-8'))
    if sys.version != inherited['python'] or Path(sys.prefix).resolve() != (ROOT / registry['isolated_environment']).resolve():
        raise ValueError('A new isolated overlay with the original Python is required')
    additions = {item['name']: item['version'] for item in registry['sources']}
    actual = {name: importlib.metadata.version(name) for name in set(inherited['packages']) | set(additions)}
    verify_additions(inherited['packages'], actual, additions)
    for item in registry['sources']:
        if sha(ROOT / item['path']) != item['checksum'].split(':', 1)[1]:
            raise ValueError('Previously registered local archive checksum differs')
    pth = Path(sys.prefix) / 'Lib/site-packages/frozen_table3_parent.pth'
    expected = [str((ROOT / path).resolve()) for path in registry['parent_site_packages']]
    if pth.read_text(encoding='utf-8').splitlines() != expected:
        raise ValueError('Only the frozen parent package roots may be inherited')
    if sha(inherited['overlay_pth']) != inherited['overlay_pth_sha256']:
        raise ValueError('The old isolated environment must remain unchanged')
    import antlr4
    import omegaconf
    own_site = (Path(sys.prefix) / 'Lib/site-packages').resolve()
    if any(not Path(module.__file__).resolve().is_relative_to(own_site) for module in (antlr4, omegaconf)):
        raise ValueError('New parser packages must resolve only inside the new overlay')
    result = dict(python=sys.version, packages=actual,
        inherited_environment_lock=registry['parent_environment_lock'], inherited_environment_sha256=sha(parent),
        parent_site_packages=expected, overlay_python=str(Path(sys.prefix) / 'Scripts/python.exe'),
        overlay_pth=str(pth), overlay_pth_sha256=sha(pth),
        addition_registry=args.config, addition_registry_sha256=sha(registry_path),
        source_pythonpath=[str(ROOT), str(ROOT / 'src')], inherited_numerical_versions_identical=True,
        parent_environment_unchanged=True, archive_receipts=registry['sources'],
        new_parser_package_roots={module.__name__: str(Path(module.__file__).resolve()) for module in (antlr4, omegaconf)},
        boundary='Pure config parsing dependencies added offline to a new overlay. No inherited numerical version, source, model budget, original YAML or dataset changed. Historical OmegaConf version is not supplied by the release; it remains an author alignment gap.')
    destination = ROOT / args.output
    if destination.exists():
        raise FileExistsError('Preserve all frozen environment registrations')
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(packages=len(actual), parser_imports_passed=True,
        numerical_libraries_imported=any(name in sys.modules for name in ['torch', 'numpy', 'tensorflow']),
        inherited_numerical_versions_identical=True, lock=args.output)))


if __name__ == '__main__':
    main()
