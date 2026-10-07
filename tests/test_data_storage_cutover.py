import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows directory-junction cutover')
ROOT = Path(__file__).resolve().parents[1]


def setup_copy(tmp_path):
    root = tmp_path / 'repo'
    source = root / 'data/public_datasets'
    target = tmp_path / 'storage/public_datasets'
    state = tmp_path / 'storage/state'
    source.mkdir(parents=True)
    (source / 'raw.bin').write_bytes(b'original raw data')
    (source / 'empty').mkdir()
    config = root / 'configs/storage/data_storage.v1.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'mounts': [{'physical_path': str(target)}],
                                 'migration_state_directory': str(state),
                                 'migration_report': 'docs/reports/migration.json'}))
    subprocess.run([sys.executable, str(ROOT / 'scripts/data_acquisition/migrate_data_storage.py'),
                    '--source', str(source), '--destination', str(target), '--state', str(state)],
                   check=True, capture_output=True)
    return root, source, target


def cutover(root):
    return subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           str(ROOT / 'scripts/data_acquisition/switch_data_storage.ps1'),
                           '-RepositoryRoot', str(root), '-PythonExecutable', sys.executable], capture_output=True)


def test_cutover_routes_old_paths_and_cleans_only_verified_source(tmp_path):
    root, source, target = setup_copy(tmp_path)
    preserved = root / 'user_book.bin'
    preserved.write_bytes(b'preserve user material')
    result = cutover(root)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert source.resolve() == target.resolve()
    assert (source / 'raw.bin').read_bytes() == b'original raw data'
    assert (target / 'empty').is_dir()
    assert preserved.read_bytes() == b'preserve user material'
    assert not list(source.parent.glob('public_datasets.retired_*'))
    report = json.loads((root / 'docs/reports/migration.json').read_text(encoding='utf-8'))
    assert report['source_copy_removed']
    assert report['sha256_verified_file_count'] == 1


def test_destination_change_blocks_cleanup_and_preserves_both_copies(tmp_path):
    root, source, target = setup_copy(tmp_path)
    (target / 'raw.bin').write_bytes(b'damaged destination')
    result = cutover(root)
    assert result.returncode != 0
    assert source.resolve() != target.resolve()
    assert (source / 'raw.bin').read_bytes() == b'original raw data'
    assert (target / 'raw.bin').read_bytes() == b'damaged destination'
