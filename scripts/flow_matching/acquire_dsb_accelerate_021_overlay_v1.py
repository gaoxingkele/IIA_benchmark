"""Install the pinned original DSB Accelerate API into a fresh separate target."""
import json
import os
from pathlib import Path
import subprocess
import zipfile
from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive

CONFIG = 'configs/acquisition/fm_dsb_accelerate_021_overlay.v1.json'


def acquire(c):
    base = Path(c['output_directory'])
    state = ROOT / c['state_root']
    base.mkdir(parents=True, exist_ok=False)
    wheel = base / c['filename']
    command = [c['aria2c'], '--all-proxy=' + c['proxy'], '--allow-overwrite=false',
               '--auto-file-renaming=false', '--continue=false', '--max-tries=3',
               '--connect-timeout=20', '--timeout=30', '--retry-wait=2',
               '--dir=' + str(base), '--out=' + c['filename'], c['url']]
    with (state / 'aria2c.stdout.log').open('xb') as out, (state / 'aria2c.stderr.log').open('xb') as err:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if wheel.stat().st_size != c['bytes'] or sha(wheel) != c['sha256']:
        raise ValueError('Official PyPI wheel SHA256/length mismatch')
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError('Original wheel ZIP CRC failed')
    target = base / 'site-packages'
    command = [c['python'], '-B', '-m', 'pip', 'install', '--no-deps', '--no-compile',
               '--disable-pip-version-check', '--no-index', '--target=' + str(target), str(wheel)]
    with (state / 'pip.stdout.log').open('xb') as out, (state / 'pip.stderr.log').open('xb') as err:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    records = [dict(path=p.relative_to(target).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(target.rglob('*')) if p.is_file()]
    receipt = dict(passed=True, package='accelerate', version='0.2.1', official_PyPI_sha256_verified=True,
                   wheel=str(wheel), sha256=sha(wheel), bytes=wheel.stat().st_size,
                   isolated_import_directory=str(target), installed_files=records,
                   base_environment_modified=False, dependency_upgrades_performed=False,
                   full_historical_environment_certified=False, native_source_import_probe_pending=True,
                   configuration_sha256=sha(ROOT / CONFIG), all_original_experiments_complete=False)
    exclusive(base / 'verified_installation.json', receipt)
    exclusive(state / 'verified_installation.json', receipt)
    return receipt


if __name__ == '__main__':
    result = acquire(read(ROOT / CONFIG))
    print(json.dumps({k: result[k] for k in ['passed', 'version', 'isolated_import_directory', 'base_environment_modified']}))
