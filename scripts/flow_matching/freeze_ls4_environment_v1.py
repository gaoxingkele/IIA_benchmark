"""Verify the isolated LS4 dependency binaries and imports before registration."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.flow_matching.spectral_table2_bootstrap_v1 import sha, write

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/acquisition/fm_ls4_resolved_dependencies.v1.json')
    parser.add_argument('--output', default='configs/reproducibility/ls4_author_py310.lock.json')
    args = parser.parse_args()
    registry = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    environment = json.loads((ROOT / registry['environment_config']).read_text(encoding='utf-8'))
    if Path(sys.executable).resolve() != (ROOT / environment['python']).resolve() or sys.version_info[:2] != (3, 10):
        raise ValueError('Isolated original Python3.10 required')
    if (ROOT / args.output).exists():
        raise FileExistsError('Preserve frozen environment identity')
    versions = {}
    for item in registry['sources']:
        if sha(ROOT / item['path']) != item['checksum'].split(':', 1)[1]:
            raise ValueError('Publisher dependency checksum differs')
        versions[item['name']] = importlib.metadata.version(item['name'])
        if versions[item['name']] != item['version']:
            raise ValueError('Original package version differs: ' + item['name'])
    # pip check is evaluated outside the repository, whose package metadata is
    # not part of this author environment. No benchmark numerical deps replace
    # the author's pins; repository functions are imported through PYTHONPATH.
    clean_env = dict(os.environ)
    clean_env.pop('PYTHONPATH', None)
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=sys.prefix,
        env=clean_env, capture_output=True, text=True, encoding='utf-8', check=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg', WANDB_MODE='disabled')
    source = ROOT / 'experiments/runs/fm_spectral_table3_baseline_acquisition_v1/sources/ls4/original'
    sys.path.insert(0, str(source))
    import torch
    import numpy
    import pandas
    import pytorch_lightning
    import wandb
    from omegaconf import OmegaConf
    import datasets
    from models.ls4 import VAE
    import metrics
    assert Path(datasets.__file__).is_relative_to(source)
    assert Path(metrics.__file__).is_relative_to(source)
    assert torch.__version__ == '1.12.1+cu116' and not torch.cuda.is_initialized()
    config = OmegaConf.load(source / 'configs/monash/vae_fred_md.yaml')
    assert config.optim.epochs == 10000
    receipt = dict(passed=True, python=sys.version, python_version=sys.version.split()[0],
        executable=str(Path(sys.executable).resolve()), packages=versions,
        dependency_registry=args.config, dependency_registry_sha256=sha(ROOT / args.config),
        package_receipts=registry['sources'], pip_check=check.stdout.strip(),
        original_dataset_model_metric_imports_passed=True, cuda_context_initialized=False,
        cauchy_backend='Original optional-extension/PyKeOps-free native fallback',
        source_pythonpath=[str(ROOT), str(ROOT / 'src')],
        captured_utc=datetime.now(timezone.utc).isoformat(),
        boundary='Original pinned dependencies retained; unspecified dependency resolutions frozen. Import check is not training or a benchmark result.')
    write(ROOT / args.output, receipt)
    print(json.dumps(dict(passed=True, packages=len(versions), python=receipt['python_version'],
                         original_source_imports=True, cuda_context_initialized=False)))


if __name__ == '__main__':
    main()
