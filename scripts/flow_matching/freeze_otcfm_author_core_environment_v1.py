"""Freeze the isolated original-author runtime only after actual native imports."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from scripts.flow_matching.timegan_native_common_v1 import ROOT, read, sha
from scripts.flow_matching.package_otcfm_independent_author_input_v1 import exclusive


def main():
    c = read(ROOT / 'configs/acquisition/fm_otcfm_author_py310_torch_wheels.v1.json')
    if Path(sys.prefix).resolve() != Path(c['environment']).resolve():
        raise ValueError('Never inspect or modify a shared Python environment as this isolated author runtime')
    receipt = read(ROOT / c['state_root'] / 'core_install_receipt.json')
    if not receipt['passed']: raise ValueError('Complete dependency installation and pip check required')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; os.environ['MPLBACKEND'] = 'Agg'; os.environ['WANDB_MODE'] = 'disabled'
    source = ROOT / 'experiments/runs/flow_matching_campaign/sources/ot_cfm/original'
    sys.path.insert(0, str(source / 'runner')); sys.path.insert(1, str(source))
    modules = {name: importlib.import_module(name) for name in ['torch', 'torchvision', 'pytorch_lightning',
        'torchmetrics', 'torchdyn', 'hydra', 'numpy', 'pandas', 'scanpy', 'scprep', 'ot',
        'src.models.cfm_module', 'src.datamodules.distribution_datamodule']}
    expected = dict(torch='1.12.1+cu113', torchvision='0.13.1+cu113', pytorch_lightning='1.8.3.post2',
                    torchmetrics='0.11.0', hydra='1.2.0')
    for name, version in expected.items():
        if modules[name].__version__ != version: raise ValueError('Exact author core version required: ' + name)
    torch = modules['torch']; torch.set_num_threads(2)
    if torch.cuda.is_initialized() or torch.version.cuda != '11.3':
        raise ValueError('Exact author CUDA11.3 build, with no GPU context initialized, is required')
    for name in ['src.models.cfm_module', 'src.datamodules.distribution_datamodule']:
        if not Path(modules[name].__file__).resolve().is_relative_to((source / 'runner').resolve()):
            raise ValueError('Actual original author module, not a substitute package, required')
    cls = importlib.import_module('src.models.components.simple_mlp').VelocityNet
    model = cls(dim=5, hidden_dims=[64, 64, 64], batch_norm=False, activation='selu')
    if sum(p.numel() for p in model.parameters()) != 9093:
        raise ValueError('Actual full5D source architecture must have9093parameters')
    error = None
    try: torch.utils.data.random_split(list(range(20)), [.8, .1, .1], generator=torch.Generator().manual_seed(42))
    except ValueError as exception: error = str(exception)
    if error is None: raise ValueError('Reaudit unexpected fractional split support in the exact old Torch engine')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions(
        path=[str(Path(sys.prefix) / 'Lib/site-packages')]) if d.metadata['Name']}
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True,
        cwd=sys.prefix, env=dict(os.environ, PYTHONPATH=''))
    path = ROOT / c['state_root'] / 'pip_freeze_all.txt'
    if path.exists(): raise FileExistsError('Existing native runtime evidence must remain intact')
    path.write_text(freeze, encoding='utf-8', newline='\n')
    result = dict(python=sys.executable, python_version=platform.python_version(), prefix=sys.prefix,
        native_source_imports_passed=True, original_core_versions=expected,
        packages=packages, cuda_build=torch.version.cuda, cuda_context_initialized=False,
        native_5D_parameter_count=9093, actual_legacy_fractional_split_error=error,
        author_requirements=c['author_requirements'], core_profile_requirements=receipt['requirements'],
        core_profile_requirements_sha256=sha(ROOT / receipt['requirements']),
        pip_freeze_path=str(path), pip_freeze_sha256=sha(path), core_install_receipt_sha256=sha(ROOT / c['state_root'] / 'core_install_receipt.json'),
        modules={k: dict(path=v.__file__, sha256=sha(v.__file__)) for k, v in modules.items()},
        notebook_pandas_scope_excluded_due_to_real_scprep_pandas_conflict=True,
        supporting_dependencies_are_original_historical_lock=False, author_OS_and_hardware_equivalence_certified=False,
        no_model_training_or_benchmark_performance_completed=True, all_original_experiments_complete=False)
    exclusive(ROOT / 'configs/reproducibility/fm_otcfm_author_core_py310.lock.v1.json', result)
    print(json.dumps(dict(native_imports_passed=True, python=platform.python_version(), versions=expected,
        actual_legacy_fractional_split_error=error, trained_models=0)), flush=True)


if __name__ == '__main__': main()
