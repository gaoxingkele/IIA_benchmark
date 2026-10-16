"""Paper-described DSB network and full-particle endpoint loader adapters.

The caller supplies the pinned source directory; all original dynamics and cache
classes execute from those files. No author source is modified.
"""
import importlib.util
from pathlib import Path
import sys


def _load(name, path, package=False):
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != Path(path).resolve():
            raise ValueError('Never mix independently pinned DSB source copies')
        return module
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(Path(path).parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def original_components(source_directory):
    root = Path(source_directory)
    basic = _load('_iia_pinned_DSB_basic_v1', root / 'bridge/models/basic/__init__.py', True)
    langevin = _load('_iia_pinned_DSB_langevin_v1', root / 'bridge/langevin.py')
    cache = _load('_iia_pinned_DSB_cacheloader_v1', root / 'bridge/data/cacheloader.py')
    return basic.ScoreNetwork, langevin.Langevin, cache.CacheLoader


def build_score_network(source_directory, dimensions=5):
    import torch
    if dimensions != 5:
        raise ValueError('This registered single-cell profile requires all five PCs')
    ScoreNetwork, _, _ = original_components(source_directory)
    model = ScoreNetwork(encoder_layers=[16], pos_dim=16,
                         decoder_layers=[64, 64, 64], x_dim=dimensions)
    # Appendix E.4 changes the decoder to three64 SELU layers. The original
    # positional/time encoders remain 16,32 with LeakyReLU.
    model.net.activation_fn = torch.nn.SELU()
    if sum(p.numel() for p in model.parameters()) != 14261:
        raise ValueError('The full paper-described DSB network has14261 parameters')
    return model


def endpoint_cache_loader(observations, particles, seed):
    import torch
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    x = torch.as_tensor(observations, dtype=torch.float32)
    if (x.ndim != 2 or x.shape[1] != 5 or len(x) == 0 or not torch.isfinite(x).all()
            or type(particles) is not int or particles < 1 or type(seed) is not int):
        raise ValueError('Finite complete endpoint population and positive particle budget required')
    # With original drop_last=True, populations below cache_npar=10000 have no
    # batches. Explicit replacement draws keep every source row eligible and
    # preserve all10000 particles; no first-N subset, fake rows or batch shrink.
    data = TensorDataset(x)
    sampler = RandomSampler(data, replacement=True, num_samples=particles,
                            generator=torch.Generator().manual_seed(seed))
    return DataLoader(data, batch_size=particles, sampler=sampler, drop_last=True,
                      num_workers=0, pin_memory=False,
                      generator=torch.Generator().manual_seed(seed + 1))


def endpoint_indices(timepoints, leftout):
    if timepoints not in (4, 5) or not 0 < leftout < timepoints - 1:
        raise ValueError('Complete four/five-timepoint interior leaveout required')
    # Native backward model reverses init->final. Initialize its forward
    # reference at the later observation, so backward sampling starts at the
    # earlier observed endpoint and reaches the future endpoint via midpoint10.
    return dict(source_initial_index=leftout + 1, source_final_index=leftout - 1,
                backward_start_index=leftout - 1, target_index=leftout,
                midpoint_path_index=9, target_excluded_from_training=True)
