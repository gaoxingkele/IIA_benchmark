"""Integrate the actual pinned DSB IPF trainer on complete single-cell inputs.

Only model/data/trace/checkpoint boundaries are specialized. The original train,
IPF update, cache refresh, seed schedule and Langevin equations execute normally.
"""
import importlib
import inspect
from pathlib import Path
import sys


def load_native_trainer(source_directory, overlay_directory):
    source = Path(source_directory).resolve()
    overlay = Path(overlay_directory).resolve()
    for path in (overlay, source):
        sys.path.insert(0, str(path))
    import accelerate
    if accelerate.__version__ != '0.2.1' or not Path(accelerate.__file__).resolve().is_relative_to(overlay):
        raise ValueError('Require the independently installed original Accelerate0.2.1 API')
    if 'bridge' in sys.modules:
        module = sys.modules['bridge']
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError('Do not reuse a foreign bridge package')
    native = importlib.import_module('bridge.runners.ipf')
    if Path(inspect.getfile(native.IPFSequential)).resolve() != source / 'bridge/runners/ipf.py':
        raise ValueError('Require the actual pinned original IPFSequential class')
    return native.IPFSequential


def require_full_budget(args):
    expected = dict(n_ipf=20, num_iter=10000, num_steps=20, batch_size=512,
                    cache_npar=10000, num_cache_batches=1, cache_refresh_stride=100,
                    gamma_min=.001, gamma_max=.001, lr=.0001,
                    device='cpu', mean_match=False, ema=False, dataparallel=False,
                    use_prev_net=False, num_workers=0, pin_memory=False,
                    gamma_space='linspace', checkpoint_run=False, plot_npar=1000,
                    gif_stride=10000, log_stride=50, grad_clipping=False)
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError('Preserve registered full native DSB argument: ' + name)
    return 20 * 2 * (10000 + 1)


def runner_class(native_class, components, source_directory, output_directory, earlier, later, seed):
    import json
    import random
    import numpy as np
    import torch
    output = Path(output_directory)

    def repeat(loader):
        while True:
            yield from loader

    class SingleCellNativeIPF(native_class):
        def __init__(self, args):
            require_full_budget(args)
            object.__setattr__(self, '_iia_updates', {'f': 0, 'b': 0})
            object.__setattr__(self, '_iia_phases', [])
            object.__setattr__(self, '_iia_endpoint_loaders', {})
            super().__init__(args)
            half = np.linspace(.001, .001, 10)
            expected_time = torch.tensor(np.concatenate([half, np.flip(half)])).sum()
            if self.gammas_dtype != 'torch.float64' or not torch.equal(self.T.cpu(), expected_time):
                raise ValueError('Keep original NumPy float64 gamma schedule and total time')

        @property
        def gammas_dtype(self):
            # IPFBase constructs T from the NumPy-derived float64 gammas;
            # original Langevin intentionally stores its own float32 schedule.
            return str(self.T.dtype)

        def build_models(self, forward_or_backward=None):
            # Match original construction order, including the unused fresh net.
            f = components.build_score_network(source_directory)
            b = components.build_score_network(source_directory)
            if forward_or_backward is None:
                self.net = torch.nn.ModuleDict({'f': f.to(self.device), 'b': b.to(self.device)})
            else:
                self.net.update({forward_or_backward: (f if forward_or_backward == 'f' else b).to(self.device)})

        def build_optimizers(self):
            super().build_optimizers()
            for role, optimizer in self.optimizer.items():
                original = optimizer.step
                def counted_step(*args, _original=original, _role=role, **kwargs):
                    value = _original(*args, **kwargs)
                    self._iia_updates[_role] += 1
                    return value
                optimizer.step = counted_step

        def build_dataloaders(self):
            self.mean_final = earlier.mean(dim=0).to(self.device)
            self.var_final = earlier.var().to(self.device)
            self.std_final = torch.sqrt(self.var_final)
            values = {}
            for name, data, count, offset in [
                ('cache_init', later, 10000, 0), ('cache_final', earlier, 10000, 1000),
                ('save_init', later, 1000, 2000), ('save_final', earlier, 1000, 3000),
            ]:
                loader = components.endpoint_cache_loader(data, count, seed + offset)
                self._iia_endpoint_loaders[name] = loader
                values[name] = self.accelerator.prepare(loader)
            self.cache_init_dl = repeat(values['cache_init'])
            self.cache_final_dl = repeat(values['cache_final'])
            self.save_init_dl = repeat(values['save_init'])
            self.save_final_dl = repeat(values['save_final'])

        def get_plotter(self):
            # Preserve every original save_step sampling call and seed reset;
            # retain its full numerical trace rather than rasterizing first2PCs.
            directory = output / 'native_saved_traces'
            directory.mkdir(parents=True, exist_ok=False)
            def save_trace(initial, trajectory, iteration, phase, role):
                p = directory / f'phase{phase}_{role}_iteration{iteration}.npz'
                if p.exists():
                    raise FileExistsError('Preserve each original native saved trace')
                np.savez_compressed(p, initial=initial.detach().cpu().numpy(),
                                    trajectory=trajectory.detach().cpu().numpy())
            return save_trace

        def ipf_step(self, forward_or_backward, n):
            before = dict(self._iia_updates)
            # Actual author loop, including all10001updates,per-step seed resets,
            # native Accelerator.backward/step,plots and terminal cache refresh.
            super().ipf_step(forward_or_backward, n)
            expected = self.args.num_iter + 1
            if (self._iia_updates[forward_or_backward] - before[forward_or_backward] != expected
                    or self._iia_updates['f' if forward_or_backward == 'b' else 'b']
                    != before['f' if forward_or_backward == 'b' else 'b']):
                raise ValueError('Every source inclusive-loop optimizer update must execute exactly once')
            if not all(torch.isfinite(p).all() for net in self.net.values() for p in net.parameters()):
                raise FloatingPointError('Preserve actual DSB numerical failure;never suppress nonfinite weights')
            self._iia_phases.append(dict(phase=n, direction=forward_or_backward,
                                        actual_phase_updates=expected, cumulative_updates=dict(self._iia_updates)))
            path = output / f'phase{n}_{forward_or_backward}_full_state.pt'
            if path.exists():
                raise FileExistsError('Preserve every full native phase checkpoint')
            torch.save(dict(networks={key: value.state_dict() for key, value in self.net.items()},
                            optimizers={key: value.state_dict() for key, value in self.optimizer.items()},
                            cumulative_updates=dict(self._iia_updates), phase_records=list(self._iia_phases),
                            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                            endpoint_sampler_rng={k: v.sampler.generator.get_state()
                                                  for k, v in self._iia_endpoint_loaders.items()}), path)
            with (output / 'phase_progress.json').open('w', encoding='utf-8') as f:
                json.dump(dict(phases=self._iia_phases, required_phases=40,
                               actual_updates=sum(self._iia_updates.values()), required_updates=400040,
                               all_original_experiments_complete=False), f, indent=2)

    return SingleCellNativeIPF
