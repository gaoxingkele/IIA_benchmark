"""Full released SHCL shape/operator audit; untrained smoke, never performance."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import platform
import sys

from scripts.flow_matching import shcl_released_native_v1 as native
from scripts.flow_matching.run_timegan_full45_completion_replay_v1 import exclusive


def environment():
    return dict(python=platform.python_version(), executable=sys.executable,
        packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'psutil')})


def profiles(model):
    commands = native.released_script_commands(model)
    unique = {}
    for command in commands:
        args = native.parse_author_cli(model, command['argv'])
        params = dict(win_size=args.win_size, seq_size=args.seq_size, c_in=args.input_c,
            c_out=args.output_c, d_model=args.d_model, e_layers=args.e_layers,
            fr=args.fr, tr=args.tr, device='cpu')
        key = (args.dataset, args.win_size)
        if key in unique and unique[key]['parameters'] != params:
            raise ValueError('Released profile changes beyond the declared window grid')
        unique[key] = dict(dataset=args.dataset, parameters=params, batch_size=args.batch_size)
    return list(unique.values())


def verify(config):
    for pin in config['source_receipts']:
        if native.sha(native.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen released operator audit changed: ' + pin['path'])
    model = native.read(native.ROOT / config['model_config'])
    native.verify(model)
    reference = native.read(native.ROOT / config['reference_resource_runtime'])
    if (config['settings'] != reference['settings'] or config['resource_lock'] != reference['resource_lock']
            or config['profiles'] != profiles(model) or len(config['profiles']) != 13
            or any(p['batch_size'] != 64 for p in config['profiles'])
            or config['environment'] != environment()
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681,
                original_data_task_records=273)):
        raise ValueError('Every released full batch/profile, pinned actual environment and existing audit caps required')
    return model


def independent_loss(outputs):
    """Independent float64 diagonal Normal KL, plus the native latent MSE."""
    import torch
    total = None
    for branch in ('single', 'pair'):
        xp, xn, mp, lp, mn, ln = outputs[branch]
        p = torch.distributions.Normal(mp.double(), (0.5 * lp.double()).exp())
        q = torch.distributions.Normal(mn.double(), (0.5 * ln.double()).exp())
        value = torch.distributions.kl_divergence(p, q).sum(-1).reshape(xp.shape[:2])
        value = value + ((xp.double() - xn.double()) ** 2).mean(-1)
        total = value if total is None else total + value
    return total


def worker(config, config_path):
    model_config = verify(config)
    import torch
    torch.set_num_threads(config['cpu_threads'])
    records = []
    base = native.ROOT / config['output_root']
    if base.exists():
        raise FileExistsError('Preserve every operator audit attempt')
    for profile in config['profiles']:
        torch.manual_seed(1)
        model, loss = native.build_model(model_config, profile['parameters'])
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        params = profile['parameters']; b = profile['batch_size']; t = params['win_size']
        x = torch.randn(b, t, params['c_in'], generator=torch.Generator().manual_seed(20261012))
        model.eval()
        captured = {'single': [], 'pair': []}
        handles = [getattr(model, branch + '_encoder').register_forward_pre_hook(
            lambda module, inputs, branch=branch: captured[branch].append(inputs[0].detach().clone()))
            for branch in ('single', 'pair')]
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            first = model(x)
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            centered = x - x.mean(1, keepdim=True)
            embedded = model.emb(centered).permute(0, 2, 1)
            embedded = (model.conv1d(embedded.reshape(-1, 1, t)).reshape(-1, params['d_model'], t)
                + embedded).permute(0, 2, 1)
            for branch, keep in (('single', torch.arange(t) % 2 == 0),
                    ('pair', torch.arange(t) // 2 % 2 == 0)):
                for i, mask in enumerate((keep, ~keep)):
                    expected = embedded.clone(); expected[:, ~mask, :] = 0
                    if len(captured[branch]) != 2 or not torch.equal(captured[branch][i], expected):
                        raise ValueError('Native full THM mask differs from independently constructed mask')
            actual_loss = loss(first)
            expected_loss = independent_loss(first)
            tolerance = config['independent_loss_tolerance']
            if actual_loss.shape != (b, t) or not torch.isfinite(actual_loss).all() or not torch.allclose(
                    actual_loss.double(), expected_loss, **tolerance):
                raise ValueError('Native loss differs from frozen independent float64 Gaussian reference')
            second = model(x)
            if all(torch.equal(a, c) for branch in first for a, c in zip(first[branch], second[branch])):
                raise ValueError('Released VAE must retain stochastic evaluation')
            torch.set_rng_state(rng)
            replay = model(x)
            if any(not torch.equal(a, c) for branch in first for a, c in zip(first[branch], replay[branch])):
                raise ValueError('All native output tensors must replay bitwise with the restored CPU RNG')
        model.train(); model.zero_grad(set_to_none=True)
        training_loss = loss(model(x)).mean(); training_loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Every released model parameter requires a finite original backward gradient')
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state.items()):
            raise ValueError('Operator audit must not update any model state')
        torch.manual_seed(1)
        alternate = dict(params, e_layers=7, fr=0.9, tr=0.1, c_out=3)
        other, _ = native.build_model(model_config, alternate)
        if any(not torch.equal(value, other.state_dict()[key]) for key, value in state.items()):
            raise ValueError('Ignored released constructor flags must not change initialized model state')
        row = dict(dataset=profile['dataset'], full_input_shape=list(x.shape),
            parameter_count=sum(p.numel() for p in model.parameters()),
            all12_native_output_tensors_RNG_replayed_bitwise=True,
            inference_remains_stochastic=True, full_THM_mask_tensors_independently_verified=True,
            loss_shape=list(actual_loss.shape), independent_float64_loss_passed=True,
            independent_loss_max_absolute_error=float((actual_loss.double()-expected_loss).abs().max()),
            frozen_loss_tolerance=tolerance, full_batch_training_loss=float(training_loss.detach()),
            all_parameter_gradients_finite=True, unused_constructor_flags_do_not_change_state=True,
            optimizer_updates_executed=0, synthetic_operator_only_not_benchmark_performance=True)
        records.append(row)
        exclusive(base / ('profile_' + str(len(records)).zfill(2) + '.json'), row)
        print(json.dumps(dict(completed_profiles=len(records), required_profiles=13,
            dataset=row['dataset'], shape=row['full_input_shape'])), flush=True)
        del model, other, first, second, replay, gradients, state, captured, x, actual_loss, expected_loss, training_loss
        gc.collect()
    exclusive(base / 'full_released_operator_audit.json', dict(passed=True,
        captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
        configuration_sha256=native.sha(native.ROOT / config_path), environment=environment(),
        original_scope=config['original_scope'], original_training_commands=49, original_test_commands=364,
        full_operator_profiles=13, records=records, original_author_source_modified=False,
        optimizer_updates_executed=0, original_training_executed=False,
        synthetic_operator_only_not_benchmark_performance=True, original_environment_equivalence=False,
        paper_performance_reproduced=False, all_original_experiments_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(); config = native.read(native.ROOT / args.config)
    verify(config)
    if args.probe_only:
        print(json.dumps(dict(full_profiles=13, batch_size=64, torch_imported='torch' in sys.modules,
            original_training_executed=False)))
    elif args.worker:
        worker(config, args.config)
    else:
        raise ValueError('Use the separately guarded audit controller')


if __name__ == '__main__':
    main()
