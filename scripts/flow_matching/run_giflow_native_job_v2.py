"""Run unabridged GiFlow author functions and save actual evaluated weights/data."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import itertools
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path

from scripts.flow_matching.giflow_native_protocol import ROOT, configured_loader, fingerprint, path_only_loader, sha, verify, write_json


def audit_data(dm, edges, weights, nodes, output):
    import numpy as np
    arrays = {'input_mask': dm.torch_dataset.mask.cpu().numpy(),
              'eval_mask': dm.torch_dataset.eval_mask.cpu().numpy(),
              'edges': edges.cpu().numpy(), 'weights': weights.cpu().numpy()}
    arrays['scaler_bias'] = np.asarray(dm.scalers['target'].bias)
    arrays['scaler_scale'] = np.asarray(dm.scalers['target'].scale)
    for split in ('train', 'val', 'test'):
        arrays[split + '_window_ids'] = np.asarray(getattr(dm, split + 'set').indices)
        arrays[split + '_timestamps'] = dm.torch_dataset.expand_indices(
            getattr(dm, split + 'set').indices, merge=True).cpu().numpy()
    if np.any(arrays['input_mask'] & arrays['eval_mask']):
        raise ValueError('Targets present in conditioning mask')
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if np.intersect1d(arrays[a + '_timestamps'], arrays[b + '_timestamps']).size:
            raise ValueError('Original split has overlapping timestamps')
    np.savez_compressed(output / 'split_mask_graph_scaler.npz', **arrays)
    report = {'nodes': nodes, 'windows': {s: len(arrays[s + '_window_ids']) for s in ('train', 'val', 'test')},
              'timestamp_counts': {s: len(arrays[s + '_timestamps']) for s in ('train', 'val', 'test')},
              'splitter': 'released TemporalSplitter', 'validation_fraction_of_non_test': .1,
              'scaler_fitted_on': 'train timestamps and input mask only (native TSL datamodule)',
              'evaluation_targets_in_input': 0, 'timestamp_intersections': 0,
              'window_overlap_within_split': True, 'native_eval_mask_count': int(arrays['eval_mask'].sum()),
              'audit_sha256': sha(output / 'split_mask_graph_scaler.npz')}
    write_json(output / 'data_binding.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--audit-output')
    parser.add_argument('--integration-check', action='store_true')
    cli = parser.parse_args()
    queue_path = ROOT / cli.queue
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    verify(queue)
    job = next(j for j in queue['jobs'] if j['id'] == cli.job_id)
    diagnostic = cli.audit_only or cli.integration_check
    if diagnostic and not cli.audit_output:
        raise ValueError('Diagnostic requires a separate output; cannot occupy a formal result slot')
    if cli.audit_only and cli.integration_check:
        raise ValueError('Select one diagnostic mode')
    output = ROOT / (cli.audit_output if diagnostic else job['output_directory'])
    if output.exists():
        raise ValueError('Preserve existing output; do not overwrite or implicitly restart')
    output.mkdir(parents=True)
    environment_versions = {}
    for line in (ROOT / queue['environment_lock']).read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if '==' not in line:
            raise ValueError('Environment lock requires review: ' + line)
        package, expected = line.strip().split('==', 1)
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f'Author environment differs: {package} {actual} versus {expected}')
        environment_versions[package] = actual
    import numpy as np
    import torch
    source = ROOT / job['author_source']
    sys.path.insert(0, str(source))
    native = importlib.import_module('main')
    data_module = importlib.import_module('data')
    config = json.loads((ROOT / queue['data_config']).read_text(encoding='utf-8'))
    data_root = ROOT / config['output_root']
    if job['track'] == 'released_mirror':
        loader, adaptation = path_only_loader(data_module, data_root)
    else:
        loader = configured_loader(data_module, data_root)
        adaptation = {'kind': 'reviewed data-root keyword', 'source_sha256': sha(source / 'data.py')}
    context = {'batches': [], 'model': None, 'optimizer_steps': 0, 'test': None}
    args = argparse.Namespace(**job['arguments'], save_dir=str(output / 'checkpoints'), data_root=str(data_root))
    if cli.integration_check:
        if job['dataset'] != 'air36':
            raise ValueError('CPU plumbing check is restricted to real Air36; not a native-budget proof')
        args.batch_size, args.training_epoch, args.device, args.cuda = 2, 1, 'cpu', False
        args.ema_start_epoch = 0  # Exercise actual native EMA in the reduced plumbing check.
        torch.cuda.is_available = lambda: False  # Isolated diagnostic process, never the formal branch.

    class DiagnosticData:
        def __init__(self, dm):
            self.dm = dm
        def __getattr__(self, name):
            if name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
                return lambda **kw: list(itertools.islice(getattr(self.dm, name)(**kw), 1))
            return getattr(self.dm, name)

    def loading(**kwargs):
        dm, edges, weights, nodes, dimensions = loader(**kwargs)
        context['native_data_module'] = dm
        context['data_binding'] = audit_data(dm, edges, weights, nodes, output)
        context['data_adaptation'] = adaptation
        return DiagnosticData(dm) if cli.integration_check else dm, edges, weights, nodes, dimensions

    native.dataset_loading = loading
    if cli.audit_only:
        keys = ('dataset_name', 'missing_rate', 'missing_type', 'window', 'stride', 'adj_threshold',
                'val_len', 'test_len', 'seed', 'batch_size')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        loading(**{k: getattr(args, k) for k in keys})
        actual_loader = context['native_data_module'].train_dataloader(batch_size=args.batch_size, shuffle=False)
        actual_loader_record = dict(samples=len(actual_loader.dataset), batches=len(actual_loader),
                                    batch_size=actual_loader.batch_size, drop_last=actual_loader.drop_last)
        verify(queue)
        write_json(output / 'audit_result.json', dict(context['data_binding'], diagnostic=True,
                   status='data_audit_passed_not_training', experiment_sha256=fingerprint(job), adaptation=adaptation,
                   observed_native_train_loader=actual_loader_record))
        print(json.dumps(context['data_binding']))
        return
    if not cli.integration_check and not torch.cuda.is_available():
        raise RuntimeError('Frozen full GPU job requires CUDA; no budget/device fallback')
    artifact_root = output / 'diagnostic_predictions' if cli.integration_check else Path(queue['prediction_artifact_root']) / job['id']
    artifact_root.mkdir(parents=True, exist_ok=False)
    original_model = native.models.FlowMatching

    def factory(arguments):
        model = original_model(arguments)
        context['model'] = model
        def capture(module, inputs):
            if context['test'] is not None and not (output / 'evaluated_model.pt').exists():
                torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, output / 'evaluated_model.pt')
        model.register_forward_pre_hook(capture)
        return model

    native.models.FlowMatching = factory
    original_optimizer = torch.optim.AdamW
    context['epoch_optimizer_steps'] = {}
    def native_frame():
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code is native.run_experiment.__code__:
                return frame
            frame = frame.f_back
        return None
    original_iterator = torch.utils.data.DataLoader.__iter__
    def observed_iterator(data_loader):
        frame = native_frame()
        if frame and frame.f_locals.get('train_loader') is data_loader:
            record = dict(samples=len(data_loader.dataset), batches=len(data_loader),
                          batch_size=data_loader.batch_size, drop_last=data_loader.drop_last)
            if record['samples'] != context['data_binding']['windows']['train']:
                raise ValueError('Observed original full train loader sample identity differs')
            context['native_train_loader'] = record
            write_json(output / 'training_observation.json', dict(native_train_loader=record,
                       optimizer_steps=context['optimizer_steps'], epoch_optimizer_steps=context['epoch_optimizer_steps']))
        return original_iterator(data_loader)
    torch.utils.data.DataLoader.__iter__ = observed_iterator

    def optimizer(*a, **kw):
        value = original_optimizer(*a, **kw)
        def step_count(*ignored):
            context['optimizer_steps'] += 1
            frame = native_frame()
            if frame is not None:
                epoch = str(int(frame.f_locals['epoch']))
                context['epoch_optimizer_steps'][epoch] = context['epoch_optimizer_steps'].get(epoch,0) + 1
            if 'native_train_loader' in context and context['optimizer_steps'] % context['native_train_loader']['batches'] == 0:
                write_json(output / 'training_observation.json', dict(native_train_loader=context['native_train_loader'],
                           optimizer_steps=context['optimizer_steps'], epoch_optimizer_steps=context['epoch_optimizer_steps']))
        value.register_step_post_hook(step_count)
        return value

    torch.optim.AdamW = optimizer
    original_test = native.preprocess_fm_test
    patched_scalers = set()

    def preprocessing(batch, *a):
        result = original_test(batch, *a)
        context['test'] = {'target': result[2], 'mask': result[4], 'recorded': False}
        scaler_class = type(batch.transform['y'])
        if scaler_class not in patched_scalers:
            original_inverse = scaler_class.inverse_transform
            def inverse(scaler, prediction, *rest, **kw):
                unscaled = original_inverse(scaler, prediction, *rest, **kw)
                state = context['test']
                if state is not None and not state['recorded']:
                    state['recorded'] = True
                    mask = state['mask']
                    pred = unscaled[mask].detach().cpu().numpy()
                    target = state['target'][mask].detach().cpu().numpy()
                    if not len(pred) or not np.isfinite(pred).all() or not np.isfinite(target).all():
                        raise ValueError('Invalid full test target/prediction')
                    path = artifact_root / f"batch_{len(context['batches']):05d}.npz"
                    np.savez_compressed(path, prediction=pred, target=target,
                                        masked_flat_positions=np.flatnonzero(mask.cpu().numpy()),
                                        batch_shape=np.asarray(mask.shape))
                    context['batches'].append({'path': str(path), 'sha256': sha(path), 'targets': len(pred)})
                return unscaled
            scaler_class.inverse_transform = inverse
            patched_scalers.add(scaler_class)
        return result

    native.preprocess_fm_test = preprocessing
    torch.set_num_threads(queue['settings']['cpu_threads'])
    if not cli.integration_check:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    log_path = output / 'native.stdout.log'
    with log_path.open('x', encoding='utf-8') as stream:
        previous = sys.stdout
        sys.stdout = native.Tee(previous, stream)
        try:
            native_metrics = tuple(float(v) for v in native.run_experiment(args))
        finally:
            sys.stdout = previous
    elapsed = time.perf_counter() - started
    if not all(math.isfinite(v) for v in native_metrics) or not context['batches']:
        raise ValueError('Full native evaluation did not produce finite metrics')
    selected_weights_match = None
    if job['track'] == 'reviewed_corrected':
        checkpoints = list((output / 'checkpoints').rglob('*.pt'))
        if len(checkpoints) != 1:
            raise ValueError('Validation-selected checkpoint missing or ambiguous')
        selected = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
        tested = torch.load(output / 'evaluated_model.pt', map_location='cpu', weights_only=True)
        selected_weights_match = set(selected) == set(tested) and all(torch.equal(selected[k], tested[k]) for k in selected)
        if not selected_weights_match:
            raise ValueError('Actually tested weights differ from validation-selected checkpoint')
    if cli.integration_check:
        verify(queue)
        item = np.load(context['batches'][0]['path'])
        delta = item['prediction'].astype('float64') - item['target'].astype('float64')
        assert math.isclose(float(np.abs(delta).mean()), native_metrics[0], rel_tol=1e-5)
        assert math.isclose(float((delta**2).mean()), native_metrics[1], rel_tol=1e-5)
        if context['optimizer_steps'] != 1 or len(context['batches']) != 1 or not (output / 'evaluated_model.pt').exists():
            raise ValueError('Real author integration did not capture one actual update and actual tested weights')
        value = {'status': 'integration_passed_not_benchmark', 'diagnostic': True, 'track': job['track'],
                 'cpu_batch': 2, 'epochs': 1, 'optimizer_updates': context['optimizer_steps'],
                 'diagnostic_ema_start_epoch': 0,
                 'euler_steps': 20, 'test_batches': 1, 'prediction_receipts': context['batches'],
                 'saved_test_weights_sha256': sha(output / 'evaluated_model.pt'),
                 'native_mae': native_metrics[0], 'native_mse': native_metrics[1],
                 'captured_predictions_match_native_errors': True, 'no_formal_result_published': True,
                 'tested_weights_match_validation_checkpoint': selected_weights_match,
                 'boundary': 'Real Air36 plumbing only; reduced diagnostic budget never satisfies a registered full experiment.'}
        write_json(output / 'integration_result.json', value)
        print(json.dumps(value))
        return
    if len(context['batches']) != math.ceil(context['data_binding']['windows']['test'] / args.batch_size):
        raise ValueError('Native full test coverage incomplete')
    verify(queue)  # Raw inputs and pinned sources must remain unchanged.
    paths = [output / 'split_mask_graph_scaler.npz', output / 'data_binding.json',
             output / 'evaluated_model.pt', log_path] + sorted((output / 'checkpoints').rglob('*.pt'))
    artifacts = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths] + context['batches']
    epochs = len(re.findall(r'train_loss=.* after epoch \d+', log_path.read_text(encoding='utf-8')))
    from scripts.flow_matching.giflow_update_budget_v2 import validate_observed_updates
    validate_observed_updates(epochs, args.training_epoch, context['optimizer_steps'],
                              context['native_train_loader'], context['epoch_optimizer_steps'])
    paths.append(output / 'training_observation.json')
    artifacts.append({'path': (output / 'training_observation.json').relative_to(ROOT).as_posix(),
                      'sha256': sha(output / 'training_observation.json')})
    if not epochs:
        raise ValueError('Actual full training update count differs from native budget')
    result = {'status': 'completed', 'diagnostic': False, 'experiment_sha256': fingerprint(job),
              'queue_sha256': sha(queue_path), 'job': job, 'data_binding': context['data_binding'],
              'data_adaptation': context['data_adaptation'], 'metrics': {'native_mean_batch_mae': native_metrics[0],
              'native_mean_batch_mse': native_metrics[1], 'native_mean_batch_mape_percent': native_metrics[2],
              'sqrt_native_mean_batch_mse': math.sqrt(native_metrics[1])},
              'metric_unit': 'masked positions in overlapping test windows; native unweighted mean of batch means',
              'selection': {'test_used_in_validation': job['track'] == 'released_mirror',
                            'tested_weights_match_validation_checkpoint': selected_weights_match,
                            'evaluated_weights': 'final EMA with per-batch reset' if job['track'] == 'released_mirror' else 'validation-selected checkpoint with persistent EMA',
                            'actually_evaluated_weights_saved': 'evaluated_model.pt'},
              'epochs_completed': epochs, 'optimizer_updates': context['optimizer_steps'],
              'native_train_loader': context['native_train_loader'], 'epoch_optimizer_steps': context['epoch_optimizer_steps'],
              'parameters': sum(p.numel() for p in context['model'].parameters()), 'total_native_seconds': elapsed,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'full_test_batches': len(context['batches']),
              'test_targets_with_window_repetitions': sum(b['targets'] for b in context['batches']),
              'environment': {'python': sys.version, 'torch': torch.__version__,
                              'packages': environment_versions, 'environment_lock_sha256': sha(ROOT / queue['environment_lock'])},
              'paper_equivalence': False, 'boundary': queue['boundary'], 'artifacts': artifacts}
    write_json(output / 'result.json', result)
    print(json.dumps(result['metrics']))


if __name__ == '__main__':
    main()
