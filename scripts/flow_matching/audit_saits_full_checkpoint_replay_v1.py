"""Replay completed original SAITS/BRITS models on every native test window."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, sha, read, write, capped_wait
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def verify_registration(config):
    for item in config['source_receipts']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen full SAITS checkpoint replay source changed')
    ids = [r['id'] for r in config['completed_jobs']]
    if len(ids) != config['full_model_count'] or len(ids) != len(set(ids)):
        raise ValueError('Exactly the registered complete models are required')


def compare_metrics(actual, recorded, tolerance):
    import math
    for key in ('mae', 'rmse', 'mre'):
        if not math.isfinite(actual[key]) or not math.isclose(
                actual[key], recorded[key], rel_tol=tolerance['rtol'], abs_tol=tolerance['atol']):
            raise ValueError('Full original checkpoint replay metric differs: ' + key)


def worker(config, config_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from scripts.flow_matching.run_saits import configure, aggregate_groups
    from scripts.flow_matching.validate_saits_full_result_v1 import validate
    torch.set_num_threads(config['cpu_threads'])
    target = ROOT / config['artifact_directory']
    if target.exists():
        raise FileExistsError('Preserve complete native model replay attempts')
    target.mkdir(parents=True)
    records = []
    for registered in config['completed_jobs']:
        native_path = ROOT / registered['config']
        native = read(native_path)
        output = ROOT / native['output_root']
        result = read(output / 'result.json')
        if result['environment'] != dict(python=sys.version, torch=torch.__version__, numpy=np.__version__):
            raise ValueError('Original model numerical environment must be retained')
        completion = validate(native_path)
        source = ROOT / result['patch_manifest']['working_path']
        sys.path.insert(0, str(source))
        import run_models as author
        # Change only the inference device. All saved model and loader settings remain original.
        inference = dict(native, device='cpu')
        args, model_args = configure(author, inference, target)
        model = author.MODEL_DICT[args.model_type](**model_args).to('cpu')
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != result['parameters']:
            raise ValueError('Actual full original model architecture differs')
        best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(best['model_state_dict'], strict=True)
        model.eval()
        loader = author.UnifiedDataLoader(args.dataset_path, args.seq_len, args.feature_num,
                                         args.model_type, args.batch_size, args.num_workers, args.MIT)
        test = loader.get_test_dataloader()
        if len(test.dataset) != completion['full_test_units']:
            raise ValueError('Complete original test windows are required')
        predictions, targets, masks, indices = [], [], [], []
        batches = 0
        with torch.no_grad():
            for data in test:
                inputs, values = author.model_processing(data, model, 'test')
                predictions.append(values['imputed_data'].cpu())
                targets.append(inputs['X_holdout'].cpu())
                masks.append(inputs['indicating_mask'].cpu())
                indices.append(inputs['indices'].cpu())
                batches += 1
        prediction, truth, mask = map(torch.cat, (predictions, targets, masks))
        ids = torch.cat(indices).numpy()
        if (not torch.isfinite(prediction).all() or prediction.shape != truth.shape
                or len(prediction) != completion['full_test_units']
                or int(mask.sum()) != result['target_count']
                or sorted(ids.tolist()) != list(range(len(prediction)))):
            raise ValueError('Full finite native replay and every test coordinate required')
        actual = {name: float(function(prediction, truth, mask)) for name, function in
                  (('mae', author.masked_mae_cal), ('rmse', author.masked_rmse_cal), ('mre', author.masked_mre_cal))}
        compare_metrics(actual, result['metrics'], config['metric_tolerance'])
        difference = (prediction.double() - truth.double()) * mask
        stats = torch.stack((difference.abs().sum((1, 2)), difference.square().sum((1, 2)),
                             mask.sum((1, 2)).double(), (truth.double() * mask).abs().sum((1, 2))), dim=1).numpy()
        with np.load(ROOT / native['dataset_root'] / 'audit_arrays.npz', allow_pickle=False) as identities:
            unit_ids = identities['test_ids'][ids]
            groups = identities['test_groups'][ids] if native.get('bootstrap_group') == 'month' else unit_ids
        group_ids, grouped = aggregate_groups(stats, groups)
        filename = 'group_statistics.npz' if native.get('bootstrap_group') == 'month' else 'patient_statistics.npz'
        with np.load(output / filename, allow_pickle=False) as saved:
            if (not np.array_equal(group_ids, saved['group_ids'])
                    or sorted(unit_ids.tolist()) != sorted(saved['unit_ids'].tolist())
                    or not np.allclose(grouped, saved['statistics'], **config['metric_tolerance'])):
                raise ValueError('All original grouped error/target statistics must agree')
        total = grouped.sum(axis=0)
        independent = dict(mae=float(total[0]/total[2]), rmse=float(np.sqrt(total[1]/total[2])),
                           mre=float(total[0]/total[3]))
        compare_metrics(independent, result['metrics'], config['metric_tolerance'])
        artifact = target / (native['id'] + '_full_replay.npz')
        np.savez_compressed(artifact, predictions=prediction.numpy(), truth=truth.numpy(), mask=mask.numpy(),
                            test_indices=ids, unit_ids=unit_ids, group_ids=group_ids, group_statistics=grouped)
        row = dict(id=native['id'], seed=native['seed'], config=registered['config'],
                   config_sha256=sha(native_path), result_path=str(output / 'result.json'),
                   result_sha256=sha(output / 'result.json'), checkpoint_path=str(output / 'best.pt'),
                   checkpoint_sha256=sha(output / 'best.pt'), full_training_completion=completion,
                   model_parameters=result['parameters'], full_test_shape=list(prediction.shape),
                   full_native_batch_size=args.batch_size, full_test_batches=batches,
                   held_out_coordinates=int(mask.sum()), validation_best=result['validation_best'],
                   replayed_metrics=actual, independently_recomputed_metrics=independent,
                   original_metrics=result['metrics'], original_full_dataset_sha256=result['dataset_sha256'],
                   replay_arrays=str(artifact), replay_arrays_sha256=sha(artifact),
                   full_original_checkpoint_and_data_identity_verified=True,
                   full_test_grid_and_grouped_statistics_replayed=True)
        records.append(row)
        write(target / 'progress.json', dict(completed_full_models=len(records), required_full_models=config['full_model_count'], latest=row))
        print(json.dumps(dict(full_models=len(records), required=config['full_model_count'], id=native['id'],
                              full_test_coordinates=row['held_out_coordinates'], replayed_metrics=actual)), flush=True)
        del model, best, loader, test, predictions, targets, masks, indices, prediction, truth, mask, difference
        gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError('Readonly CPU inference must not initialize CUDA')
    report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), configuration=config_path,
                  configuration_sha256=sha(ROOT / config_path), full_models_replayed=len(records),
                  full_checkpoint_contents_and_re_inference_passed=True, metric_tolerance=config['metric_tolerance'],
                  actual_cpu_threads=torch.get_num_threads(), cuda_context_initialized=False,
                  additional_training_seed_slots=0, author_equivalence_certified=False,
                  all_original_experiments_complete=False, records=records,
                  boundary='Actual full original validation-selected checkpoints and every native test window/held-out coordinate. Native early stopping, batch size, model, data and grouped uncertainty identities preserved. CPU replay metrics and grouped errors agree under preregistered existing full-result tolerance. Original prediction arrays were not saved; no prediction bitwise-equivalence claim. Missing historical preprocessing/settings and unfinished tasks remain obligations.')
    write(target / 'checkpoint_replay_audit.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / args.config)
    verify_registration(config)
    if args.worker:
        return worker(config, args.config)
    api = resource_api()
    base = ROOT / config['state_root']
    base.mkdir(parents=True, exist_ok=True)
    owner = api['exclusive_lock'](base / 'controller.lock')
    if owner is None:
        raise RuntimeError('This full native replay already has a controller')
    state = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), state='starting', active_pid=None)
    def update(values):
        state.update(values)
        write(base / 'status.json', state)
    try:
        with api['resource_slot'](ROOT / config['resource_lock'], config['settings'], update):
            command = [config['python'], '-X', 'utf8', '-u', '-m',
                       'scripts.flow_matching.audit_saits_full_checkpoint_replay_v1', '--config', args.config, '--worker']
            with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                    env=dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),
                             PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
                             OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads'])),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                process = psutil.Process(child.pid)
                update(dict(active_pid=process.pid, active_create_time=process.create_time(), actual_command=process.cmdline()))
                code, usage = capped_wait(child, config, api, update)
        proof = ROOT / config['artifact_directory'] / 'checkpoint_replay_audit.json'
        passed = bool(code == 0 and proof.exists() and read(proof)['full_checkpoint_contents_and_re_inference_passed'])
        write(base / 'receipt.json', dict(exit_code=code, full_native_model_replay_passed=passed, **usage))
        update(dict(state='completed' if passed else 'failed_preserved', active_pid=None,
                    full_native_model_replay_passed=passed))
        if not passed:
            raise RuntimeError('Native model replay failed; preserve original full artifacts and evidence')
    finally:
        owner.close()


if __name__ == '__main__':
    main()
