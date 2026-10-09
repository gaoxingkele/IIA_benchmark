"""Train and evaluate one complete frozen CFM-TS original-task experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.flow_matching.prepare_cfm_ts_original_data import ROOT, read, sha


def fingerprint(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def verify(queue):
    for receipt in queue['source_receipts']:
        if sha(ROOT / receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen source changed: '+receipt['path'])
    if len(queue['jobs']) != 285 or len({j['id'] for j in queue['jobs']}) != 285:
        raise ValueError('Full registered scope differs')


def complete(job):
    path = ROOT / job['output_directory'] / 'result.json'
    if not path.exists():
        return False
    result = read(path)
    if result['status'] != 'completed' or result['job_sha256'] != fingerprint(job):
        raise ValueError('Completed result does not match registration')
    if result['epochs_completed'] != job['epochs'] or result['optimizer_updates'] != job['expected_optimizer_updates']:
        raise ValueError('Partial training is not a complete result')
    if result['test_prediction_shape'] != [job['test_trajectories'], job['test_points_per_trajectory'], job['observed_dimensions']]:
        raise ValueError('Full test coverage differs')
    if result['diagnostic_smoke'] is not False or result['author_equivalence_certified'] is not False:
        raise ValueError('Unproved fidelity promoted')
    if any(sha(a['path']) != a['sha256'] for a in result['artifacts']):
        raise ValueError('Actual artifacts changed')
    import numpy as np
    losses = np.asarray(result['training_losses'],dtype=float)
    if losses.shape != (job['epochs'],) or not np.isfinite(losses).all():
        raise ValueError('Actual full epoch history missing/nonfinite')
    saved = next(a['path'] for a in result['artifacts'] if Path(a['path']).name == 'predictions.npz')
    with np.load(saved,allow_pickle=False) as arrays, np.load(ROOT / job['data_path'],allow_pickle=False) as data:
        ids = arrays['test_ids']
        if not np.array_equal(ids,data['test_ids']) or not np.array_equal(ids,result['test_trajectory_ids']):
            raise ValueError('Held-out trajectory identities differ')
        if not np.array_equal(arrays['truth'],data['test_states'][ids]) or not np.array_equal(arrays['times'],data['test_times'][ids]):
            raise ValueError('Saved full evaluation grid/ground truth differs')
        if arrays['predictions'].shape != tuple(result['test_prediction_shape']) or not np.isfinite(arrays['predictions']).all():
            raise ValueError('Actual prediction coverage/nonfinite values')
        actual = float(np.square(arrays['predictions'].astype(np.float64)-arrays['truth']).mean())
        if not np.isclose(actual,result['metrics']['MSE'],rtol=1e-12,atol=1e-12):
            raise ValueError('MSE does not match full predictions')
    return True


def write(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def run(queue, job):
    import numpy as np
    import torch
    from scripts.flow_matching.register_cfm_ts_original import environment
    from scripts.flow_matching.model_registry import build_configured_method
    verify(queue)
    if complete(job):
        return
    if sha(ROOT / job['data_path']) != job['data_sha256'] or sha(ROOT / job['model_config']) != job['model_config_sha256']:
        raise ValueError('Frozen data/model changed')
    if environment() != read(ROOT / queue['environment_lock']):
        raise ValueError('Frozen original-task environment changed')
    output = ROOT / job['output_directory']; artifacts = Path(job['artifact_directory'])
    if output.exists() or artifacts.exists():
        raise FileExistsError('Partial output preserved; register exact-budget retry')
    output.mkdir(parents=True); artifacts.mkdir(parents=True)
    torch.set_num_threads(queue['resource_policy']['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    with np.load(ROOT / job['data_path'], allow_pickle=False) as arrays:
        # The physical hidden state is deliberately not loaded into the model.
        times, values, train_ids = arrays['observed_times'], arrays['observed_states'], arrays['train_ids']
        test_ids, test_times, test_states, initial = (arrays[k] for k in ['test_ids','test_times','test_states','initial_states'])
    started = time.perf_counter()
    model = build_configured_method(ROOT / job['model_config'], job['model_overrides'])
    trajectories = [(times[index], values[index]) for index in train_ids]
    model.fit(trajectories, progress_callback=lambda p: write(output/'progress.json',dict(p,job_sha256=fingerprint(job))))
    training_seconds = time.perf_counter()-started
    if model.optimizer_updates_ != job['expected_optimizer_updates'] or len(model.loss_history_) != job['epochs']:
        raise ValueError('Full actual training budget differs')
    checkpoint = artifacts / 'model.pt'
    torch.save({'state_dict':model.model.state_dict(), 'job':job, 'job_sha256':fingerprint(job),
                'network_parameters':model.network_parameters, 'training_losses':model.loss_history_,
                'preprocessing':model.preprocessing_, 'optimizer_updates':model.optimizer_updates_}, checkpoint)
    started = time.perf_counter(); predictions = []
    for position, index in enumerate(test_ids):
        physical_times = np.concatenate([[0.],test_times[index]])
        prediction = model.simulate(initial[index], physical_times)[1:]
        predictions.append(prediction)
        write(output/'progress.json', {'phase':'scoring','test_trajectories_completed':position+1,
                                      'required_test_trajectories':len(test_ids),'job_sha256':fingerprint(job)})
    predictions = np.asarray(predictions); truth = test_states[test_ids]
    if not np.isfinite(predictions).all() or predictions.shape != truth.shape:
        raise ValueError('Incomplete/nonfinite prediction')
    errors = np.square(predictions.astype(np.float64)-truth)
    per_trajectory = errors.mean(axis=(1,2))
    metrics = {'MSE':float(errors.mean()), 'MAE':float(np.abs(predictions.astype(np.float64)-truth).mean()),
               'RMSE':float(np.sqrt(errors.mean())), 'trajectory_MSE_std':float(per_trajectory.std(ddof=1)) if len(per_trajectory)>1 else None,
               'trajectory_MSE_mean':float(per_trajectory.mean()), 'MSE_by_dimension':errors.mean(axis=(0,1)).tolist()}
    score_path = artifacts / 'predictions.npz'
    np.savez_compressed(score_path, predictions=predictions, truth=truth, times=test_times[test_ids], test_ids=test_ids,
                        initial_states=initial[test_ids], per_trajectory_MSE=per_trajectory)
    result = {'status':'completed', 'job':job, 'job_sha256':fingerprint(job), 'metrics':metrics,
              'epochs_completed':len(model.loss_history_), 'optimizer_updates':model.optimizer_updates_,
              'training_items':model.training_items_, 'training_losses':model.loss_history_,
              'training_seconds':training_seconds, 'score_seconds':time.perf_counter()-started,
              'parameter_count':sum(p.numel() for p in model.model.parameters()), 'test_prediction_shape':list(predictions.shape),
              'training_trajectory_ids':train_ids.tolist(), 'test_trajectory_ids':test_ids.tolist(),
              'data_sha256':job['data_sha256'], 'environment':environment(), 'diagnostic_smoke':False,
              'author_equivalence_certified':False, 'test_used_for_training_or_selection':False,
              'uncertainty_boundary':'Per-trajectory STD is descriptive, not five-seed uncertainty. Five paired data/model seeds are separately required; pendulum uses new times on the same trajectory.',
              'artifacts':[{'path':str(p),'sha256':sha(p)} for p in [checkpoint,score_path]], 'boundary':queue['boundary']}
    write(output/'result.json', result)
    if not complete(job):
        raise AssertionError('Result failed its full-budget verifier')
    print(json.dumps({'job':job['id'],'metrics':metrics,'full_epochs':job['epochs']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True); parser.add_argument('--job-id',required=True)
    args = parser.parse_args(); queue = read(ROOT / args.queue)
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    run(queue, job)
