"""Read-only GiFlow imputation capture; retries retain original seed slots."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

import psutil
from scipy.stats import t

from scripts.flow_matching.giflow_native_protocol import ROOT, complete, sha, verify
from scripts.flow_matching.giflow_update_budget_v2 import validate_observed_updates
from scripts.flow_matching.transition_giflow_at_boundary_v1 import compare_full_jobs

METRICS = ('native_mean_batch_mae', 'native_mean_batch_mse',
           'native_mean_batch_mape_percent', 'sqrt_native_mean_batch_mse')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observed_result(job, queue_path, root=ROOT):
    if not complete(job, root):
        return None
    output = root / job['output_directory']
    result = read(output / 'result.json')
    if result['job'] != job or result['queue_sha256'] != sha(root / queue_path):
        raise ValueError('GiFlow result belongs to a different queue or job')
    required_paths = [output/'native.stdout.log', output/'data_binding.json', output/'evaluated_model.pt']
    recorded_paths = {(root/a['path']).resolve() for a in result['artifacts']}
    if any(p.resolve() not in recorded_paths for p in required_paths):
        raise ValueError('GiFlow native training/data/evaluated weights receipts missing')
    if read(output/'data_binding.json') != result['data_binding']:
        raise ValueError('GiFlow persisted data binding differs')
    log = (output/'native.stdout.log').read_text(encoding='utf-8')
    epochs = [int(e) for e in re.findall(r'train_loss=.* after epoch (\d+)',log)]
    if epochs != list(range(result['epochs_completed'])):
        raise ValueError('GiFlow native epoch log differs')
    if len(epochs) < job['arguments']['training_epoch'] and 'Early stopping' not in log:
        raise ValueError('GiFlow short training lacks original early-stop evidence')
    loader = result['native_train_loader']
    if loader['batch_size'] != job['arguments']['batch_size'] or not loader['drop_last']:
        raise ValueError('Original full GiFlow batch/drop_last changed')
    if loader['samples'] != result['data_binding']['windows']['train']:
        raise ValueError('GiFlow observed training coverage changed')
    validate_observed_updates(result['epochs_completed'], job['arguments']['training_epoch'],
        result['optimizer_updates'], loader, result['epoch_optimizer_steps'])
    observation_path = output / 'training_observation.json'
    observation = read(observation_path)
    expected = dict(native_train_loader=loader, optimizer_steps=result['optimizer_updates'],
                    epoch_optimizer_steps=result['epoch_optimizer_steps'])
    if observation != expected or not any(
            (root / a['path']).resolve() == observation_path.resolve()
            and a['sha256'] == sha(observation_path) for a in result['artifacts']):
        raise ValueError('GiFlow persisted update evidence differs or lacks receipt')
    test_batches = math.ceil(result['data_binding']['windows']['test'] / loader['batch_size'])
    predictions = [a for a in result['artifacts'] if 'targets' in a]
    if result['full_test_batches'] != test_batches or len(predictions) != test_batches:
        raise ValueError('GiFlow full native test coverage incomplete')
    if sum(a['targets'] for a in predictions) != result['test_targets_with_window_repetitions']:
        raise ValueError('GiFlow prediction target receipts differ')
    values = result['metrics']
    if set(values) != set(METRICS) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('GiFlow native imputation metrics invalid')
    if not math.isclose(values['sqrt_native_mean_batch_mse']**2,
                        values['native_mean_batch_mse'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('GiFlow native square-root MSE differs')
    corrected = job['track'] == 'reviewed_corrected'
    if result['selection']['test_used_in_validation'] != (not corrected):
        raise ValueError('GiFlow selection boundary differs')
    if corrected and result['selection']['tested_weights_match_validation_checkpoint'] is not True:
        raise ValueError('GiFlow corrected branch lacks actual selected weight match')
    return result


def live_workers(root, queue_path, worker, processes=None):
    """State files alone never prove a live native model."""
    live = {}
    processes = psutil.process_iter() if processes is None else processes
    for process in processes:
        try:
            command = process.cmdline()
            if worker not in command or '--queue' not in command or '--job-id' not in command:
                continue
            value = command[command.index('--queue') + 1]
            if (root / value).resolve() != (root / queue_path).resolve():
                continue
            job_id = command[command.index('--job-id') + 1]
            live[job_id] = dict(pid=process.pid, create_time=process.create_time(), command=command)
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            continue
    return live


def aggregate(values):
    if not values:
        return dict(n=0, mean=None, std=None, se=None, ci95_low=None, ci95_high=None)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    se = std / len(values)**.5 if std is not None else None
    margin = float(t.ppf(.975, len(values)-1)) * se if se is not None else None
    return dict(n=len(values), mean=mean, std=std, se=se,
                ci95_low=mean-margin if margin is not None else None,
                ci95_high=mean+margin if margin is not None else None)


def summarize(settings, root=ROOT):
    old_path, new_path = settings['canonical_queue'], settings['observed_attempt_queue']
    old, new = read(root / old_path), read(root / new_path)
    verify(old, root); verify(new, root); compare_full_jobs(old, new)
    if new['original_canonical_queue'] != old_path or new['original_canonical_queue_sha256'] != sha(root / old_path):
        raise ValueError('GiFlow original canonical queue binding differs')
    if len(old['jobs']) != settings['required_canonical_slots']:
        raise ValueError('GiFlow canonical scope shortened')
    originals = live_workers(root, old_path, 'scripts.flow_matching.run_giflow_native_job')
    successors = live_workers(root, new_path, 'scripts.flow_matching.run_giflow_native_job_v2')
    stamp = datetime.now(timezone.utc).isoformat()
    jobs, groups = [], defaultdict(list)
    for original, attempt in zip(old['jobs'], new['jobs']):
        if attempt['original_canonical_id'] != original['id'] or attempt['original_output_directory'] != original['output_directory']:
            raise ValueError('GiFlow retry resolved a different original slot')
        record = observed_result(attempt, new_path, root)
        previous = root / original['output_directory']
        current = root / attempt['output_directory']
        running = successors.get(attempt['id']) or originals.get(original['id'])
        status = ('completed' if record else 'running' if running else
                  'partial_or_failed_preserved' if previous.exists() or current.exists() else 'pending')
        row = dict(algorithm='GiFlow/'+original['recipe'], task='imputation', id=original['id'],
            dataset=original['dataset'], track=original['track'], recipe=original['recipe'], seed=original['seed'],
            missing_ratio=original['arguments']['missing_rate'], pattern=original['arguments']['missing_type'],
            status=status, canonical_queue=old_path, observed_attempt_queue=new_path,
            original_output_directory=original['output_directory'], output_directory=attempt['output_directory'],
            original_attempt_preserved=previous.exists(), observed_attempt_exists=current.exists(),
            live_worker=running, metrics=record['metrics'] if record else None,
            result_sha256=sha(current/'result.json') if record else None,
            completed_epochs=record['epochs_completed'] if record else None,
            optimizer_updates=record['optimizer_updates'] if record else None,
            total_native_seconds=record['total_native_seconds'] if record else None,
            parameters=record['parameters'] if record else None,
            peak_cuda_allocated_bytes=record['peak_cuda_allocated_bytes'] if record else None,
            paper_equivalence=False, independently_replayed=False,
            legacy_attempt_boundary='Lost ephemeral update counters are not manufactured; old artifacts preserved.')
        jobs.append(row)
        groups[original['track'],original['dataset'],original['recipe']].append(row)
    metrics = []
    for (track,dataset,recipe), members in groups.items():
        if len(members) != settings['required_seeds_per_group'] or len({m['seed'] for m in members}) != len(members):
            raise ValueError('GiFlow seed scope differs or retry counted as an extra seed')
        ready = [m for m in members if m['status'] == 'completed']
        for metric in METRICS:
            metrics.append(dict(algorithm='GiFlow/'+recipe, dataset=dataset, task='imputation',
                protocol='giflow_native/'+track, metric=metric, pattern=members[0]['pattern'],
                missing_ratio=members[0]['missing_ratio'], required_repeats=len(members),
                **aggregate([m['metrics'][metric] for m in ready]),
                author_equivalence_certified=False, captured_utc=stamp, source=new_path,
                metric_unit='Masked overlapping test windows; unweighted mean of batch means; MAPE percent',
                comparison='source_fidelity_observed_updates_not_author_equivalence',
                selection_boundary='test_used_in_validation_final_reset_EMA' if track=='released_mirror'
                else 'validation_only_checkpoint_persistent_EMA'))
    return dict(captured_utc=stamp, canonical_slots=len(jobs), seed_groups=len(groups),
        job_counts=dict(Counter(j['status'] for j in jobs)), jobs=jobs, metric_rows=metrics,
        all_original_experiments_complete=False,
        source_receipts=[dict(path=p,sha256=sha(root/p)) for p in (old_path,new_path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args(); settings = read(ROOT/args.config)
    for item in settings['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Pinned GiFlow capture implementation changed')
    target = ROOT/settings['output_directory']
    if target.exists():
        raise FileExistsError('Keep prior result captures immutable')
    report = summarize(settings)
    target.mkdir(parents=True)
    report['source_receipts'] += settings['source_receipts'] + [dict(path=args.config,sha256=sha(ROOT/args.config))]
    (target/'execution_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,rows in [('canonical_jobs',report['jobs']),('algorithm_dataset_metrics',report['metric_rows'])]:
        with (target/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows([{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
    validation=dict(canonical_slots=report['canonical_slots'],seed_groups=report['seed_groups'],
        metric_rows=len(report['metric_rows']),measured_metric_rows=sum(r['n']>0 for r in report['metric_rows']),
        job_counts=report['job_counts'],same_canonical_seed_slots=True,task='imputation',
        independent_model_reinference_complete=False,all_original_experiments_complete=False,
        outputs={p.name:sha(p) for p in target.iterdir() if p.is_file()})
    (target/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
