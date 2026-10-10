"""Independently recompute every full native GiFlow test batch from saved arrays."""
import argparse
import math
from pathlib import Path

import numpy as np

from scripts.flow_matching.audit_cfm_checkpoint_replay_v1 import ROOT, read, sha, write
from scripts.flow_matching.summarize_giflow_canonical_v1 import observed_result


def batch_errors(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(target):
        raise ValueError('Complete nonempty masked target/prediction vectors required')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Full finite native test arrays required')
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    eligible = np.abs(target) > 1e-6
    if not eligible.any():
        raise ValueError('Native MAPE undefined: no nonzero targets')
    return dict(native_mean_batch_mae=float(np.abs(delta).mean()),
        native_mean_batch_mse=float((delta ** 2).mean()),
        native_mean_batch_mape_percent=float(np.abs(delta[eligible] / target[eligible]).mean() * 100))


def audit(config):
    for item in config['source_receipts']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen independent GiFlow array audit input changed')
    queue = read(ROOT/config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    result = observed_result(job, config['queue'])
    if result is None:
        raise ValueError('Actual full completed native training result required')
    rows = []
    for item in result['artifacts']:
        if 'targets' not in item:
            continue
        path = Path(item['path'])
        with np.load(path, allow_pickle=False) as arrays:
            positions, shape = arrays['masked_flat_positions'], arrays['batch_shape']
            if (len(positions) != item['targets'] or len(np.unique(positions)) != len(positions)
                    or np.any(positions < 0) or np.any(positions >= np.prod(shape))
                    or shape[1] != job['arguments']['window'] or shape[2] != result['data_binding']['nodes']):
                raise ValueError('Original masked positions/window/node geometry differs')
            errors = batch_errors(arrays['prediction'], arrays['target'])
            rows.append(dict(path=str(path), sha256=sha(path), targets=len(positions), batch_shape=shape.tolist(), **errors))
    if len(rows) != result['full_test_batches'] or sum(r['targets'] for r in rows) != result['test_targets_with_window_repetitions']:
        raise ValueError('All full native test batches/targets required')
    # Native aggregation is the unweighted mean of batch errors, not target-weighted.
    values = {name: float(np.mean([r[name] for r in rows])) for name in
        ('native_mean_batch_mae','native_mean_batch_mse','native_mean_batch_mape_percent')}
    values['sqrt_native_mean_batch_mse'] = math.sqrt(values['native_mean_batch_mse'])
    for name, value in values.items():
        if not math.isclose(value, result['metrics'][name], **config['tolerance']):
            raise ValueError('Independent whole-test native metric differs: '+name)
    return dict(job_id=job['id'], seed=job['seed'], task='imputation', dataset=job['dataset'], track=job['track'],
        actual_full_training_epochs=result['epochs_completed'], configured_maximum_epochs=job['arguments']['training_epoch'],
        native_early_stopping_retained=True, actual_optimizer_updates=result['optimizer_updates'],
        full_test_batches=len(rows), masked_targets_with_window_repetitions=sum(r['targets'] for r in rows),
        independently_recomputed_metrics=values, native_metrics=result['metrics'], batch_receipts=rows,
        result_sha256=sha(ROOT/job['output_directory']/'result.json'),
        full_test_arrays_and_metrics_passed=True, independent_model_reinference_complete=False,
        no_extra_seed_or_retraining=True, author_equivalence_certified=False, all_original_experiments_complete=False,
        boundary='Every saved masked test prediction and native batch-mean metric independently checked in float64 under frozen float32 reduction tolerance. This is full-array/metric verification, not model re-inference. Native overlapping windows, MAPE target cutoff1e-6, test-in-validation and final EMA reset behavior retained; one of five required training seeds.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read(ROOT/args.config);target=ROOT/config['output_directory']
    if target.exists():
        raise FileExistsError('Preserve immutable full-array audit')
    proof=audit(config);target.mkdir(parents=True)
    write(target/'full_array_audit.json',proof)
    write(target/'validation.json',dict(passed=True, full_test_arrays_and_metrics_passed=True,
        independent_model_reinference_complete=False, source_receipts=config['source_receipts']+
        [dict(path=args.config,sha256=sha(ROOT/args.config))], outputs={'full_array_audit.json':sha(target/'full_array_audit.json')}))
    print(dict(full_test_batches=proof['full_test_batches'], optimizer_updates=proof['actual_optimizer_updates'],
               epochs=proof['actual_full_training_epochs'],metrics=proof['independently_recomputed_metrics']))


if __name__=='__main__':
    main()
