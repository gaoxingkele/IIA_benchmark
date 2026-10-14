"""Complete point/native/independent replay of the original causal PSM1106.

Reuse the frozen full replay stages, guards and score tolerances. Preserve all
1140 original jobs and six seeds. This audit adds no training and does not
certify journal equivalence or strict TAB performance.
"""
import argparse
import json
import sys
from types import FunctionType

from scripts.flow_matching import run_pi_causal_completed_pair_replay_v1 as original

ROOT, read, digest = original.ROOT, original.read, original.digest
CANONICAL = 'pi_author__causal_axis_repaired__psm__full__seed1106'
IDS = [CANONICAL]
SCOPE = dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)
BUDGET = dict(additional_training_seed_slots=0, required_original_training_updates=1554,
              required_full_train_score_points=13238200,
              required_full_test_score_points=87800, required_full_saved_score_streams=15)


def namespace():
    return dict(vars(original), IDS=IDS, read=read, digest=digest)


def verify(config):
    if any(config.get(k) != v for k, v in BUDGET.items()):
        raise ValueError('All original training updates and full score streams required')
    queue, native, review = FunctionType(original.verify.__code__, namespace())(config)
    point = read(config['cases'][0]['point_configuration'])
    _, job, result = original.completion.binding(point)
    if (result.get('status') != 'completed' or job['seed'] != 1106
            or result.get('active_attention_layers') != 1
            or [e.get('epoch') for e in result['epochs']] != [1, 2, 3]
            or any(e.get('early_stop') is not False for e in result['epochs'])
            or result.get('native_train_points') != 132481
            or result.get('training_windows') != 132382
            or result.get('native_test_points') != 87841
            or result.get('scored_test_points') != 87800
            or result.get('dropped_test_tail_points') != 41
            or job['author_config']['data']['win_size'] != 100):
        raise ValueError('Exact original complete PSM1106 budget and covered points required')
    return queue, native, review


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    queue, _, _ = verify(config)
    if args.probe_only:
        print(json.dumps(dict(passed=True, required_original_jobs=len(queue['jobs']),
            selected_ready_original_ids=IDS, **BUDGET,
            torch_imported='torch' in sys.modules, numpy_imported='numpy' in sys.modules,
            full_native_checkpoint_reinference_passed=False,
            all_original_experiments_complete=False)))
        return
    values = dict(namespace(), verify=verify)
    FunctionType(original.controller.__code__, values)(config, args.config)


if __name__ == '__main__':
    main()
