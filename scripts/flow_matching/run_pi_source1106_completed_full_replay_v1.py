"""Replay the complete canonical source-runtime PSM1106 without new training.

Reuse the frozen point/native/independent stages and exact numerical guards.
The binding-only adapter selects the source-runtime branch rather than the
causal-axis control; all 1140 original jobs and six seeds remain required.
"""
import argparse
import ast
import inspect
import json
import sys
from types import FunctionType

from scripts.flow_matching import run_pi_causal_completed_pair_replay_v1 as original

ROOT, read, digest = original.ROOT, original.read, original.digest
CANONICAL = 'pi_author__source_runtime__psm__full__seed1106'
IDS = [CANONICAL]
BUDGET = dict(additional_training_seed_slots=0, required_original_training_updates=1554,
              required_full_train_score_points=13238200,
              required_full_test_score_points=87800, required_full_saved_score_streams=15)


def source_binding_ast(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    expected = ast.parse("job['branch'] != 'causal_axis_repaired'", mode='eval').body
    matches = [n for n in ast.walk(function) if ast.dump(n) == ast.dump(expected)]
    if len(matches) != 1:
        raise ValueError('Exactly one original branch binding boundary required')
    matches[0].comparators[0] = ast.Constant(value='source_runtime')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def namespace():
    return dict(vars(original), IDS=IDS, read=read, digest=digest)


def verify(config):
    if any(config.get(k) != v for k, v in BUDGET.items()):
        raise ValueError('All original training updates and full score streams required')
    values = namespace()
    exec(compile(source_binding_ast(inspect.getsource(original.verify)),
                 '<unchanged full Pi verifier with source-runtime binding>', 'exec'), values)
    queue, native, review = values['verify'](config)
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
        raise ValueError('Exact original complete source-runtime PSM1106 budget required')
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
    FunctionType(original.controller.__code__, dict(namespace(), verify=verify))(config, args.config)


if __name__ == '__main__':
    main()
