"""Replay all actual saved native generator/discriminator stages before recovery.

This is an additional read-only audit of the successful predecessor: three
generator checkpoints, full generation and ten complete trained discriminators.
It does not replace the mandatory full45/all20-evaluator final replay gate.
The complete source replay loops and bitwise variable/prediction/RNG checks
are inherited from the full45 worker without altering their numerical code.
"""
import ast
import copy
from datetime import datetime, timezone
import inspect
import json
import math
from types import FunctionType

from scripts.flow_matching import run_timegan_full45_completion_replay_v1 as original
from scripts.flow_matching.timegan_posthoc_recovery_v1 import binding


def verify(config):
    for pin in config['source_receipts']:
        if original.sha(original.ROOT / pin['path']) != pin['sha256']:
            raise ValueError('Frozen complete saved predecessor audit changed: ' + pin['path'])
    recovery = original.read(original.ROOT / config['recovery_configuration'])
    queue, model, job, checkpoints, evaluators = binding(recovery)
    reference = original.read(original.ROOT / config['reference_audit_runtime'])
    if (config['settings'] != reference['settings'] or config['resource_lock'] != reference['resource_lock']
            or config['python'] != queue['python'] or config['canonical_job'] != job
            or config['original_scope'] != dict(papers=45, method_baseline_ablation_records=681, original_data_task_records=273)):
        raise ValueError('Same actual predecessor, full research scope and existing read-only guards required')
    return queue, model, job, checkpoints, evaluators


def prepare_inputs(config):
    queue, model, job, checkpoints, evaluators = verify(config)
    recovery = original.read(original.ROOT / config['recovery_configuration'])
    route = dict(canonical_job=job, execution_job=job)
    ledger = dict(checkpoints=checkpoints, evaluators=evaluators,
        graph_seeds=[dict(index=i, kind='timegan' if i == 0 else 'discriminative',
            repeat=None if i == 0 else i - 1, seed=job['seed'] + 1009*i) for i in range(11)],
        scores=dict(discriminative=sum(r['score'] for r in evaluators)/10))
    # The actual author mean uses NumPy; evaluate it in the frozen native worker.
    import numpy as np
    ledger['scores']['discriminative'] = float(np.mean([r['score'] for r in evaluators]))
    identity = dict(recovery_configuration_sha256=original.sha(original.ROOT / config['recovery_configuration']),
        original_failed_terminal_receipt=recovery['terminal_receipt'])
    return queue, route, ledger, identity


def finish(config, ledger, identity, data_audit, stages, evaluators, generation_digest):
    queue, model, job, checkpoints, saved_evaluators = verify(config)
    if ledger['checkpoints'] != checkpoints or ledger['evaluators'] != saved_evaluators:
        raise ValueError('Actual saved predecessor stages changed during complete replay')
    if (len(stages) != 3 or len(evaluators) != 10
            or any(r['all_native_variables_restored_bitwise'] is not True for r in stages)
            or sum(r['source_protocol_iterations_reconstructed'] for r in evaluators) != 20000):
        raise ValueError('All three successful generator stages and all ten complete discriminators required')
    for index, row in enumerate(evaluators):
        if (row['kind'] != 'discriminative' or row['repeat'] != index
                or row['source_protocol_iterations_reconstructed'] != 2000
                or row['complete_test_windows'] != 1466 or row['numerical_optimizer_updates_executed'] != 0
                or any(row[k] is not True for k in ('all_native_variables_restored_bitwise',
                    'full_native_predictions_bitwise_verified', 'full_native_score_bitwise_verified',
                    'full_post_evaluator_NumPy_state_bitwise_verified', 'no_model_or_optimizer_variables_changed'))):
            raise ValueError('Every original full discriminator inference/state and bitwise gate required')
    if (identity['recovery_configuration_sha256'] != original.sha(original.ROOT / config['recovery_configuration'])
            or original.sha(original.ROOT / identity['original_failed_terminal_receipt']['path'])
                != identity['original_failed_terminal_receipt']['sha256']):
        raise ValueError('Preserved failure or recovery registration changed')
    report = dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(), canonical_job=job,
        full_original_data_shape=[3661,24,6], full_generated_windows_replayed=3661,
        generation_array_sha256=generation_digest, complete_normalized_and_renormalized_generation_bitwise_verified=True,
        actual_saved_checkpoint_restores=13, stage_checkpoint_restores=stages,
        actual_saved_trained_evaluator_replays=10, evaluators=evaluators,
        original_source_protocol_iterations_reconstructed=20000, numerical_optimizer_updates_executed=0,
        independent_full_loader_bitwise_verified=True, data_audit=data_audit,
        complete_saved_discriminative_mean_score_bitwise_verified=True,
        source_predecessor_and_all_completed_artifacts_unchanged=True,
        training_executed=False, additional_training_seed_slots=0, author_equivalence_certified=False,
        strict_TAB_result=False, complete_all20_native_evaluator_replay=False,
        missing_full_predictive_evaluator_trainings=10, additional_full_predictive_updates_required=50000,
        final_full45_checkpoint_restores_still_required=1035, final_full45_trained_evaluator_replays_still_required=900,
        new_benchmark_results=0, experiment_complete=False, all_original_experiments_complete=False)
    output = original.ROOT / config['output_root'] / job['id'] / 'saved_predecessor_native_replay.json'
    original.write(output, report)


def worker_ast():
    tree = ast.parse(inspect.getsource(original.worker))
    function = tree.body[0]
    environment = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'environment' for t in n.targets))
    # Successful complete experiment gates belong to the unchanged final replay.
    # Here inputs are explicitly the actual 13 saved successful stage receipts.
    function.body[:environment] = ast.parse('queue, route, result, binding = _prepare_inputs(config)').body
    changes = 0
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) and all(isinstance(v, ast.Name) for v in t.elts) and
            [v.id for v in t.elts] == ['evaluator_rows', 'scores'] for t in node.targets):
            node.value = ast.parse("([], dict(discriminative=[]))", mode='eval').body
            changes += 1
    restored = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Tuple) and all(isinstance(t, ast.Attribute) for t in n.targets[0].elts)
        and [t.attr for t in n.targets[0].elts] == ['reset_default_graph','Session'])
    function.body[restored+1:] = ast.parse(
        '_finish(config, result, binding, data_audit, stage_rows, evaluator_rows, generation_digest)').body
    if changes != 1:
        raise ValueError('Only the actually saved discriminative mean can be aggregated')
    return ast.fix_missing_locations(tree)


def compiled_worker():
    namespace = dict(original.__dict__, _prepare_inputs=prepare_inputs, _finish=finish)
    exec(compile(worker_ast(), __file__, 'exec'), namespace)
    return namespace['worker']


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    config = original.read(original.ROOT / args.config)
    queue, model, job, records, evaluators = verify(config)
    function = compiled_worker()
    if args.probe_only:
        import sys
        print(json.dumps(dict(passed=True, actual_saved_checkpoints=13, original_jobs=45,
            tensorflow_imported='tensorflow' in sys.modules, training_executed=False,
            final_full45_all20_gate_still_required=True)))
        return
    function(config, None, None)


if __name__ == '__main__':
    main()
