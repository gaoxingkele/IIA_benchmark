"""Run complete original GiFlow replay with coverage metadata bound to each seed."""
import ast
import copy
from pathlib import Path

from scripts.flow_matching import audit_giflow_checkpoint_replay_v1 as base


def released_boundary(proof, result):
    windows = result['data_binding']['windows']['test']
    batches = result['full_test_batches']
    targets = result['test_targets_with_window_repetitions']
    if (result['job']['track'] != 'released_mirror' or result['selection']['test_used_in_validation'] is not True
            or proof['job_id'] != result['job']['id'] or proof['seed'] != result['job']['seed']
            or proof['full_test_windows'] != windows or proof['full_test_batches'] != batches
            or proof['full_masked_test_targets'] != targets or proof['observed_forward_calls'] != 20 * batches
            or proof['observed_window_evaluations'] != 20 * windows
            or proof['full_checkpoint_contents_and_reinference_passed'] is not True):
        raise ValueError('Complete released native model, original seed and all native targets required')
    reviewed = copy.deepcopy(proof)
    reviewed.update(track='released_mirror', selection=result['selection'],
        actual_full_training_epochs=result['epochs_completed'], actual_optimizer_updates=result['optimizer_updates'],
        source_test_block_unchanged=True, additional_training_seed_slots=0,
        boundary=f'Actual released-source evaluated EMA checkpoint, original whole twenty-step Euler test. '
                 f'All {windows} windows/{targets} masked targets/{batches} batches/{20*batches} forwards. '
                 'Frozen original CPU/GPU tolerances; not a bitwise prediction claim. '
                 'Native overlapping-window unweighted batch means, test-in-validation and per-batch EMA reset '
                 'limitations retained. Existing training seed verified, no added seed, no author/TAB equivalence.')
    return reviewed


def worker(config, config_path):
    base.worker(config, config_path)
    queue = base.read(base.ROOT / config['queue'])
    job = next(j for j in queue['jobs'] if j['id'] == config['job_id'])
    result = base.read(base.ROOT / job['output_directory'] / 'result.json')
    path = base.ROOT / config['output_directory'] / 'checkpoint_replay_audit.json'
    proof = released_boundary(base.read(path), result)
    proof.update(original_raw_replay_proof=str(path), original_raw_replay_proof_sha256=base.sha(path),
                 metadata_boundary_clarification_only=True)
    base.write(path.parent / 'released_seed_checkpoint_replay_audit.json', proof)


def main():
    tree = ast.parse(Path(base.__file__).read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    count = 0
    class ChildEntry(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal count
            if node.value == 'scripts.flow_matching.audit_giflow_checkpoint_replay_v1':
                count += 1
                return ast.copy_location(ast.Constant('scripts.flow_matching.audit_giflow_released_seed_checkpoint_v1'), node)
            return node
    definition = ChildEntry().visit(definition)
    if count != 1:
        raise ValueError('Frozen original full replay child-entry boundary changed')
    namespace = dict(vars(base), worker=worker)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), base.__file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
