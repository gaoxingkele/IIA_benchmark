"""Retain all five full native replays; route seed1104 to its completed366+134 run."""
import ast
from pathlib import Path

from scripts.flow_matching import run_rcgan_full500_completion_replay_queue_v1 as original

ROOT = original.ROOT


def verify(config):
    previous = original.read(ROOT / 'configs/runtime/fm_rcgan_full500_completion_replay_queue.v1.json')
    recovery_path = 'configs/reproducibility/fm_rcgan_seed1104_full134_state_recovery.2026-10-12_v3.json'
    recovery = original.read(ROOT / recovery_path)
    expected = dict(previous['routes'][1], execution_job=recovery['recovery_job'],
                    recovery_configuration=recovery_path,
                    native_receipt=recovery['state_root'] + '/receipt.json',
                    result_path=(Path(recovery['recovery_job']['output_directory']) / 'result.json').as_posix())
    if config['routes'] != [previous['routes'][0], expected] + previous['routes'][2:]:
        raise ValueError('Only the actual completed seed1104 mapping may change; all five seeds remain')
    for key in ('original_queue', 'required_epochs_per_job', 'required_generated_points_per_epoch',
                'required_generator_calls_per_epoch', 'full_fit_points', 'full_MMD_points',
                'equivalence_gate', 'cpu_threads', 'python', 'settings', 'resource_lock',
                'poll_seconds', 'original_scope', 'citation'):
        if config[key] != previous[key]:
            raise ValueError('Original full replay contract changed: ' + key)
    return original.verify(config)


def mapped_tree():
    source = ROOT / 'scripts/flow_matching/run_rcgan_full500_completion_replay_queue_v1.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    class Mapping(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'scripts.flow_matching.rcgan_native_training_resume_runtime_v2':
                node.module = 'scripts.flow_matching.rcgan_native_training_resume_runtime_v3'
            return node

        def visit_Constant(self, node):
            if node.value == 'scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v1':
                node.value = 'scripts.flow_matching.run_rcgan_full500_completion_replay_queue_v2'
            return node

    return ast.fix_missing_locations(Mapping().visit(tree))


def bind():
    source = ROOT / 'scripts/flow_matching/run_rcgan_full500_completion_replay_queue_v1.py'
    namespace = dict(original.__dict__)
    exec(compile(mapped_tree(), str(source), 'exec'), namespace)
    namespace['verify'] = verify
    return namespace


if __name__ == '__main__':
    bind()['main']()
