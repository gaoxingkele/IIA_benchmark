"""Full23-checkpoint/all20-evaluator replay with the actual saved generator graph.

TimeGAN, Yoon et al., NeurIPS2019. The original source forward graph must equal
the saved graph before it is used. All original numeric checks, source sampler
loops, full data, seed accounting and resource guards remain unchanged.
"""
import ast
import copy
import inspect
import sys
from types import FunctionType

from scripts.flow_matching import timegan_saved_joint_full_replay_v1 as supplement
from scripts.flow_matching import run_timegan_full45_completion_replay_v5 as replay
from scripts.flow_matching import audit_timegan_stock1104_saved_graph_v1 as audit

BASE, ROOT = replay.BASE, replay.BASE.ROOT
PREVIOUS = 'configs/runtime/fm_timegan_full45_completion_replay.v6.json'
CANONICAL = 'timegan_native__gru__stock__seed1104'


def verify(config):
    previous = BASE.read(ROOT / PREVIOUS)
    if (config['previous_full45_configuration'] != PREVIOUS or config['selected_canonical_ids'] != [CANONICAL]
            or config['routes'] != previous['routes']):
        raise ValueError('All45 exact native routes must remain unchanged')
    for key in ('original_queue', 'settings', 'resource_lock', 'python', 'cpu_threads',
            'required_checkpoints_each', 'required_evaluators_each', 'equality_gate',
            'protocol_iterations_each', 'required_full_generated_windows',
            'required_full_predictive_test_windows', 'required_full_discriminative_test_windows', 'original_scope'):
        if config[key] != previous[key]:
            raise ValueError('Original full native replay budget/guard/scope changed: ' + key)
    proof = BASE.read(ROOT / config['saved_graph_proof'])
    receipt = BASE.read(ROOT / config['saved_graph_proof_receipt'])
    graph = proof['forward_graph_comparison']
    if (receipt['exit_code'] != 0 or receipt['resource_guard_aborted'] or not receipt['saved_graph_observation_captured']
            or proof['canonical_job']['id'] != CANONICAL
            or proof['generation_comparison']['original_bitwise_gate_passed'] is not True
            or proof['generation_comparison']['unequal_coordinates'] != 0
            or proof['renormalized_comparison']['array_equal'] is not True
            or proof['full_saved_variables_bitwise_verified'] != 284
            or proof['saved_graph_inference_calls'] != 1
            or not graph['forward_definitions_equal_excluding_output_shape_metadata']
            or graph['shape_metadata_changes'] or not graph['function_library_bitwise_equal']):
        raise ValueError('Actual complete saved graph proof and unchanged full forward definitions required')
    return replay.verify(config)


def worker_ast():
    tree = ast.parse(inspect.getsource(BASE.worker)); changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'constructor_prefix':
            if ast.dump(node) != ast.dump(ast.parse("constructor_prefix(author_model, 'timegan')", mode='eval').body):
                raise ValueError('Only original generator graph construction may change')
            node.func.id = '_saved_generator_constructor'; changed += 1
    if changed != 1:
        raise ValueError('Exactly one full generator construction binding required')
    return ast.fix_missing_locations(tree)


def saved_generator_constructor(module, name):
    from scripts.flow_matching.timegan_native_common_v1 import constructor_prefix
    if name != 'timegan':
        raise ValueError('Every trained evaluator graph remains reconstructed by unchanged source')
    frame = inspect.currentframe().f_back
    original_reset = frame.f_locals['original_reset']
    graph_seeds = frame.f_locals['graph_seeds']
    result, route, config = (frame.f_locals[k] for k in ('result', 'route', 'config'))
    directory = BASE.checkpoint_directory(result['checkpoints'][2])
    source_constructor = constructor_prefix(module, name)

    def construct(real, parameters):
        import tensorflow as tf
        nodes = source_constructor(real, parameters)
        names = {k: nodes[k].name for k in ('X_hat', 'Z', 'X', 'T')}
        source_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
        meta = tf.MetaGraphDef(); meta.ParseFromString((directory / 'model.meta').read_bytes())
        graph = audit.graph_difference(source_graph, meta.graph_def, names['X_hat'])
        if (not graph['forward_definitions_equal_excluding_output_shape_metadata'] or graph['shape_metadata_changes']
                or not graph['function_library_bitwise_equal'] or len(graph_seeds) != 1
                or graph_seeds[0]['seed'] != route['canonical_job']['seed']):
            raise ValueError('Exact same source math/metadata and one original generator seed required')
        # Keep the original logical graph seed ledger at one generator. No
        # numeric call/initializer/optimizer is executed while importing.
        original_reset(); tf.set_random_seed(route['canonical_job']['seed'])
        tf.train.import_meta_graph(str(directory / 'model.meta'), clear_devices=False)
        for key, tensor_name in names.items():
            nodes[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if len(tf.global_variables()) != 284:
            raise ValueError('Every original model and Adam variable must remain')
        BASE.write(ROOT / config['output_root'] / CANONICAL / 'saved_generator_graph_binding.json', dict(
            saved_meta_graph_path=str(directory / 'model.meta'), saved_meta_graph_sha256=BASE.sha(directory / 'model.meta'),
            full_forward_graph_comparison=graph, original_logical_generator_graph_seeds=copy.deepcopy(graph_seeds),
            numeric_initializers_or_optimizer_updates_executed=0, source_generation_and_all20_evaluator_code_unchanged=True,
            all_original_full_data_and_bitwise_gates_required=True, additional_training_seed_slots=0))
        return nodes
    return construct


def native_namespace():
    namespace = replay.runtime_namespace()
    namespace.update(verify=verify, _saved_generator_constructor=saved_generator_constructor)
    exec(compile(worker_ast(), __file__, 'exec'), namespace)
    return namespace


def main_ast():
    tree = ast.parse(inspect.getsource(supplement.main)); changed = 0
    legacy = sys.version_info < (3, 8)
    literal_kind = ast.Str if legacy else ast.Constant
    for node in ast.walk(tree):
        if not isinstance(node, literal_kind):
            continue
        text = node.s if legacy else node.value
        if not isinstance(text, str):
            continue
        old = "runpy.run_module('scripts.flow_matching.timegan_saved_joint_full_replay_v1'"
        if old in text:
            updated = text.replace(old, "runpy.run_module('scripts.flow_matching.timegan_stock1104_saved_graph_full_replay_v1'")
            if legacy: node.s = updated
            else: node.value = updated
            changed += 1
    if changed != 1:
        raise ValueError('Exactly one full native child route must select the saved graph worker')
    return ast.fix_missing_locations(tree)


def main():
    namespace = dict(supplement.main.__globals__, CANONICAL=CANONICAL, verify=verify, native_namespace=native_namespace)
    # Reuse the entire guarded single-case dispatcher and all45 evidence union,
    # without touching or re-running the existing failed full45 controller.
    collect = FunctionType(supplement.collect_full45_evidence.__code__,
        dict(supplement.collect_full45_evidence.__globals__, CANONICAL=CANONICAL))
    namespace['collect_full45_evidence'] = collect
    exec(compile(main_ast(), __file__, 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
