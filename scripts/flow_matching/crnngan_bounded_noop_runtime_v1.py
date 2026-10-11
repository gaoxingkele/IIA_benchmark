"""Reuse graph-local inert fetches; retain all original train/data/model work."""
import ast

from scripts.flow_matching.crnngan_reviewed_source_v1 import instrumented_main as original_instrumented_main


def instrumented_main(path):
    tree = original_instrumented_main(path)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'tf' and node.func.attr == 'no_op']
    if len(calls) != 8 or any(node.args or node.keywords for node in calls):
        raise ValueError('Exactly eight inert original loop fetch constructors required')
    for node in calls:
        node.func = ast.Name(id='_native_cached_no_op', ctx=ast.Load())
    helper = ast.parse('''
def _native_cached_no_op():
    graph = tf.get_default_graph()
    operation = getattr(graph, '_iia_native_cached_noop', None)
    if operation is None:
        operation = tf.no_op(name='iia_native_cached_noop')
        graph._iia_native_cached_noop = operation
    return operation
''').body[0]
    tree.body.insert(0, helper)
    return ast.fix_missing_locations(tree)


def run(queue, job):
    from scripts.flow_matching import crnngan_native_training_runtime_v1 as original
    original.instrumented_main = instrumented_main
    return original.run(queue, job)
