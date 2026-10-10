"""Keep the full original sigma optimizer; certify reusable raw input graph.

The cached-Gram derivative failed full-data validation and is never used for
training. The full original fit function/gradient and2000-step stopping rule are
retained. Literal NumPy-input vs reusable native-placeholder MMD is validated.
"""
import ast
import inspect
from scripts.flow_matching import audit_rcgan_mmd_native_v1 as base
from scripts.flow_matching import audit_rcgan_mmd_native_v2 as diagnostic


def string_slice(node):
    node = node.value if isinstance(node, ast.Index) else node
    return node.s if isinstance(node, ast.Str) else getattr(node, 'value', None)


def retain_original_fit(function):
    changed_fit = 0
    changed_checks = 0
    changed_literal_check = 0
    changed_repeat = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.Subscript) and string_slice(node.slice) == 'cached'
                and isinstance(node.value, ast.Subscript) and string_slice(node.value.slice) == 'fit'
                and isinstance(node.value.value, ast.Name) and node.value.value.id == 'cases'):
            node.slice = ast.Index(value=ast.Str(s='original'))
            changed_fit += 1
        if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and [x.id for x in node.target.elts if isinstance(x, ast.Name)] == ['actual', 'expected']:
            # Cached gradients are recorded as rejected diagnostics, not certified.
            node.iter = ast.parse('[(cached_values, original_values)]', mode='eval').body
            changed_checks += 1
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'allclose'
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'literal_values'):
            node.args[1] = ast.Name(id='original_test', ctx=ast.Load())
            changed_literal_check += 1
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'unused'
                and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range'):
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Subscript) and string_slice(subnode.slice) == 'cached':
                    subnode.slice = ast.Index(value=ast.Str(s='original'))
                    changed_repeat += 1
    if changed_fit != 2 or changed_checks != 1 or changed_literal_check != 1 or changed_repeat != 1:
        raise ValueError('Exactly original sigma loss/fetch bindings and forward-only cache audit required')
    return function


def run(config):
    tree = ast.parse(inspect.getsource(diagnostic.run))
    function = tree.body[0]
    position = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scope')
    function.body[position:position] = ast.parse('function = _retain_original_fit(function)').body
    scope = dict(diagnostic.__dict__, _retain_original_fit=retain_original_fit)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function])), diagnostic.__file__, 'exec'), scope)
    base.ast = ast
    report = scope['run'](config)
    report.update(cached_gram_sigma_gradient_rejected=True, cached_gram_sigma_optimizer_used=False,
                  sigma_fit_uses_unchanged_original_function_and_gradient=True,
                  production_training_uses_no_gram_caching=True,
                  reusable_original_test_input_graph_compared_to_literal_author_numpy_call=True)
    return report


def main():
    base.audit = run
    base.main()


if __name__ == '__main__':
    main()
