"""Exact native RCGAN kernel cache and verified zero-distance roundoff bridge.

The author's MMD/variance formulas, including their diagonal convention, are
retained. Only XX/XY/YY contractions for fixed evaluator input arrays are cached.
Reference: Esteban, Hyland and Raetsch, arXiv:1706.02633v2.
"""
import ast
import copy
from pathlib import Path


def kernel_ast(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_mix_rbf_kernel'))
    changed = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('XX', 'XY', 'YY'):
                if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr in ('matmul', 'tensordot')):
                    raise ValueError('Original native Gram contraction changed')
                node.value = ast.parse("_cached_grams['" + name + "']", mode='eval').body
                changed.append(name)
    if sorted(changed) != ['XX', 'XX', 'XY', 'XY', 'YY', 'YY']:
        raise ValueError('Exactly six original matrix/tensor Gram bindings required')
    function.name = '_mix_rbf_kernel_cached'
    return function


def cached_kernel(module, X, Y, sigmas, grams):
    import sys
    function = kernel_ast(module)
    unit = ast.Module(body=[function]) if sys.version_info < (3, 8) else ast.Module(body=[function], type_ignores=[])
    scope = dict(module.__dict__, _cached_grams=grams)
    exec(compile(ast.fix_missing_locations(unit), module.__file__, 'exec'), scope)
    return scope[function.name](X, Y, sigmas)


def repair_proven_zero_distances(squared, X, Y):
    """Clamp only negative computed distances proven to be identical vectors.

    Nonzero true-distance negatives fail; this is not an arbitrary tolerance or
    approximation. Neither input data array is changed.
    """
    import numpy as np
    if not np.isfinite(squared).all():
        raise ValueError('Nonfinite squared distances cannot be repaired')
    rows, columns = np.where(squared < 0)
    minimum = float(np.min(squared))
    for start in range(0, len(rows), 2048):
        a = X[rows[start:start + 2048]]
        b = Y[columns[start:start + 2048]]
        if not np.array_equal(a, b):
            raise ValueError('Negative squared distances include nonidentical vectors')
    squared[rows, columns] = 0
    return dict(negative_pairs=len(rows), minimum_squared_distance=minimum,
                every_repaired_pair_bitwise_identical_input_vectors=True,
                positive_squared_distances_unchanged=True)
