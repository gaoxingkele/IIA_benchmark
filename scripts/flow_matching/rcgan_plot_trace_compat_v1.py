"""Restore legacy missing-column selection for the original trace plot only."""
import ast
from pathlib import Path


def plotting_ast(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'plot_trace']
    if len(functions) != 1:
        raise ValueError('Exactly one original trace plotting function required')
    function = functions[0]
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'da_ll']
    if len(assignments) != 1:
        raise ValueError('Exactly one original likelihood plotting selection required')
    expected = ast.parse("da_ll = da.loc[:, ['epoch', 'll', 'real_ll']].dropna()").body[0]
    if ast.dump(assignments[0]) != ast.dump(expected):
        raise ValueError('Do not alter an unknown original plotting expression')
    assignments[0].value = ast.parse("da.reindex(columns=['epoch', 'll', 'real_ll']).dropna()", mode='eval').body
    tree.body = [function]
    return ast.fix_missing_locations(tree)


def install(plotting):
    """Retain missing likelihoods as NaN; leave raw trace/source untouched."""
    path = Path(plotting.__file__)
    exec(compile(plotting_ast(path), str(path), 'exec'), plotting.__dict__)
    return dict(original_plotting_source=str(path), patched_expressions=1,
                original_function_body_preserved_except_missing_column_selection=True,
                missing_likelihood_columns_reindexed_as_nan=True,
                raw_trace_columns_and_values_unchanged=True,
                training_and_evaluator_functions_modified=False)
