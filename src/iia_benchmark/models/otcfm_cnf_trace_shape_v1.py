"""Restore a scalar log-density derivative per observation on private instances."""
import ast
import inspect
import textwrap
from types import MethodType


def corrected_forward(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    expected = ast.parse('-self.trace_estimator(dx, x) + 0 * x', mode='eval').body
    if len(returns) != 1 or ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError('Require the exact frozen original CNFReg scalar-trace/broadcast expression')
    returns[0].value = ast.parse('-self.trace_estimator(dx, x) + 0 * x.sum(dim=1)', mode='eval').body
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<isolated_original_CNFReg_scalar_shape>', 'exec'), namespace)
    return namespace[function.name]


def install_model(model):
    regs = model.augmentations.regs
    if len(regs) != 1 or type(regs[0]).__name__ != 'CNFReg':
        raise ValueError('This plain-CNF profile requires one original exact-trace regularizer')
    reg = regs[0]
    if model.aug_net.augmentation_list[0] is not reg:
        raise ValueError('Patch the actual shared original augmented-ODE regularizer instance')
    if getattr(reg, '_iia_scalar_trace_shape_corrected', False):
        raise ValueError('Do not install a numerical adapter twice')
    original = type(reg).forward
    reg.forward = MethodType(corrected_forward(original), reg)
    reg._iia_scalar_trace_shape_corrected = True
    return dict(original_class='CNFReg', corrected_instances=1, exact_trace_estimator_unchanged=True,
                negative_trace_per_observation=True, zero_state_graph_connection_retained=True,
                source_class_or_files_modified=False)
