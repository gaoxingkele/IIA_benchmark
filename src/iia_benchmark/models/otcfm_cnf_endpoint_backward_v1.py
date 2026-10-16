"""Evaluate the unchanged additive trajectory CNF objective one graph at a time."""
import ast
import inspect
from types import MethodType


def endpoint_step_tree(source_text, original_tree_builder):
    tree = original_tree_builder(source_text, True)
    fn = tree.body[0]
    fn.args.kwonlyargs.append(ast.arg(arg='endpoint_index'))
    fn.args.kw_defaults.append(ast.Constant(value=None))
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(len(reversed_ts) - 1)':
        raise ValueError('Require the frozen original additive endpoint loop')
    condition = ast.parse('if endpoint_index is not None and len(even_ts)-1-t != endpoint_index:\n    continue').body[0]
    loops[0].body.insert(1, condition)
    return ast.fix_missing_locations(tree)


def install_model(model, source_class, original_tree_builder):
    if getattr(model, '_iia_endpoint_backward_installed', False):
        raise ValueError('Do not install a source numerical adapter twice')
    tree = endpoint_step_tree(inspect.getsource(source_class), original_tree_builder)
    namespace = dict(source_class.step.__globals__)
    exec(compile(tree, '<isolated_original_CNF_endpoint_selection>', 'exec'), namespace)
    model.step = MethodType(namespace['step'], model)
    model._iia_endpoint_backward_installed = True


def backward_all_endpoints(model, batch, leftout):
    import torch
    if not model.is_trajectory or len(batch) not in (4, 5) or not 0 < leftout < len(batch)-1:
        raise ValueError('Complete four/five-timepoint trajectory and interior leaveout required')
    indices = [i for i in reversed(range(len(batch))) if i != leftout]
    values = []
    for index in indices:
        reg, loss = model.step(batch, training=True, endpoint_index=index)
        contribution = (reg + loss) / len(indices)
        contribution.backward()
        values.append(contribution.detach())
        del reg, loss, contribution
    return torch.stack(values).sum(), dict(
        all_retained_endpoint_indices=indices, equal_endpoint_weight=1/len(indices),
        complete_objective_terms=len(indices), graphs_backward_one_at_a_time=True,
        optimizer_updates=0, observations_subsampled=False, original_exact_trace_unchanged=True)
