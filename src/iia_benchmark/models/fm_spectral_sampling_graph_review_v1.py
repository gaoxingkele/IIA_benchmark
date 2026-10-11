"""Review only higher-order graph retention in full native Spectral sampling.

Keep the native first derivative, every tensor operation and entire batch. This
is an inference-only repair candidate until full native operator and sampling
audits pass. Training's higher-order graph is never replaced.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import inspect
import textwrap
import types


def reviewed_forward(function):
    source=textwrap.dedent(inspect.getsource(function))
    original=ast.parse(source)
    tree=copy.deepcopy(original)
    replacements=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.autograd.grad':
            keywords=[kw for kw in node.keywords if kw.arg=='create_graph']
            if len(keywords)!=1 or not isinstance(keywords[0].value,ast.Constant) or keywords[0].value.value is not True:
                raise ValueError('Exactly the original explicit higher-order gradient graph required')
            replacements.append(keywords[0])
    if len(replacements)!=1 or len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Exactly one complete native derivative implementation required')
    replacements[0].value=ast.Constant(value=False)
    ast.fix_missing_locations(tree)
    namespace=dict(function.__globals__)
    exec(compile(tree,inspect.getsourcefile(function),'exec'),namespace)
    reviewed=namespace[function.__name__]
    receipt=dict(original_forward_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_AST_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        reviewed_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        only_change='native torch.autograd.grad create_graph=True -> False inside inference-only scope',
        native_first_derivative_operations_unchanged=True,full_batch_unchanged=True,
        training_forward_unchanged=True,full_native_numerical_equivalence_certified=False)
    return reviewed,receipt


@contextmanager
def sampling_first_derivative_graph(model):
    import torch
    if torch.is_grad_enabled():
        raise ValueError('Review candidate may only be installed inside explicit no-grad sampling')
    network=model.net
    if 'forward' in network.__dict__:
        raise ValueError('Preserve existing native forward override')
    original=network.forward.__func__
    reviewed,receipt=reviewed_forward(original)
    network.forward=types.MethodType(reviewed,network)
    try:
        yield receipt
    finally:
        del network.forward
        if network.forward.__func__ is not original:
            raise RuntimeError('Original higher-order training derivative was not restored')
