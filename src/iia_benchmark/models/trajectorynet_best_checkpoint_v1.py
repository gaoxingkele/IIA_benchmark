"""Isolated correction of the original caller's lost best-loss state.

The original train/validation bodies are retained except for returning the local
best loss and assigning that return in the caller. No author file is modified.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path


def corrected_functions(native):
    path = Path(inspect.getfile(native.train)).resolve()
    if path.name != 'main.py' or path.parent.name != 'TrajectoryNet':
        raise ValueError('Require the actual original TrajectoryNet/main.py module')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'd2a0ece7162369faa4a8ff4875376067d3cbea942191590085348029889827ab':
        raise ValueError('Pinned original training source changed')
    tree = ast.parse(raw.decode('utf-8'))
    original = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    train, validation = copy.deepcopy(original['train']), copy.deepcopy(original['train_eval'])
    if any(isinstance(n, ast.Return) for n in ast.walk(validation)):
        raise ValueError('Original validation contract changed')
    count = 0

    class ReturnToCaller(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal count
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'train_eval':
                count += 1
                return ast.copy_location(ast.Assign(targets=[ast.Name(id='best_loss', ctx=ast.Store())], value=node.value), node)
            return self.generic_visit(node)

    train = ReturnToCaller().visit(train)
    if count != 1:
        raise ValueError('Exactly one original validation call must be repaired')
    validation.body.append(ast.Return(value=ast.Name(id='best_loss', ctx=ast.Load())))
    fixed = ast.fix_missing_locations(ast.Module(body=[validation, train], type_ignores=[]))
    namespace = dict(native.__dict__)
    exec(compile(fixed, str(path) + '::best_checkpoint_state_correction_v1', 'exec'), namespace)
    return namespace['train'], namespace['train_eval']
