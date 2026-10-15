"""Preserve the related MOMENT train body and isolate the paper epoch contract.

The upstream OR condition is a distinct source policy. Setting max_epoch=1
alone does not impose a one-epoch limit. This adapter replaces only that
condition in a private AST copy; no source file or running task is modified.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import MethodType


SOURCE_CONDITION = 'opt_steps < self.args.max_opt_steps or cur_epoch < self.args.max_epoch'
PAPER_CONDITION = 'cur_epoch < self.args.max_epoch'


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def train_node(source_path, expected_sha256):
    content = Path(source_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Original source SHA256 changed; do not silently patch another version')
    tree = ast.parse(content.decode('utf-8'))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pretraining']
    if len(classes) != 1:
        raise ValueError('Exactly one original Pretraining class required')
    functions = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'train']
    if len(functions) != 1:
        raise ValueError('Exactly one original train function required')
    original = functions[0]
    loops = [n for n in ast.walk(original) if isinstance(n, ast.While)]
    if len(loops) != 1 or ast.unparse(loops[0].test) != SOURCE_CONDITION:
        raise ValueError('Original two-minimum OR stopping policy not recognized')
    if any(isinstance(n, (ast.Break, ast.Continue, ast.Yield, ast.YieldFrom)) for n in ast.walk(original)):
        raise ValueError('Unexpected source loop exit or generator policy')
    return original


def paper_train_node(source_path, expected_sha256):
    original = train_node(source_path, expected_sha256)
    private = deepcopy(original)
    loop = next(n for n in ast.walk(private) if isinstance(n, ast.While))
    loop.test = ast.parse(PAPER_CONDITION, mode='eval').body
    ast.fix_missing_locations(private)
    # Replacing the condition back must reproduce the entire original AST.
    restored = deepcopy(private)
    next(n for n in ast.walk(restored) if isinstance(n, ast.While)).test = ast.parse(SOURCE_CONDITION, mode='eval').body
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise AssertionError('Only the top-level epoch stopping condition may change')
    return private


def build_paper_train(source_path, expected_sha256, namespace):
    private = paper_train_node(source_path, expected_sha256)
    isolated = dict(namespace)
    filename = '<isolated-paper-one-full-epoch:' + expected_sha256 + '>'
    exec(compile(ast.Module(body=[private], type_ignores=[]), filename, 'exec'), isolated)
    return isolated['train']


def bind_one_full_epoch(original_task, config, root):
    """Return an unexecuted bound train method, preserving the numerical body.

    This is not a full pretraining entry point. The task, complete data loader,
    model, environment, original resources and final checkpoint publication
    must be prepared and guarded by a separate scientific runner.
    """
    if config['paper_epochs'] != 1 or original_task.args.max_epoch != 1:
        raise ValueError('The scheduler argument and isolated loop must both use one full epoch')
    loader = original_task.train_dataloader
    positive_integer(len(loader), 'Full training loader batch count')
    if getattr(loader, 'drop_last', None) is not False:
        raise ValueError('Full epoch must retain the final partial batch')
    original = original_task.train.__func__
    source = Path(root) / config['source_path']
    frozen = train_node(source, config['source_sha256'])
    actual = ast.parse(textwrap.dedent(inspect.getsource(original))).body
    if len(actual) != 1 or ast.dump(actual[0], include_attributes=False) != ast.dump(frozen, include_attributes=False):
        raise ValueError('Supplied task does not expose the complete pinned original train method')
    adapted = build_paper_train(source, config['source_sha256'], original.__globals__)
    return MethodType(adapted, original_task)


def budget_outcome(batches_per_epoch, max_opt_steps, max_epoch, policy):
    """Exact counter-only outcome; examples are not measured dataset budgets."""
    batches_per_epoch = positive_integer(batches_per_epoch, 'Batches per epoch')
    max_opt_steps = positive_integer(max_opt_steps, 'Source step minimum')
    max_epoch = positive_integer(max_epoch, 'Epoch count')
    if policy == 'source_two_minima_OR':
        epochs = max(max_epoch, (max_opt_steps + batches_per_epoch - 1) // batches_per_epoch)
    elif policy == 'paper_complete_epochs':
        epochs = max_epoch
    else:
        raise ValueError('Keep explicit original-source and paper-epoch policies separate')
    return dict(epochs=epochs, optimizer_steps=epochs * batches_per_epoch,
                final_epoch_full=True, actual_neural_training=False)


def counter_projection(source_path, expected_sha256, policy):
    """Execute original loop/counters with a counter-only batch body.

    No torch import or model call. The selected stop condition and original
    counter initialization/increments are retained from the pinned source.
    """
    if policy == 'source_two_minima_OR':
        node = train_node(source_path, expected_sha256)
    elif policy == 'paper_complete_epochs':
        node = paper_train_node(source_path, expected_sha256)
    else:
        raise ValueError('Unknown loop projection policy')
    assignments = [n for n in node.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ['opt_steps', 'cur_epoch'] for t in n.targets)]
    loop = next(n for n in node.body if isinstance(n, ast.While))
    batches = [n for n in loop.body if isinstance(n, ast.For)]
    epoch_increments = [n for n in loop.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'cur_epoch' for t in n.targets)]
    if len(assignments) != 2 or len(batches) != 1 or len(epoch_increments) != 1:
        raise ValueError('Original epoch/optimizer counter structure changed')
    updates = [n for n in batches[0].body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'opt_steps' for t in n.targets)]
    if len(updates) != 1:
        raise ValueError('Exactly one original update counter per batch required')
    projected_loop = deepcopy(loop)
    projected_batch = deepcopy(batches[0])
    projected_batch.iter = ast.parse('range(batches_per_epoch)', mode='eval').body
    projected_batch.body = deepcopy(updates)
    projected_loop.body = [projected_batch, *deepcopy(epoch_increments)]
    projected = ast.FunctionDef(name='count', args=ast.parse('def count(self,batches_per_epoch): pass').body[0].args,
        body=[*deepcopy(assignments), projected_loop, ast.parse('return cur_epoch,opt_steps').body[0]], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[]))
    namespace = {}
    exec(compile(module, '<counter-only-original-loop-projection>', 'exec'), namespace)
    return namespace['count']
