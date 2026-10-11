"""Restore the author's floating precision without changing frozen old methods.

The pinned imperial-qore/TranAD main.py uses model_class(dims).double().
This remains a local window/preprocessing/validation adaptation, rather than
the author's complete padding, preprocessing, POT or historical environment.
Float32 and float64 are not generally ranking- or overflow-equivalent.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from . import mtsad_detectors as original
from .fm_legacy_window_adapter import LegacyWindowAdapter


def precision_method_ast(source, name):
    if name not in ('fit', 'score'):
        raise ValueError('Only the exact TranAD fit and score methods may be adapted')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'TranADDetector')
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    casts = [n for n in ast.walk(method) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'np' and n.attr == 'float32']
    if len(casts) != 1:
        raise ValueError('Exactly one original NumPy floating input cast required')
    casts[0].attr = 'float64'
    if name == 'fit':
        assignments = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
                       and ast.unparse(n) == 'model = self._build(n_features).to(self._device)']
        if len(assignments) != 1:
            raise ValueError('Exactly the original full model device assignment required')
        call = assignments[0].value
        call.func.value = ast.Call(func=ast.Attribute(value=call.func.value, attr='double', ctx=ast.Load()),
                                   args=[], keywords=[])
    return ast.fix_missing_locations(method)


def precision_method(name):
    source = Path(original.__file__).read_text(encoding='utf-8')
    namespace = dict(vars(original))
    node = precision_method_ast(source, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), original.__file__, 'exec'), namespace)
    return namespace[name]


class ReferenceFloat64TranADDetector(original.TranADDetector):
    """Identical local architecture and algorithm; full float64 fit and score."""
    fit = precision_method('fit')
    score = precision_method('score')


class ReferenceFloat64TranADAdapter(LegacyWindowAdapter):
    """Callable frozen runner adapter retaining the original complete-score checks."""
    def fit(self, windows):
        if self.method != 'tranad':
            raise ValueError('Only the explicitly named TranAD precision variant is accepted')
        parameters = {**self.parameters, 'epochs': self.epochs, 'seed': self.seed, 'device': self.device}
        self._delegate = ReferenceFloat64TranADDetector(**parameters)
        self._delegate.fit(windows)
        self._model = self._delegate._model
        return self
