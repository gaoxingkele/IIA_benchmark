import ast
import builtins
from pathlib import Path
from types import CodeType

from scripts.flow_matching.run_spectral_table2_sdformer_preflight_v1 import sdformer_function, original
from scripts.flow_matching import run_spectral_table2_sdformer_preflight_v1 as adapter
from scripts.flow_matching.run_maelnet_stage_resource_queue_v1 import original_namespace, FUNCTIONS


def executable_body(code):
    # Compiler inheritance can change future-annotation flags, not these bodies.
    constants = tuple(executable_body(x) if isinstance(x, CodeType) else x for x in code.co_consts)
    return (code.co_code, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, constants)


def test_sdformer_full_source_body_retained_except_explicit_coverage_assertion(monkeypatch):
    captured = []
    def capture(source, *a, **kw):
        captured.append(source)
        return builtins.compile(source, *a, **kw)
    monkeypatch.setattr(adapter, 'compile', capture, raising=False)
    function = sdformer_function()
    original_function = original.preflight_gpu
    source = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    expected = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight_gpu')
    comparison = next(n for n in ast.walk(expected) if isinstance(n, ast.Compare)
        and isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Name) and n.left.func.id == 'len'
        and isinstance(n.left.args[0], ast.Name) and n.left.args[0].id == 'proof')
    assert comparison.comparators[0].value == 9
    comparison.comparators[0].value = 6
    assert ast.dump(captured[0].body[0]) == ast.dump(expected)
    assert function.__globals__['workspace'] is not original_function.__globals__['workspace']


def test_maelnet_all_stage_functions_are_original_complete_definitions():
    namespace = original_namespace()
    source = Path(namespace['ROOT']) / 'scripts/flow_matching/run_maelnet_author_queue.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    for name in FUNCTIONS:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        unit = ast.Module(body=[node], type_ignores=[])
        expected = dict(namespace)
        exec(compile(unit, str(source), 'exec'), expected)
        assert executable_body(namespace[name].__code__) == executable_body(expected[name].__code__)
