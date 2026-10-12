"""Same telemetry wire layout/formula, bounded ctypes cache and client binding."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.flow_matching import cached_resource_snapshot_v1 as cached


def test_wire_layout_is_identical_to_original_Windows_structure():
    tree = ast.parse(Path('scripts/flow_matching/heavy_resources.py').read_text(encoding='utf-8'))
    structure = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'PerformanceInfo')
    namespace = dict(ctypes=ctypes)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[structure],type_ignores=[])),'<original-wire-layout>','exec'),namespace)
    expected = namespace['PerformanceInfo']
    assert expected._fields_ == cached.PerformanceInfo._fields_
    assert ctypes.sizeof(expected) == ctypes.sizeof(cached.PerformanceInfo)
    assert all(getattr(expected,name).offset == getattr(cached.PerformanceInfo,name).offset for name,_ in expected._fields_)


def test_thousand_actual_Windows_polls_do_not_add_pointer_types():
    if cached.sys.platform != 'win32': pytest.skip('Windows native telemetry')
    cached.resource_snapshot()
    before = len(ctypes._pointer_type_cache)
    for _ in range(1000):
        result = cached.resource_snapshot()
        assert set(result) == {'physical_available_bytes','commit_available_bytes'}
    assert len(ctypes._pointer_type_cache) == before


def test_exact_original_commit_formula_and_physical_available_value(monkeypatch):
    monkeypatch.setattr(cached.sys,'platform','win32')
    monkeypatch.setattr(cached.psutil,'virtual_memory',lambda:SimpleNamespace(available=987654321))
    def api(pointer, size):
        assert size == ctypes.sizeof(cached.PerformanceInfo)
        info=ctypes.cast(pointer,ctypes.POINTER(cached.PerformanceInfo)).contents
        assert info.cb == size
        info.CommitLimit=200; info.CommitTotal=71; info.PageSize=4096
        return 1
    monkeypatch.setattr(cached,'windows_api',lambda:api)
    assert cached.resource_snapshot() == dict(physical_available_bytes=987654321,commit_available_bytes=129*4096)


def test_API_failure_is_not_converted_to_success(monkeypatch):
    if cached.sys.platform != 'win32': pytest.skip('Windows error contract')
    monkeypatch.setattr(cached,'windows_api',lambda:lambda *args:0)
    with pytest.raises(OSError): cached.resource_snapshot()


def test_existing_headroom_check_keeps_both_original_limits():
    from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api
    api=resource_api()
    settings=dict(minimum_available_memory_bytes=24,minimum_commit_headroom_bytes=28)
    assert api['has_headroom'](dict(physical_available_bytes=24,commit_available_bytes=28),settings)
    assert not api['has_headroom'](dict(physical_available_bytes=23,commit_available_bytes=100),settings)
    assert not api['has_headroom'](dict(physical_available_bytes=100,commit_available_bytes=27),settings)
