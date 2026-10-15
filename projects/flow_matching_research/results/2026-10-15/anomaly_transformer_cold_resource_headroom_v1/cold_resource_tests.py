"""Cold import, unchanged threshold and frozen-resource identity checks."""
import json
import os
import subprocess
import sys

import pytest

from scripts.flow_matching import run_anomaly_transformer_cold_resource_entry_v1 as subject


@pytest.mark.parametrize('role',subject.ENTRIES)
def test_exact_entry_import_and_resource_resolution_remain_numerically_cold(role):
    code=r'''
import importlib,json,sys,psutil
from scripts.flow_matching import run_anomaly_transformer_cold_resource_entry_v1 as s
api=s.install_facades();module,args,queue,count=s.ENTRIES[sys.argv[1]]
importlib.import_module(module)
from scripts.flow_matching.heavy_resources import guarded_wait
from scripts.flow_matching.heavy_scheduler_v2 import resource_slot
from scripts.flow_matching.run_tsad_queue import exclusive_lock
assert guarded_wait is api['guarded_wait'] and resource_slot is api['resource_slot']
assert exclusive_lock is api['exclusive_lock']
assert not any(m in sys.modules for m in ['numpy','pandas','torch'])
assert len(s.read(s.ROOT/queue)['jobs'])==count
assert psutil.Process().memory_info().private<128*1024**2
'''
    result=subprocess.run([sys.executable,'-B','-c',code,role],cwd=subject.ROOT,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr[-3000:]


def test_threshold_boundaries_and_original_guard_bodies_are_retained():
    code=r'''
import ast,inspect
from scripts.flow_matching import run_anomaly_transformer_cold_resource_entry_v1 as s
api=s.install_facades()
limits=dict(minimum_available_memory_bytes=32*1024**3,minimum_commit_headroom_bytes=40*1024**3)
snapshot=dict(physical_available_bytes=32*1024**3,commit_available_bytes=40*1024**3)
assert api['has_headroom'](snapshot,limits)
assert not api['has_headroom'](dict(snapshot,physical_available_bytes=32*1024**3-1),limits)
assert not api['has_headroom'](dict(snapshot,commit_available_bytes=40*1024**3-1),limits)
for name in ['has_headroom','guarded_wait','terminate_tree']:
 tree=ast.parse((s.ROOT/'scripts/flow_matching/heavy_resources.py').read_text())
 original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 actual=ast.parse(inspect.getsource(api[name])).body[0]
 assert ast.dump(original,include_attributes=False)==ast.dump(actual,include_attributes=False)
try:s.install_facades()
except RuntimeError:pass
else:raise AssertionError('Refused to replace already imported resource modules')
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr[-3000:]


def test_cached_windows_type_is_stable_across_repeated_real_snapshots():
    code=r'''
import ctypes,sys
from scripts.flow_matching import run_anomaly_transformer_cold_resource_entry_v1 as s
api=s.install_facades();snapshot=api['resource_snapshot'];snapshot()
before=len(ctypes._pointer_type_cache)
for _ in range(100):
 value=snapshot();assert value['physical_available_bytes']>0 and value['commit_available_bytes']>0
assert len(ctypes._pointer_type_cache)==before
assert snapshot.__name__=='cached_snapshot'
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr[-3000:]


def test_controller_entry_import_does_not_load_numerical_libraries():
    code='import sys;import scripts.flow_matching.run_anomaly_transformer_cold_resource_entry_v1;assert not any(m in sys.modules for m in ["numpy","pandas","torch"])'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=subject.ROOT,
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
