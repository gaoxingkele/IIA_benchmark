from contextlib import contextmanager
import json
from types import SimpleNamespace
import sys

import pytest

from scripts.flow_matching import run_grasp_protocol_queue as original
from scripts.flow_matching.run_grasp_full_fair_queue_v5 import bound_main
from scripts.flow_matching import transition_grasp_full_fairness_v5 as handoff


def test_full_original_guard_runs_and_slot_releases_before_longer_yield(tmp_path,monkeypatch):
    function=bound_main(35);namespace=function.__globals__;events=[]
    assert namespace['guarded_gpu_wait'] is original.guarded_gpu_wait
    assert namespace['complete'] is original.complete and namespace['verify'] is original.verify
    @contextmanager
    def slot(*_):
        events.append('owned');yield {};events.append('released')
    api=dict(exclusive_lock=lambda _:object(),resource_slot=slot,wait_for_headroom=lambda *_:{})
    queue=dict(state_root='run',runtime_config='runtime.json',resource_lock='gpu.lock',
        settings=dict(minimum_gpu_free_bytes=1,cpu_threads=2,gpu_index=0),python=sys.executable,
        jobs=[dict(id='full_job',output_directory='output')])
    (tmp_path/'queue.json').write_text(json.dumps(queue));(tmp_path/'runtime.json').write_text('{}')
    namespace.update(ROOT=tmp_path,verify=lambda _:None,load_controller=lambda *_:(None,api,None),
        gpu_snapshot=lambda _:{'gpu_free_bytes':10},complete=lambda _:'guarded' in events,
        guarded_gpu_wait=lambda *_:(events.append('guarded') or 0,{}),
        time=SimpleNamespace(sleep=lambda seconds:events.append(('sleep',seconds))))
    monkeypatch.setattr(namespace['subprocess'],'Popen',lambda *_,**__:SimpleNamespace(pid=123))
    monkeypatch.setattr(sys,'argv',['controller','--queue',str(tmp_path/'queue.json')])
    function()
    assert events==['owned','guarded','released',('sleep',35)]


@pytest.mark.parametrize('case',['ready','active_child','new_child','partial','reused_pid','wrong_command','wrong_queue'])
def test_default_queue_parent_handoff_requires_full_childless_identity(monkeypatch,case):
    expected=dict(pid=42,create_time=1.,command=['python','-m','default_full_native'],queue_sha256='frozen',job_ids={'full'})
    state=dict(pid=42,queue_sha256='other' if case=='wrong_queue' else 'frozen',active_job=None,active_pid=10)
    parent=SimpleNamespace(pid=42,create_time=lambda:2. if case=='reused_pid' else 1.,
        cmdline=lambda:['wrong'] if case=='wrong_command' else expected['command'],
        children=lambda recursive=True:[object()] if case=='new_child' else [])
    monkeypatch.setattr(handoff.psutil,'pid_exists',lambda _:case=='active_child')
    monkeypatch.setattr(handoff,'console_helper',lambda _:False)
    assert handoff.boundary_ready(parent,state,expected,'full',lambda _:case!='partial')==(case=='ready')


def test_short_yield_cannot_be_registered():
    with pytest.raises(ValueError,match='polling interval'):bound_main(2)
