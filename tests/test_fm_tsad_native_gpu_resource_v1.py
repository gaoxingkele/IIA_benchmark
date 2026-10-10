import json
from types import SimpleNamespace

import pytest

from scripts.flow_matching import run_tsad_native_gpu_resource_queue_v1 as adapter


def test_full_gpu_guard_called_once_and_slot_released_after_terminal_child(tmp_path,monkeypatch):
    events=[];process=SimpleNamespace(pid=12)
    class Reservation:
        def __exit__(self,*_):events.append('slot_released')
    def guard(child,settings,api,update):
        assert child is process;events.append('full_guard');return 0,dict(resource_guard_aborted=False)
    monkeypatch.setattr(adapter.time,'sleep',lambda seconds:events.append(('yield',seconds)))
    receipt=tmp_path/'receipt.json'
    wrapped=adapter.NativeGPUChild(process,Reservation(),dict(guarded_gpu_wait=guard),
        dict(controller_yield_seconds=35),lambda _:None,receipt,dict(full_original_budget_retained=True))
    assert wrapped.poll()==0 and wrapped.poll()==0 and wrapped.wait()==0
    assert events==['full_guard','slot_released',('yield',35)]
    assert json.loads(receipt.read_text())['full_original_budget_retained']


def test_foreign_model_or_queue_rejected_before_resource_admission(tmp_path):
    queue=dict(jobs=[dict(id='original_full')]);registration=dict(model_worker='original.model')
    factory=adapter.guarded_factory({},registration,lambda _:None,tmp_path/'queue.json',queue,tmp_path)
    for command in [
        ['python','-m','wrong.model','--queue',str(tmp_path/'queue.json'),'--job-id','original_full'],
        ['python','-m','original.model','--queue',str(tmp_path/'other.json'),'--job-id','original_full'],
        ['python','-m','original.model','--queue',str(tmp_path/'queue.json'),'--job-id','wrong_seed']]:
        with pytest.raises(ValueError,match='identity differs'):factory(command)
