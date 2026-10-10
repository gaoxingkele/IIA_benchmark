from contextlib import contextmanager
import subprocess

from scripts.flow_matching import run_tsad_native_gpu_resource_queue_v1 as adapter


def test_transient_gpu_observation_retries_and_releases_unstarted_slot(monkeypatch):
    observations=[subprocess.TimeoutExpired('nvidia-smi',10),{'gpu_free_bytes':100},
        subprocess.TimeoutExpired('nvidia-smi',10),{'gpu_free_bytes':100},{'gpu_free_bytes':100}]
    events=[];updates=[]
    def gpu(_):
        observed=observations.pop(0)
        if isinstance(observed,Exception):raise observed
        return observed
    @contextmanager
    def slot(*_):
        events.append('acquired')
        try:yield {}
        finally:events.append('released')
    monkeypatch.setattr(adapter.time,'sleep',lambda seconds:events.append(('retry_wait',seconds)))
    api=dict(gpu_snapshot=gpu,resource_slot=slot)
    registration=dict(settings={'minimum_gpu_free_bytes':50},resource_lock='unit_fixture_gpu.lock')
    with adapter.native_slot(api,registration,updates.append):
        assert events==[('retry_wait',10),'acquired','released',('retry_wait',10),'acquired']
    assert events[-1]=='released' and not observations
    assert sum(u['state']=='waiting_gpu_observation_retry' for u in updates)==2
