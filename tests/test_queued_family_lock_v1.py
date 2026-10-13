from pathlib import Path
import subprocess
import sys
import time
import pytest
from scripts.flow_matching.queued_family_lock_v1 import QueuedLease,resource_slot
from scripts.flow_matching.run_tsad_queue import exclusive_lock


def test_kernel_queue_interoperates_with_original_byte_zero_lock(tmp_path):
    lock=tmp_path/'family.lock';waiting=tmp_path/'waiting';acquired=tmp_path/'acquired'
    owner=exclusive_lock(lock);assert owner is not None
    code="from pathlib import Path;import sys;from scripts.flow_matching.queued_family_lock_v1 import QueuedLease;Path(sys.argv[2]).write_text('waiting');h=QueuedLease(Path(sys.argv[1]));Path(sys.argv[3]).write_text('acquired');sys.stdin.buffer.read(1);h.close()"
    child=subprocess.Popen([sys.executable,'-X','utf8','-c',code,str(lock),str(waiting),str(acquired)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        deadline=time.monotonic()+8
        while not waiting.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert waiting.exists();time.sleep(.1)
        assert child.poll() is None and not acquired.exists()
        owner.close();owner=None
        while not acquired.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert acquired.exists(),child.stderr.read() if child.poll() is not None else 'queued acquisition timed out'
        assert exclusive_lock(lock) is None
        child.stdin.write(b'x');child.stdin.flush();assert child.wait(timeout=8)==0
        next_owner=exclusive_lock(lock);assert next_owner is not None;next_owner.close()
    finally:
        if owner is not None:owner.close()
        if child.poll() is None:child.kill();child.wait(timeout=8)


def test_headroom_drop_after_grant_releases_before_memory_wait(monkeypatch):
    events=[];snapshots=iter([{'ready':True},{'ready':False},{'ready':False},{'ready':True},{'ready':True}])
    class Lease:
        def __init__(self,path):events.append('acquire')
        def close(self):events.append('close')
    def sleep(seconds):
        assert events[-1]=='close';events.append('sleep')
    monkeypatch.setattr('scripts.flow_matching.queued_family_lock_v1.time.sleep',sleep)
    api={'resource_snapshot':lambda:next(snapshots),'has_headroom':lambda s,settings:s['ready']}
    with resource_slot(Path('unused'),{'original':True},api,acquire=Lease) as snapshot:
        assert snapshot['ready'];events.append('work')
    assert events==['acquire','close','sleep','acquire','work','close']


def test_exception_after_lock_grant_cannot_leak_family_ownership():
    events=[]
    class Lease:
        def __init__(self,path):events.append('acquire')
        def close(self):events.append('close')
    calls=iter([{'ready':True},RuntimeError('snapshot failed')])
    def snapshot():
        item=next(calls)
        if isinstance(item,Exception):raise item
        return item
    with pytest.raises(RuntimeError,match='snapshot failed'):
        with resource_slot(Path('unused'),{}, {'resource_snapshot':snapshot,'has_headroom':lambda s,p:True},acquire=Lease):pass
    assert events==['acquire','close']
