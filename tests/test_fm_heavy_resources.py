from scripts.flow_matching import heavy_resources as h


def test_start_gate_requires_both_physical_and_system_commit_headroom():
    settings={'minimum_available_memory_bytes':24,'minimum_commit_headroom_bytes':16}
    assert h.has_headroom({'physical_available_bytes':24,'commit_available_bytes':16},settings)
    assert not h.has_headroom({'physical_available_bytes':100,'commit_available_bytes':15},settings)
    assert not h.has_headroom({'physical_available_bytes':23,'commit_available_bytes':100},settings)


def test_shared_lock_waits_for_actual_ownership_and_releases(monkeypatch,tmp_path):
    class Handle:
        closed=False
        def close(self):self.closed=True
    handle=Handle(); candidates=iter([None,None,handle]); observed=[]
    monkeypatch.setattr(h,'exclusive_lock',lambda path:next(candidates))
    monkeypatch.setattr(h.time,'sleep',lambda duration:None)
    with h.waited_lock(tmp_path/'lock',observed.append):
        assert not handle.closed
    assert handle.closed and len(observed)==2


def test_resource_guard_stops_only_its_child_before_commit_exhaustion(monkeypatch):
    class Process:
        pid=-99
        stopped=False
        def poll(self):return 1 if self.stopped else None
        def wait(self):return 1
    process=Process()
    monkeypatch.setattr(h,'resource_snapshot',lambda:{'physical_available_bytes':100,'commit_available_bytes':7})
    monkeypatch.setattr(h.psutil,'Process',lambda pid:(_ for _ in ()).throw(h.psutil.NoSuchProcess(pid)))
    monkeypatch.setattr(h,'terminate_tree',lambda child:setattr(child,'stopped',True))
    code,record=h.guarded_wait(process,{'emergency_commit_headroom_bytes':8})
    assert code==1 and record['resource_guard_aborted'] and process.stopped
