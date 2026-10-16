"""Readiness accepts new complete proofs and still refuses missing guards.

The frozen v1 test assumed every family was pending at registration time.
That assumption ceased to hold after the real Reflow native replay completed.
Keep that old file unchanged because live numerical configurations pin it.
"""
from pathlib import Path
from scripts.flow_matching import run_common105_matched577_TAB_v1 as s

def test_every_native_family_retains_full_guard_checker_after_real_progress():
    c=s.registration()
    for family in s.native.SPECS:
        entry=next(e for e in c['entries'] if e['component']==family);module=s.component(entry)
        nc=s.read(s.ROOT/module.CONFIG)
        assert entry['case'] in nc['cases']
        assert module.ready_native.__code__ is s.native.component(family).ready_native.__code__
        assert module.ready_native.__globals__['CONFIG']==s.native.CONFIGS[family]
        assert module.worker_command(entry['case'],'review')==s.native.worker_command(family,entry['case'],'review')
        ready=s.ready_case(c,entry)
        if ready is not None:
            assert ready['job']['id']==entry['case']['id']
            assert ready['result']['status']=='completed'
            assert s.sha(ready['lineage']['native_proof'])==ready['lineage']['native_sha256']
            assert ready['dataset']['grouped_splits_verified']

def test_absent_actual_fresh_review_guard_still_returns_not_ready(monkeypatch):
    c=s.registration();entry=next(e for e in c['entries'] if e['component']=='reflow')
    module=s.component(entry);nc=s.read(s.ROOT/module.CONFIG)
    attempt,output=module.paths(nc,entry['case'],'review')
    blocked=attempt/'guard_receipt.json';original=Path.exists
    monkeypatch.setattr(Path,'exists',lambda self:False if self==blocked else original(self))
    assert s.ready_case(c,entry) is None
