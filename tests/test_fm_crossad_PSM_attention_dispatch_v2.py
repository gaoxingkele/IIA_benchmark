import pytest
from scripts.flow_matching import run_crossad_PSM_attention_dispatch_v2 as dispatch


def test_wait_checks_exact_process_identity_instead_of_status_or_PID_only(monkeypatch):
    class Path:
        def exists(self):return False
    class Root:
        def __truediv__(self,other):return Path()
    class Actor:
        def create_time(self):return 123.5
        def cmdline(self):return ['python','exact-diagnostic']
    monkeypatch.setattr(dispatch.full,'ROOT',Root())
    monkeypatch.setattr(dispatch.psutil,'Process',lambda pid:Actor())
    config=dict(diagnostic_terminal='pending',diagnostic_owner=dict(pid=42,create_time=123.5,command=['python','exact-diagnostic']))
    assert dispatch.diagnostic_ready(config) is False
    config['diagnostic_owner']['create_time']=124
    with pytest.raises(ValueError):dispatch.diagnostic_ready(config)


@pytest.mark.parametrize('change',[
    dict(exit_code=1),dict(passed=False),dict(resource_guard_aborted=True),dict(diagnostic_checks=0),dict(diagnostic_checks=15)])
def test_incomplete_or_failed_diagnostics_cannot_admit_full_PSM(monkeypatch,change):
    class Path:
        def exists(self):return True
    class Root:
        def __truediv__(self,other):return Path()
    receipt=dict(exit_code=0,passed=True,resource_guard_aborted=False,diagnostic_checks=16);receipt.update(change)
    monkeypatch.setattr(dispatch.full,'ROOT',Root());monkeypatch.setattr(dispatch.full,'read',lambda path:receipt)
    with pytest.raises(ValueError):dispatch.diagnostic_ready(dict(diagnostic_terminal='terminal'))
