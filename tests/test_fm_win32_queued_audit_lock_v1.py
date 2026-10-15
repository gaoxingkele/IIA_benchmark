import json
import subprocess
import sys
import pytest
from scripts.flow_matching import win32_queued_audit_lock_v1 as s
from scripts.flow_matching.run_cfm_ts_low_memory_queue_v2 import resource_api


def test_existing_file_only_and_no_lock_bytes_are_written(tmp_path):
    path=tmp_path/'metadata.lock';data=b'unit-metadata-lock';path.write_bytes(data)
    lock=s.acquire(path,timeout_seconds=2)
    assert lock.acquired and not lock.pending
    lock.close();lock.close()
    assert path.read_bytes()==data
    with pytest.raises(FileNotFoundError):s.acquire(tmp_path/'missing.lock',timeout_seconds=1)
    assert not (tmp_path/'missing.lock').exists()


CHILD=r'''
import json,sys
from scripts.flow_matching.win32_queued_audit_lock_v1 import QueuedAuditLock
q=QueuedAuditLock(sys.argv[1]);print(json.dumps(dict(queued=True,pending=q.pending)),flush=True)
q.wait(timeout_seconds=10);print(json.dumps(dict(acquired=q.acquired)),flush=True)
sys.stdin.readline();q.close();print(json.dumps(dict(released=True)),flush=True)
'''


def test_actual_queue_interoperates_with_original_msvcrt_lock_and_receives_release(tmp_path):
    path=tmp_path/'metadata.lock';path.write_bytes(b'1');api=resource_api();old=api['exclusive_lock'](path);assert old is not None
    child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u','-c',CHILD,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        ready=json.loads(child.stdout.readline());assert ready==dict(queued=True,pending=True)
        old.close();old=None
        # Kernel queued ownership is granted at release; an immediate legacy
        # nonblocking reacquisition cannot take the narrow boundary first.
        other=api['exclusive_lock'](path)
        if other is not None:other.close()
        assert other is None
        assert json.loads(child.stdout.readline())==dict(acquired=True)
        stdout,stderr=child.communicate('\n',timeout=10)
        assert child.returncode==0 and not stderr and json.loads(stdout)==dict(released=True)
        again=api['exclusive_lock'](path);assert again is not None;again.close()
    finally:
        if old is not None:old.close()
        if child.poll() is None:child.communicate('\n',timeout=12)


def test_timeout_cancels_only_own_queued_request_and_releases_all_handles(tmp_path):
    path=tmp_path/'metadata.lock';path.write_bytes(b'1');api=resource_api();old=api['exclusive_lock'](path)
    try:
        q=s.QueuedAuditLock(path);assert q.pending
        with pytest.raises(TimeoutError):q.wait(timeout_seconds=.25)
        assert q.closed and q.file is None and q.event is None
    finally:old.close()
    again=api['exclusive_lock'](path);assert again is not None;again.close()


def test_callback_exception_cancels_wait_without_unlocking_the_other_owner(tmp_path):
    path=tmp_path/'metadata.lock';path.write_bytes(b'1');api=resource_api();old=api['exclusive_lock'](path)
    try:
        q=s.QueuedAuditLock(path)
        def stop(value):raise RuntimeError('controlled metadata observer stop')
        with pytest.raises(RuntimeError):q.wait(stop)
        assert q.closed and api['exclusive_lock'](path) is None
    finally:old.close()
    again=api['exclusive_lock'](path);assert again is not None;again.close()
