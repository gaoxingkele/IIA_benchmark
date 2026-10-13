"""Real file-lock boundary test, without model execution or benchmark data."""
from pathlib import Path
import subprocess
import sys
import time
from scripts.flow_matching.run_tsad_queue import exclusive_lock


def test_pending_kernel_request_is_granted_before_immediate_old_poller_reacquires(tmp_path):
    lock=tmp_path/'same_byte0.lock';waiting=tmp_path/'queued';acquired=tmp_path/'granted'
    owner=exclusive_lock(lock);assert owner is not None
    code="from pathlib import Path;import sys;from scripts.flow_matching.queued_family_lock_v1 import QueuedLease;Path(sys.argv[2]).write_text('queue');h=QueuedLease(Path(sys.argv[1]));Path(sys.argv[3]).write_text('grant');sys.stdin.buffer.read(1);h.close()"
    child=subprocess.Popen([sys.executable,'-X','utf8','-c',code,str(lock),str(waiting),str(acquired)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    stolen=None
    try:
        deadline=time.monotonic()+8
        while not waiting.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert waiting.exists();time.sleep(.2)
        assert not acquired.exists()
        owner.close();owner=None
        stolen=exclusive_lock(lock)
        assert stolen is None,'An already queued byte-zero request must not miss this release boundary'
        while not acquired.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert acquired.exists()
        child.stdin.write(b'x');child.stdin.flush();assert child.wait(timeout=8)==0
    finally:
        if stolen is not None:stolen.close()
        if owner is not None:owner.close()
        if child.poll() is None:child.kill();child.wait(timeout=8)
