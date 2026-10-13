"""Queue for the same byte-zero family lock, then recheck unchanged admission.

Windows synchronous LockFileEx waits in the kernel without acquiring an owned
lock until the previous worker releases it. This avoids missing brief boundaries
between successive original jobs. No training or resource threshold changes.
"""
from contextlib import contextmanager
import ctypes
from pathlib import Path
import sys
import time


class Overlapped(ctypes.Structure):
    _fields_=[('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
        ('Offset',ctypes.c_uint32),('OffsetHigh',ctypes.c_uint32),('hEvent',ctypes.c_void_p)]


class QueuedLease:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=path.open('a+b');self.closed=False
        try:
            if sys.platform=='win32':
                import msvcrt
                self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                self.lock=self.kernel.LockFileEx;self.unlock=self.kernel.UnlockFileEx
                self.lock.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(Overlapped)]
                self.unlock.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(Overlapped)]
                self.lock.restype=self.unlock.restype=ctypes.c_int
                self.native_handle=msvcrt.get_osfhandle(self.handle.fileno());self.overlapped=Overlapped()
                if not self.lock(self.native_handle,2,0,1,0,ctypes.byref(self.overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl
                fcntl.flock(self.handle,fcntl.LOCK_EX)
        except BaseException:
            self.handle.close();raise

    def close(self):
        if self.closed:return
        try:
            if sys.platform=='win32':
                if not self.unlock(self.native_handle,0,1,0,ctypes.byref(self.overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl
                fcntl.flock(self.handle,fcntl.LOCK_UN)
        finally:self.closed=True;self.handle.close()


@contextmanager
def resource_slot(path,settings,api,update=None,poll_seconds=10,acquire=QueuedLease):
    while True:
        memory=api['resource_snapshot']()
        if not api['has_headroom'](memory,settings):
            if update:update(dict(memory,state='waiting_for_memory',shared_lock_owned=False))
            time.sleep(poll_seconds);continue
        if update:update(dict(memory,state='waiting_for_family_lock_kernel_queue',shared_lock_owned=False))
        lease=acquire(path)
        try:
            memory=api['resource_snapshot']()
            if api['has_headroom'](memory,settings):break
        except BaseException:
            lease.close();raise
        lease.close()
        if update:update(dict(memory,state='waiting_for_memory',shared_lock_owned=False))
    try:
        if update:update(dict(memory,state='family_lock_owned_admission_rechecked',shared_lock_owned=True))
        yield memory
    finally:
        lease.close()
        if update:update(dict(shared_lock_owned=False))
