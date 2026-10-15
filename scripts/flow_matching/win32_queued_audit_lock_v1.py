"""Wait on the existing byte-zero audit lock using overlapped LockFileEx.

Metadata scheduling only. The file must already exist; no byte is written.
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex
"""
import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
import time


class Overlapped(ctypes.Structure):
    _fields_=[('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
        ('Offset',w.DWORD),('OffsetHigh',w.DWORD),('hEvent',w.HANDLE)]


def kernel():
    if os.name!='nt':raise RuntimeError('This handoff lock requires native Windows')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    declarations={
        'CreateFileW':([w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE],w.HANDLE),
        'CreateEventW':([w.LPVOID,w.BOOL,w.BOOL,w.LPCWSTR],w.HANDLE),
        'LockFileEx':([w.HANDLE,w.DWORD,w.DWORD,w.DWORD,w.DWORD,ctypes.POINTER(Overlapped)],w.BOOL),
        'UnlockFileEx':([w.HANDLE,w.DWORD,w.DWORD,w.DWORD,ctypes.POINTER(Overlapped)],w.BOOL),
        'GetOverlappedResult':([w.HANDLE,ctypes.POINTER(Overlapped),ctypes.POINTER(w.DWORD),w.BOOL],w.BOOL),
        'CancelIoEx':([w.HANDLE,ctypes.POINTER(Overlapped)],w.BOOL),
        'WaitForSingleObject':([w.HANDLE,w.DWORD],w.DWORD),
        'CloseHandle':([w.HANDLE],w.BOOL)}
    for name,(args,result) in declarations.items():getattr(k,name).argtypes=args;getattr(k,name).restype=result
    return k


class QueuedAuditLock:
    def __init__(self,path):
        self.k=kernel();self.path=Path(path).resolve(strict=True);self.closed=False;self.acquired=False;self.pending=False
        self.file=None;self.event=None;self.overlapped=Overlapped()
        try:
            # OPEN_EXISTING, FILE_FLAG_OVERLAPPED; read/write sharing retains
            # compatibility with the original append-mode msvcrt lock helper.
            self.file=self.k.CreateFileW(str(self.path),0xC0000000,0x7,None,3,0x40000000,None)
            if self.file in (None,ctypes.c_void_p(-1).value):self.file=None;raise ctypes.WinError(ctypes.get_last_error())
            self.event=self.k.CreateEventW(None,True,False,None)
            if not self.event:raise ctypes.WinError(ctypes.get_last_error())
            self.overlapped.hEvent=self.event
            if self.k.LockFileEx(self.file,2,0,1,0,ctypes.byref(self.overlapped)):
                self.acquired=True
            else:
                error=ctypes.get_last_error()
                if error!=997:raise ctypes.WinError(error)
                self.pending=True
        except BaseException:self.close();raise

    def wait(self,update=lambda value:None,timeout_seconds=None):
        started=time.monotonic()
        try:
            while not self.acquired:
                if timeout_seconds is not None and time.monotonic()-started>=timeout_seconds:raise TimeoutError('Queued metadata lock wait expired')
                update(dict(state='waiting_queued_actual_audit_lock_no_process_stopped',queued_seconds=time.monotonic()-started))
                code=self.k.WaitForSingleObject(self.event,200)
                if code==258:continue
                if code!=0:raise ctypes.WinError(ctypes.get_last_error())
                transferred=w.DWORD()
                if not self.k.GetOverlappedResult(self.file,ctypes.byref(self.overlapped),ctypes.byref(transferred),False):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.pending=False;self.acquired=True
            return self
        except BaseException:self.close();raise

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.pending and self.file:
                # The OVERLAPPED memory/event must stay alive until cancellation
                # or a concurrent grant actually completes, before freeing them.
                self.k.CancelIoEx(self.file,ctypes.byref(self.overlapped))
                transferred=w.DWORD()
                complete=self.k.GetOverlappedResult(self.file,ctypes.byref(self.overlapped),ctypes.byref(transferred),True)
                if complete:self.acquired=True
                self.pending=False
            if self.acquired and self.file:
                self.k.UnlockFileEx(self.file,0,1,0,ctypes.byref(self.overlapped));self.acquired=False
        finally:
            if self.file:self.k.CloseHandle(self.file);self.file=None
            if self.event:self.k.CloseHandle(self.event);self.event=None


def acquire(path,update=lambda value:None,timeout_seconds=None):
    return QueuedAuditLock(path).wait(update,timeout_seconds)
