"""One asynchronous exclusive Windows byte-range lock, compatible with msvcrt.

The request remains pending in Windows rather than repeatedly trying an
unlocked range. No process is suspended, signalled or terminated by this class.
"""
import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path

_unfinished_cancellations = []


class Overlapped(ctypes.Structure):
    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                ('Offset', w.DWORD), ('OffsetHigh', w.DWORD), ('hEvent', w.HANDLE)]


class QueuedByteLock:
    def __init__(self, path):
        if os.name != 'nt':
            raise OSError('Windows byte-range lock required; do not substitute a different live lock')
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.acquired = False
        self.closed = False
        self.file = None
        self.event = None
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self.k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
        self.k.CreateFileW.restype = w.HANDLE
        self.k.CreateEventW.argtypes = [w.LPVOID, w.BOOL, w.BOOL, w.LPCWSTR]
        self.k.CreateEventW.restype = w.HANDLE
        self.k.LockFileEx.argtypes = [w.HANDLE, w.DWORD, w.DWORD, w.DWORD, w.DWORD, ctypes.POINTER(Overlapped)]
        self.k.LockFileEx.restype = w.BOOL
        self.k.UnlockFileEx.argtypes = [w.HANDLE, w.DWORD, w.DWORD, w.DWORD, ctypes.POINTER(Overlapped)]
        self.k.UnlockFileEx.restype = w.BOOL
        self.k.GetOverlappedResult.argtypes = [w.HANDLE, ctypes.POINTER(Overlapped), ctypes.POINTER(w.DWORD), w.BOOL]
        self.k.GetOverlappedResult.restype = w.BOOL
        self.k.CancelIoEx.argtypes = [w.HANDLE, ctypes.POINTER(Overlapped)]
        self.k.CancelIoEx.restype = w.BOOL
        self.k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.k.WaitForSingleObject.restype = w.DWORD
        self.k.CloseHandle.argtypes = [w.HANDLE]
        self.k.CloseHandle.restype = w.BOOL
        # Read/write sharing matches the existing Python a+b lock handle.
        # OPEN_ALWAYS preserves contents; no byte is read, written or truncated.
        self.file = self.k.CreateFileW(str(self.path), 0xC0000000, 0x7, None, 4, 0x40000000, None)
        if self.file == ctypes.c_void_p(-1).value:
            self.file = None
            raise ctypes.WinError(ctypes.get_last_error())
        self.event = self.k.CreateEventW(None, True, False, None)
        if not self.event:
            error = ctypes.get_last_error()
            self.k.CloseHandle(self.file)
            self.file = None
            raise ctypes.WinError(error)
        self.overlapped = Overlapped()
        self.overlapped.hEvent = self.event
        # Exclusive, offset0, length1; omit FAIL_IMMEDIATELY to queue the request.
        if self.k.LockFileEx(self.file, 0x2, 0, 1, 0, ctypes.byref(self.overlapped)):
            self.acquired = True
        else:
            error = ctypes.get_last_error()
            if error != 997:  # ERROR_IO_PENDING
                self.k.CloseHandle(self.file)
                self.k.CloseHandle(self.event)
                self.file = self.event = None
                raise ctypes.WinError(error)

    def poll(self):
        if self.closed:
            raise RuntimeError('A closed request cannot prove ownership')
        if self.acquired:
            return True
        transferred = w.DWORD()
        if self.k.GetOverlappedResult(self.file, ctypes.byref(self.overlapped), ctypes.byref(transferred), False):
            self.acquired = True
            return True
        error = ctypes.get_last_error()
        if error == 996:  # ERROR_IO_INCOMPLETE
            return False
        raise ctypes.WinError(error)

    def close(self):
        if self.closed:
            return
        if not self.acquired:
            # Cancellation can race with a successful grant. Retain the
            # OVERLAPPED/event until completion and unlock if the grant won.
            transferred = w.DWORD()
            if not self.k.CancelIoEx(self.file, ctypes.byref(self.overlapped)):
                error = ctypes.get_last_error()
                if error != 1168:  # ERROR_NOT_FOUND, possibly already complete
                    raise ctypes.WinError(error)
            result = self.k.WaitForSingleObject(self.event, 5000)
            if result != 0:
                _unfinished_cancellations.append(self)
                raise RuntimeError('Cancellation not complete; retain original request storage and handles')
            if self.k.GetOverlappedResult(self.file, ctypes.byref(self.overlapped), ctypes.byref(transferred), False):
                self.acquired = True
            else:
                error = ctypes.get_last_error()
                if error != 995:  # ERROR_OPERATION_ABORTED
                    raise ctypes.WinError(error)
        if self.acquired and not self.k.UnlockFileEx(self.file, 0, 1, 0, ctypes.byref(self.overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.k.CloseHandle(self.file)
        self.k.CloseHandle(self.event)
        self.file = self.event = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
