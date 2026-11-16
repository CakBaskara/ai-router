import ctypes
import ctypes.wintypes as w
import threading
from typing import Callable

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
HANDLE = w.HANDLE
PSEUDOCONSOLE = 0x00020016
EXTENDED_STARTUPINFO = 0x00080000
STARTF_USESTDHANDLES = 0x100
STILL_ACTIVE = 259
INFINITE = 0xFFFFFFFF


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class STARTUPINFOEX(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("lpReserved", ctypes.c_void_p), ("lpDesktop", ctypes.c_void_p),
                ("lpTitle", ctypes.c_void_p), ("dwX", w.DWORD), ("dwY", w.DWORD), ("dwXSize", w.DWORD),
                ("dwYSize", w.DWORD), ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD),
                ("dwFillAttribute", w.DWORD), ("dwFlags", w.DWORD), ("wShowWindow", ctypes.c_ushort),
                ("cbReserved2", ctypes.c_ushort), ("lpReserved2", ctypes.c_void_p), ("hStdInput", ctypes.c_void_p),
                ("hStdOutput", ctypes.c_void_p), ("hStdError", ctypes.c_void_p), ("lpAttributeList", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE), ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


k32.CreatePseudoConsole.argtypes = [COORD, HANDLE, HANDLE, w.DWORD, ctypes.POINTER(ctypes.c_void_p)]
k32.CreatePseudoConsole.restype = ctypes.c_long
k32.ResizePseudoConsole.argtypes = [ctypes.c_void_p, COORD]
k32.ResizePseudoConsole.restype = ctypes.c_long
k32.ClosePseudoConsole.argtypes = [ctypes.c_void_p]
k32.ClosePseudoConsole.restype = None
k32.CreatePipe.argtypes = [ctypes.POINTER(HANDLE), ctypes.POINTER(HANDLE), ctypes.c_void_p, w.DWORD]
k32.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.c_size_t)]
k32.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, w.DWORD, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_void_p]
k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
k32.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL, w.DWORD, ctypes.c_void_p,
                               w.LPCWSTR, ctypes.POINTER(STARTUPINFOEX), ctypes.POINTER(PROCESS_INFORMATION)]
k32.ReadFile.argtypes = [HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k32.WriteFile.argtypes = [HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]
k32.CloseHandle.argtypes = [HANDLE]
k32.GetExitCodeProcess.argtypes = [HANDLE, ctypes.POINTER(w.DWORD)]
k32.TerminateProcess.argtypes = [HANDLE, ctypes.c_uint]
k32.WaitForSingleObject.argtypes = [HANDLE, w.DWORD]
k32.WaitForSingleObject.restype = w.DWORD


def _check(ok, what: str):
    if not ok:
        raise OSError(f"{what} gagal (kode {ctypes.get_last_error()})")


class Pty:
    def __init__(self, cmdline: str, cols: int, rows: int, on_output: Callable[[bytes], None],
                 cwd: str | None = None):
        in_read, in_write, out_read, out_write = HANDLE(), HANDLE(), HANDLE(), HANDLE()
        _check(k32.CreatePipe(ctypes.byref(in_read), ctypes.byref(in_write), None, 0), "CreatePipe")
        _check(k32.CreatePipe(ctypes.byref(out_read), ctypes.byref(out_write), None, 0), "CreatePipe")
        self.hpc = ctypes.c_void_p()
        result = k32.CreatePseudoConsole(COORD(cols, rows), in_read, out_write, 0, ctypes.byref(self.hpc))
        if result != 0:
            raise OSError(f"CreatePseudoConsole gagal ({result & 0xFFFFFFFF:#x})")
        size = ctypes.c_size_t()
        k32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        attributes = ctypes.create_string_buffer(size.value)
        _check(k32.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)), "AttributeList")
        _check(k32.UpdateProcThreadAttribute(attributes, 0, PSEUDOCONSOLE, self.hpc, ctypes.sizeof(ctypes.c_void_p),
                                             None, None), "UpdateProcThreadAttribute")
        info = STARTUPINFOEX()
        info.cb = ctypes.sizeof(STARTUPINFOEX)
        # Without this flag a parent whose own std handles are redirected passes them on, and the
        # child then talks to those pipes instead of the pseudo console.
        info.dwFlags = STARTF_USESTDHANDLES
        info.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
        self.process = PROCESS_INFORMATION()
        try:
            _check(k32.CreateProcessW(None, ctypes.create_unicode_buffer(cmdline), None, None, False,
                                      EXTENDED_STARTUPINFO, None, cwd, ctypes.byref(info),
                                      ctypes.byref(self.process)), "CreateProcess")
        finally:
            k32.DeleteProcThreadAttributeList(attributes)
            k32.CloseHandle(in_read)
            k32.CloseHandle(out_write)
        self.input, self.output = in_write, out_read
        self.on_output = on_output
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        buffer = ctypes.create_string_buffer(65536)
        count = w.DWORD()
        while k32.ReadFile(self.output, buffer, len(buffer), ctypes.byref(count), None) and count.value:
            self.on_output(buffer.raw[:count.value])

    def write(self, data: bytes):
        count = w.DWORD()
        with self.lock:
            if self.hpc is None:
                raise OSError("pseudo console sudah ditutup")
            _check(k32.WriteFile(self.input, data, len(data), ctypes.byref(count), None), "WriteFile")

    def resize(self, cols: int, rows: int):
        with self.lock:
            if self.hpc is not None:
                k32.ResizePseudoConsole(self.hpc, COORD(max(cols, 2), max(rows, 2)))

    def alive(self) -> bool:
        code = w.DWORD()
        k32.GetExitCodeProcess(self.process.hProcess, ctypes.byref(code))
        return code.value == STILL_ACTIVE

    def wait(self, timeout: float | None = None) -> bool:
        millis = INFINITE if timeout is None else int(timeout * 1000)
        return k32.WaitForSingleObject(self.process.hProcess, millis) == 0

    def close(self):
        with self.lock:
            if self.hpc is None:
                return
            if self.alive():
                k32.TerminateProcess(self.process.hProcess, 0)
            k32.ClosePseudoConsole(self.hpc)
            k32.CloseHandle(self.input)
            self.hpc = None
