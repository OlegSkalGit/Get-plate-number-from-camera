import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, wait


_IO_LOCK = threading.Lock()
_IO_EXECUTOR = None
_PENDING_FUTURES = set()


def _get_executor() -> ThreadPoolExecutor:
    global _IO_EXECUTOR
    with _IO_LOCK:
        if _IO_EXECUTOR is None:
            _IO_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nvr_io")
        return _IO_EXECUTOR


def submit_io(fn, *args, **kwargs):
    """Ставить дискову операцію у спільний обмежений пул I/O-потоків."""
    ex = _get_executor()
    try:
        fut = ex.submit(fn, *args, **kwargs)
    except RuntimeError:
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
        return None

    with _IO_LOCK:
        _PENDING_FUTURES.add(fut)

    def _on_done(f):
        with _IO_LOCK:
            _PENDING_FUTURES.discard(f)

    try:
        fut.add_done_callback(_on_done)
    except Exception:
        with _IO_LOCK:
            _PENDING_FUTURES.discard(fut)
    return fut


def flush_io(timeout: float = 5.0):
    """Очікує завершення всіх запланованих дискових операцій перед виходом."""
    with _IO_LOCK:
        futs = list(_PENDING_FUTURES)
    if futs:
        try:
            wait(futs, timeout=max(0.1, float(timeout)))
        except Exception:
            pass


def shutdown_io(wait_pending: bool = True, timeout: float = 5.0):
    global _IO_EXECUTOR
    flush_io(timeout=timeout) if wait_pending else None
    with _IO_LOCK:
        ex = _IO_EXECUTOR
        _IO_EXECUTOR = None
        _PENDING_FUTURES.clear()
    if ex is not None:
        try:
            ex.shutdown(wait=bool(wait_pending))
        except Exception:
            pass


atexit.register(flush_io, 2.0)
