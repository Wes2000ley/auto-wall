"""Small ctypes boundary for the optional C++ pixel-graph tracer."""

import ctypes
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np


class _Output(ctypes.Structure):
    _fields_ = [("points", ctypes.POINTER(ctypes.c_int32)),
                ("offsets", ctypes.POINTER(ctypes.c_int64)),
                ("point_count", ctypes.c_int64), ("path_count", ctypes.c_int64)]


_Cancel = ctypes.CFUNCTYPE(ctypes.c_int)


@lru_cache(maxsize=1)
def _library():
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    path = Path(__file__).with_name("_native") / ("auto_wall_geometry" + suffix)
    if not path.exists():
        return None
    library = ctypes.CDLL(str(path))
    library.aw_geometry_abi.argtypes = []
    library.aw_geometry_abi.restype = ctypes.c_int
    if library.aw_geometry_abi() != 1:
        raise RuntimeError("Native geometry version mismatch. Rebuild with scripts/build_native.py.")
    library.aw_trace_skeleton.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32,
                                          ctypes.c_int32, _Cancel, ctypes.POINTER(_Output)]
    library.aw_trace_skeleton.restype = ctypes.c_int
    library.aw_free_trace.argtypes = [ctypes.POINTER(_Output)]
    library.aw_free_trace.restype = None
    return library


def native_available():
    return _library() is not None


def trace_skeleton(skeleton, cancelled=None):
    """Return ordered x/y paths, or None when the optional library is absent.

    ctypes releases the GIL during native work. The cancellation callback briefly
    reacquires it; callback exceptions are propagated after native cleanup.
    """
    library = _library()
    if library is None:
        return None
    if skeleton.ndim != 2 or not skeleton.size or max(skeleton.shape) > np.iinfo(np.int32).max:
        raise ValueError("Native tracing needs a nonempty two-dimensional skeleton.")
    if skeleton.size > np.iinfo(np.int32).max:
        raise ValueError("Skeleton exceeds the native pixel index limit.")
    pixels = np.ascontiguousarray(skeleton, dtype=np.uint8)
    if not np.any(pixels):
        return []
    errors = []
    def check_cancelled():
        try:
            return int(bool(cancelled and cancelled()))
        except Exception as error:
            errors.append(error)
            return 1
    callback = _Cancel(check_cancelled) if cancelled else _Cancel()
    output = _Output()
    status = library.aw_trace_skeleton(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                      pixels.shape[1], pixels.shape[0], callback, ctypes.byref(output))
    try:
        if errors:
            raise errors[0]
        if status == 2:
            from src.core.fast_pipeline import DetectionCancelled
            raise DetectionCancelled()
        if status:
            raise RuntimeError(f"Native centerline tracing failed (status {status}).")
        if not output.path_count:
            return []
        indices = np.ctypeslib.as_array(output.points, shape=(output.point_count,))
        offsets = np.ctypeslib.as_array(output.offsets, shape=(output.path_count + 1,))
        # Allocate Python-owned coordinates before releasing the native buffers.
        points = np.column_stack((indices % pixels.shape[1], indices // pixels.shape[1])).astype(np.float32)
        return [points[begin:end] for begin, end in zip(offsets[:-1], offsets[1:])]
    finally:
        library.aw_free_trace(ctypes.byref(output))
