import os
import numpy as cpu_np


USE_CUPY = os.environ.get("USE_CUPY", "auto").lower()

if USE_CUPY in ("1", "true", "yes", "auto"):
    try:
        import cupy as xp

        GPU_ENABLED = True
    except ImportError:
        xp = cpu_np
        GPU_ENABLED = False
else:
    xp = cpu_np
    GPU_ENABLED = False


def to_device(array):
    return xp.asarray(array)


def to_cpu(array):
    if GPU_ENABLED:
        return xp.asnumpy(array)
    return array


def scalar_to_float(value):
    return float(to_cpu(value))


def scalar_to_int(value):
    return int(to_cpu(value))
