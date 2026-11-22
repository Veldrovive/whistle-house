"""Thread limits for the live server (``whistle-house serve``).

The streaming pipeline does many small operations per chunk: a filterbank multiply in numpy
(OpenBLAS), then a tiny network (PyTorch or ONNX Runtime). Each library brings its own thread
pool sized to the machine, and on the Pi 5 the pools fight over the four cores: measured there,
one ``harmonic`` forward pass takes 9 ms alone and 73 ms when a numpy multiply runs between
calls. More threads give these small models nothing, so the server runs every pool with one.

* BLAS reads its thread count from the environment when the library is loaded, so
  ``limit_inference_threads`` must run before numpy is first imported. The variables are
  removed again afterwards: processes the server starts (training, reports) keep all cores.
* PyTorch is imported later, and only by the ``learned`` detector, which calls
  ``limit_torch`` after importing it.
* ONNX Runtime sessions get their thread count when they are built (the detector's
  ``threads`` param).

A variable that is already set in the environment is left alone, so
``OPENBLAS_NUM_THREADS=4 whistle-house serve`` overrides the limit.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

_BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_limited = False


def limit_inference_threads() -> None:
    """One thread per numeric library in this process. Call before numpy is imported."""
    global _limited
    _limited = True
    if "numpy" in sys.modules:
        log.warning("numpy was imported before the thread limit: BLAS keeps its own thread count")
        return
    added = [k for k in _BLAS_ENV if k not in os.environ]
    for k in added:
        os.environ[k] = "1"
    try:
        import numpy  # noqa: F401  (loads OpenBLAS, which reads the variables now)
        import scipy.linalg  # noqa: F401  (scipy ships its own copy of OpenBLAS)
    finally:
        for k in added:
            del os.environ[k]


def limit_torch(torch) -> None:
    """Apply the limit to PyTorch (no-op unless ``limit_inference_threads`` was called)."""
    if _limited and torch.get_num_threads() != 1:
        torch.set_num_threads(1)
