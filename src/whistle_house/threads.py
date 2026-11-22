"""Thread limits for the live server (``whistle-house serve``) and for data workers.

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

Training's example-generation and labelling workers have the same problem many times over
(one pool per worker, each sized to the machine): ``limit_worker_threads`` runs in each worker.
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


def limit_worker_threads(*_args: object) -> None:
    """One thread per numeric library in a data-generation worker process.

    Used as the ``worker_init_fn`` of the training DataLoader and the ``initializer`` of the
    labelling process pool. Each worker is one of many processes, but numpy's OpenBLAS (and
    torch's pool) size themselves to the whole machine: on a 112-core box 8 workers ran ~12
    threads each, used ~97 cores between them and starved the GPU (epochs 5.7x slower than
    with one thread each). The environment variables cover libraries loaded later in the
    worker; ``threadpoolctl`` caps the pools that are already loaded. Variables already set in
    the environment are respected.
    """
    for k in _BLAS_ENV:
        os.environ.setdefault(k, "1")
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401  (scipy ships its own copy of OpenBLAS)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # inference-only installs; training has it via the ml extra
        log.debug("threadpoolctl not installed: BLAS pools already loaded keep their size")
    else:
        threadpool_limits(limits=int(os.environ["OPENBLAS_NUM_THREADS"]))
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
