"""Data-generation workers run every numeric library with one thread (threads.limit_worker_threads)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pytest

from whistle_house.data.training import _mp_context
from whistle_house.threads import limit_worker_threads

threadpoolctl = pytest.importorskip("threadpoolctl")


def _pool_sizes() -> list[int]:
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info

    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


def test_worker_initializer_limits_blas(monkeypatch):
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    with ProcessPoolExecutor(1, mp_context=_mp_context(), initializer=limit_worker_threads) as pool:
        sizes = pool.submit(_pool_sizes).result()
    assert sizes and all(n == 1 for n in sizes), sizes


def test_without_limit_pools_follow_the_machine(monkeypatch):
    """Guard for the test above: a plain worker gets more threads (skipped on 1-core boxes)."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("single core")
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    with ProcessPoolExecutor(1, mp_context=_mp_context()) as pool:
        sizes = pool.submit(_pool_sizes).result()
    assert max(sizes) > 1, sizes
