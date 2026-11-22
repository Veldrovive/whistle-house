"""Optional Weights & Biases logging for ``ml/train.py`` (``--wandb-project``).

Off unless a project is given. ``wandb`` is the ``wandb`` extra (``uv sync --extra ml --extra
wandb``), installed on the training machine only, never the Pi; log in once with ``uv run wandb
login`` (or set ``WANDB_API_KEY``). ``WANDB_MODE=offline`` keeps runs local for a later
``wandb sync``.

Every chart uses ``global_step`` (optimiser steps) as its x axis instead of wandb's own step
counter: a resumed run re-does the steps after its last saved epoch, and wandb would drop
points logged at a step it has already passed. The wandb run id is kept in the training status,
so a resumed training run continues the same wandb run.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class Tracker:
    """Thin wrapper around one wandb run; every method is a no-op when tracking is off."""

    def __init__(self, run: Any = None):
        self.run = run

    @classmethod
    def start(
        cls,
        project: str | None,
        name: str,
        group: str | None,
        config: dict,
        run_id: str | None = None,
    ) -> Tracker:
        """Start (or with ``run_id``, continue) a run in ``project``; off when it is None."""
        if not project:
            return cls()
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError(
                "--wandb-project needs the wandb extra: uv sync --extra ml --extra wandb"
            ) from e
        run = wandb.init(
            project=project,
            name=name,
            group=group,
            config=config,
            id=run_id,
            resume="allow" if run_id else None,
            job_type=config.get("mix"),
        )
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        log.info("wandb run %s: %s", run.id, run.url)
        return cls(run)

    @property
    def info(self) -> dict | None:
        """``{"id", "url", "project"}`` for the training status, or None when off."""
        if self.run is None:
            return None
        return {"id": self.run.id, "url": self.run.url, "project": self.run.project}

    def log(self, step: int, values: dict, prefix: str = "") -> None:
        if self.run is None:
            return
        row = {f"{prefix}{k}": v for k, v in values.items() if isinstance(v, int | float)}
        self.run.log({"global_step": step, **row})

    def summary(self, values: dict) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self, failed: bool = False) -> None:
        if self.run is not None:
            self.run.finish(exit_code=1 if failed else 0)
            self.run = None
