"""Outputs: what happens when a tune is recognised.

Output *types* are registered classes with a pydantic Config (like stages, so the manager can
render a form). Output *instances* (type + params + name) live in ``<data_dir>/outputs.json``;
tunes reference instances by id. Firing never blocks the audio thread.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from whistle_house.types import TuneMatch

log = logging.getLogger(__name__)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Output(ABC):
    type: ClassVar[str]
    description: ClassVar[str] = ""
    Config: ClassVar[type[OutputConfig]] = OutputConfig

    def __init__(self, config: OutputConfig):
        self.config = config

    @abstractmethod
    def fire(self, match: TuneMatch) -> None: ...


_TYPES: dict[str, type[Output]] = {}


def register_output(cls: type[Output]) -> type[Output]:
    _TYPES[cls.type] = cls
    return cls


def list_output_types() -> list[dict]:
    return [
        {
            "type": c.type,
            "description": c.description,
            "schema": c.Config.model_json_schema(),
            "defaults": c.Config().model_dump(mode="json"),
        }
        for c in _TYPES.values()
    ]


@register_output
class LogOutput(Output):
    type = "log"
    description = "Write the match to the server log"

    def fire(self, match: TuneMatch) -> None:
        log.info("TUNE MATCH %s (score %.2f)", match.tune_name, match.score)


@register_output
class WebhookOutput(Output):
    type = "webhook"
    description = "HTTP request to a URL with the match as JSON"

    class Config(OutputConfig):
        url: str = Field("http://localhost:8123/", description="Target URL")
        method: str = Field("POST", description="HTTP method")
        headers: dict[str, str] = Field(default_factory=dict)
        timeout_s: float = 5.0

    def fire(self, match: TuneMatch) -> None:
        c = self.config
        r = httpx.request(
            c.method, c.url, json=match.to_json(), headers=c.headers, timeout=c.timeout_s
        )
        r.raise_for_status()


@register_output
class HomeAssistantEvent(Output):
    type = "home_assistant_event"
    description = "Fire a Home Assistant event (automations decide what lights do)"

    class Config(OutputConfig):
        base_url: str = Field("http://homeassistant.local:8123", description="Home Assistant URL")
        token: str = Field("", description="Long-lived access token")
        event_type: str = Field("whistle_tune", description="Event type to fire")
        extra_data: dict[str, str] = Field(
            default_factory=dict, description="Merged into event data"
        )

    def fire(self, match: TuneMatch) -> None:
        c = self.config
        data = {
            "tune_id": match.tune_id,
            "tune": match.tune_name,
            "score": match.score,
            **c.extra_data,
        }
        r = httpx.post(
            f"{c.base_url.rstrip('/')}/api/events/{c.event_type}",
            json=data,
            headers={"Authorization": f"Bearer {c.token}"},
            timeout=5.0,
        )
        r.raise_for_status()


class OutputInstance(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    type: str
    params: dict = Field(default_factory=dict)
    enabled: bool = True

    def build(self) -> Output:
        cls = _TYPES[self.type]
        return cls(cls.Config.model_validate(self.params))


class OutputStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._items: dict[str, OutputInstance] = {}
        if path.exists():
            for d in json.loads(path.read_text()):
                o = OutputInstance.model_validate(d)
                self._items[o.id] = o
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="outputs")

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([o.model_dump(mode="json") for o in self._items.values()], indent=2)
        )

    def list(self) -> list[OutputInstance]:
        return list(self._items.values())

    def get(self, oid: str) -> OutputInstance:
        return self._items[oid]

    def put(self, o: OutputInstance) -> OutputInstance:
        if o.type not in _TYPES:
            raise ValueError(f"unknown output type {o.type!r}")
        o.build()  # validate params
        with self._lock:
            self._items[o.id] = o
            self._save()
        return o

    def delete(self, oid: str) -> None:
        with self._lock:
            self._items.pop(oid, None)
            self._save()

    def fire(self, ids: list[str], match: TuneMatch) -> None:
        """Fire outputs asynchronously; errors are logged, never raised."""
        for oid in ids:
            inst = self._items.get(oid)
            if inst is None or not inst.enabled:
                continue
            self._pool.submit(self._fire_one, inst, match)

    @staticmethod
    def _fire_one(inst: OutputInstance, match: TuneMatch) -> None:
        try:
            inst.build().fire(match)
        except Exception:
            log.exception("output %s (%s) failed", inst.name, inst.type)
