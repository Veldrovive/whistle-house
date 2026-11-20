"""Match the most recent note events against fitted per-note tune templates.

Templates are key- and tempo-invariant, so a tune fitted from whistled examples also matches
when it is hummed a few octaves lower. Whistled and hummed notes are kept apart: a tune is
matched within one kind, and the match says which (``TuneMatch.kind``).
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from whistle_house.stages.base import StageConfig, StageContext, TuneRecognizer, register
from whistle_house.tunes import Tune, note_features
from whistle_house.types import TuneMatch, WhistleEvent


def score_instance(tune: Tune, events: list[WhistleEvent], tolerance: float = 1.0) -> float:
    """exp(-0.5 * mean z^2) of the events' features under the tune template, in (0, 1]."""
    tpl = tune.template
    assert tpl is not None and len(events) == len(tpl.notes)
    feats, log_span = note_features(events)
    z = []
    for j, note in enumerate(tpl.notes):
        stats = [note.pitch, note.onset, note.log_duration, note.glide]
        for k, st in enumerate(stats):
            # note 0 pitch/onset are 0 by construction; onset of the last note is always 1.
            if (j == 0 and k in (0, 1)) or (j == len(tpl.notes) - 1 and k == 1 and j > 0):
                continue
            z.append((feats[j, k] - st.mean) / (st.std * tolerance))
    if len(tpl.notes) > 1:
        z.append((log_span - tpl.log_span.mean) / (tpl.log_span.std * tolerance))
    return float(np.exp(-0.5 * np.mean(np.square(z)))) if z else 1.0


@register
class NoteTemplateRecognizer(TuneRecognizer):
    """Key- and tempo-invariant per-note template matching over the last N notes."""

    name = "note_template"

    class Config(StageConfig):
        threshold: float = Field(
            0.35, description="Score (0-1) needed to fire, unless the tune overrides"
        )
        tolerance: float = Field(1.0, description="Multiplier on template std devs (>1 = looser)")
        max_note_gap_s: float = Field(1.5, description="Max silence between notes of one tune")
        cooldown_s: float = Field(2.0, description="Ignore further matches of a tune for this long")

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        self.reset()

    def reset(self) -> None:
        self._recent: dict[str, list[WhistleEvent]] = {}  # per event kind (whistle, hum)
        self._last_fired: dict[str, float] = {}

    def process(self, events: list[WhistleEvent], now: float) -> list[TuneMatch]:
        if not events or self.ctx.tunes is None:
            return []
        c = self.config
        matches = []
        for kind in sorted({e.kind for e in events}):
            recent = sorted(
                self._recent.get(kind, []) + [e for e in events if e.kind == kind],
                key=lambda e: e.start,
            )
            # Keep only the run of events connected by short gaps to the newest one.
            run = [recent[-1]]
            for ev in reversed(recent[:-1]):
                if run[0].start - ev.end > c.max_note_gap_s:
                    break
                run.insert(0, ev)
            recent = self._recent[kind] = run[-32:]

            for tune in self.ctx.tunes.list():
                if not tune.enabled or tune.template is None:
                    continue
                n = len(tune.template.notes)
                if len(recent) < n:
                    continue
                if now - self._last_fired.get(tune.id, -1e9) < c.cooldown_s:
                    continue
                cand = recent[-n:]
                score = score_instance(tune, cand, c.tolerance)
                if score >= (tune.threshold if tune.threshold is not None else c.threshold):
                    matches.append(TuneMatch(tune.id, tune.name, now, score, cand, kind))
        if matches:
            best = max(matches, key=lambda m: m.score)
            self._last_fired[best.tune_id] = now
            self._recent[best.kind] = []  # consume: these notes must not start another match
            return [best]
        return []
