"""Match the detected pitch curve against tunes written as sheet music (``Tune.score``).

A written tune says which pitches follow each other and how long each lasts in beats. A
performance is in any key, at any tempo, and its notes are not cleanly separated: a slide
through two notes is one detector event, a breath splits one note in two. So the events'
boundaries are ignored and the *score* decides where the notes are:

1. **Curve.** The recent events of one kind (whistle or hum) are one pitch-versus-time curve.
2. **Steps.** The curve is cut wherever the pitch moves about half a semitone away
   (``contour_steps``): a held note is one long step, a slide a staircase of short ones.
   Long steps are then cut into short pieces (``subdivide``), so a note may also begin in
   the middle of a held pitch.
3. **Alignment.** Dynamic programming gives every written note a run of consecutive pieces,
   in order (``align``). Time a note spends off its pitch (a scoop into it, the slide to the
   next one) costs a flat rate, so slides are allowed, but most of every note must be on its
   pitch. A repeated note that is not re-articulated (no gap) costs a little extra.
   Two options make this more lenient (both off by default): ``off_pitch_cost`` lowers that
   flat rate, and ``median_weight`` judges a note partly or wholly by its central pitch
   (the time-weighted median of its pieces) instead of by every moment of it, so a scoop,
   an overshoot after a leap or a drift at the end of a note hardly counts.
4. **Key and tempo** are free: a few candidates of each are tried, then both are refitted
   from the best alignment. Timing is measured from note onset to note onset, relative to
   the tempo, so a note followed by a rest fills the same time as a longer note.

The score is ``exp(-mean cost per note)`` in (0, 1]. A tune must normally stand alone: quiet
before its first note and after its last (``quiet_before_s`` / ``quiet_after_s``), so a short
tune inside a longer melody does not fire.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field

from whistle_house.stages.base import StageConfig, StageContext, TuneRecognizer, register
from whistle_house.tunes import Score, Tune
from whistle_house.types import TuneMatch, WhistleEvent

PITCH_COST_CAP = 2.0  # default cost of time spent off a note's pitch (2 tolerances away or more)
ON_PITCH_TOLERANCES = 2.0  # within this many tolerances a step counts as "on" the note
MIN_ON_PITCH_S = 0.04  # every note needs this much time on its pitch ...
MIN_ON_PITCH_FRACTION = 0.5  # ... and this share of its sounded time
LENGTH_RATIO_LIMIT = 2.5  # a note more than this many times too long/short never matches
LAST_NOTE_SLACK = 2.0  # the last note is held or clipped freely: its timing counts this much less
DEFAULT_FRAME_S = 512 / 44100


def hz_to_midi(hz: float | np.ndarray) -> float | np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)


def midi_to_hz(midi: float) -> float:
    return float(440.0 * 2.0 ** ((midi - 69.0) / 12.0))


# --------------------------------------------------------------------------- steps


@dataclass
class Step:
    """A stretch of the pitch curve that stays at one pitch (MIDI scale, fractional)."""

    start: float
    end: float
    pitch: float
    gap_before: float = 0.0  # silence between the previous step and this one (inf: nothing before)


def contour_steps(
    times: np.ndarray, pitch: np.ndarray, tolerance: float, frame_s: float = DEFAULT_FRAME_S
) -> list[Step]:
    """Cut one contour into steps. A new step starts when the pitch has left the current
    step's mean by more than ``tolerance`` semitones for two frames running, or after frames
    the detector bridged (a short dropout: the mark of a re-articulated note)."""
    n = len(times)
    if n == 0:
        return []
    bounds = [0]
    total, count = float(pitch[0]), 1
    for i in range(1, n):
        mean = total / count
        dropout = times[i] - times[i - 1] > 1.5 * frame_s
        if not dropout and abs(pitch[i] - mean) <= tolerance:
            total += float(pitch[i])
            count += 1
        elif dropout or i + 1 >= n or abs(pitch[i + 1] - mean) > tolerance:
            bounds.append(i)
            total, count = float(pitch[i]), 1
    bounds.append(n)
    steps: list[Step] = []
    for a, b in itertools.pairwise(bounds):
        end = float(times[b - 1]) + frame_s
        gap = float(times[a]) - steps[-1].end if steps else 0.0
        steps.append(Step(float(times[a]), end, float(np.median(pitch[a:b])), max(gap, 0.0)))
    return steps


def subdivide(steps: list[Step], piece_s: float) -> list[Step]:
    """Cut steps longer than ``piece_s`` into equal pieces, so notes can change inside a held
    pitch. Only the first piece keeps the step's ``gap_before``."""
    out: list[Step] = []
    for s in steps:
        n = max(1, int(np.ceil((s.end - s.start) / piece_s - 1e-6)))
        edges = np.linspace(s.start, s.end, n + 1)
        out += [
            Step(float(edges[m]), float(edges[m + 1]), s.pitch, s.gap_before if m == 0 else 0.0)
            for m in range(n)
        ]
    return out


# --------------------------------------------------------------------------- score


@dataclass(frozen=True)
class PreparedScore:
    """A score reduced to what is matched: its sounding notes, each with the time until the
    next one starts."""

    pitches: tuple[float, ...]
    beats: tuple[float, ...]  # onset to next onset (note + rest after it); last: the note alone
    rest_fractions: tuple[float, ...]  # share of ``beats`` that is written rest
    seconds_per_beat: float

    @property
    def n(self) -> int:
        return len(self.pitches)

    @property
    def total_beats(self) -> float:
        """First onset to the end of the last note."""
        return sum(self.beats)

    def repeats_next(self, i: int) -> bool:
        """Note i + 1 is the same pitch with no rest between: only a gap tells them apart."""
        return (
            i + 1 < self.n
            and self.pitches[i + 1] == self.pitches[i]
            and self.rest_fractions[i] == 0
        )


def prepare_score(score: Score) -> PreparedScore | None:
    """None when the score has no sounding note. Rests before the first note and after the
    last are dropped (they cannot be heard)."""
    pitches: list[float] = []
    sounded: list[float] = []
    rests: list[float] = []
    for note in score.notes:
        if note.pitch is None:
            if rests:
                rests[-1] += note.beats
        else:
            pitches.append(float(note.pitch))
            sounded.append(float(note.beats))
            rests.append(0.0)
    if not pitches:
        return None
    rests[-1] = 0.0
    beats = [b + r for b, r in zip(sounded, rests, strict=True)]
    fractions = [r / t for r, t in zip(rests, beats, strict=True)]
    return PreparedScore(tuple(pitches), tuple(beats), tuple(fractions), score.seconds_per_beat)


# --------------------------------------------------------------------------- alignment


@dataclass
class AlignParams:
    pitch_tolerance: float = 0.6  # semitones
    timing_tolerance: float = 0.25  # log units, about +-25%
    rest_tolerance: float = 0.5  # share of a note's time
    slur_penalty: float = 0.3  # cost of a repeated note that was not re-articulated
    min_tempo_ratio: float = 0.5
    max_tempo_ratio: float = 2.0
    off_pitch_cost: float = PITCH_COST_CAP  # most that time off a note's pitch costs
    median_weight: float = 0.0  # share of a note's pitch cost taken from its median pitch


@dataclass
class Alignment:
    cost: float  # mean cost per written note
    score: float  # exp(-cost), in (0, 1]
    key: float  # semitones: performed pitch - written pitch
    seconds_per_beat: float
    start: float
    end: float
    notes: list[tuple[float, float]]  # (start, end of its sound) of every written note
    slurred: int  # repeated notes that were not re-articulated
    bounds: list[int] = field(default_factory=list)  # piece index where each note starts, + end


class _Curve:
    """The pieces as arrays, with the per-(first piece, end piece) quantities every cost
    needs. Boundary ``c`` is the start of piece ``c``; boundary ``J`` is the end of the last."""

    def __init__(self, steps: list[Step]):
        self.n = len(steps)
        t0 = np.array([s.start for s in steps])
        t1 = np.array([s.end for s in steps])
        self.pitch = np.array([s.pitch for s in steps])
        self.dur = np.maximum(t1 - t0, 1e-4)
        self.bounds = np.append(t0, t1[-1])
        self.step_ends = t1
        # a note can start at boundary c without a slur if there is a gap just before it
        self.articulated = np.append(np.array([s.gap_before > 0 for s in steps]), True)
        self.voiced_to = np.concatenate([[0.0], np.cumsum(self.dur)])
        self.span = np.maximum(self.bounds[None, :] - self.bounds[:, None], 1e-6)  # [a, c]
        self.voiced = np.maximum(self.voiced_to[None, :] - self.voiced_to[:, None], 1e-6)
        self.forward = np.triu(np.ones((self.n + 1, self.n + 1), bool), 1)  # a < c
        self.log_span = np.log(self.span)
        self.silent = np.clip(1.0 - self.voiced / self.span, 0.0, 1.0)
        self._medians: np.ndarray | None = None

    def medians(self) -> np.ndarray:
        """[a, c]: time-weighted median pitch of pieces a..c-1 (the lower one when two pieces
        share the middle). Only meaningful for a < c. Computed on first use."""
        if self._medians is None:
            order = np.argsort(self.pitch, kind="stable")
            edges = np.arange(self.n + 1)
            inside = (order[None, None, :] >= edges[:, None, None]) & (
                order[None, None, :] < edges[None, :, None]
            )  # [a, c, rank]: is the piece of this pitch rank in a..c-1
            upto = np.cumsum(inside * self.dur[order][None, None, :], axis=2)
            half = 0.5 * upto[:, :, -1:] * (1.0 - 1e-12)
            self._medians = self.pitch[order][np.argmax(upto >= half, axis=2)]
        return self._medians


def _pitch_terms(curve: _Curve, target: np.ndarray, p: AlignParams):
    """target (K,) performed pitch per key -> per-piece cost*duration and on-pitch duration,
    both (K, J)."""
    dev = np.abs(curve.pitch[None, :] - target[:, None])
    cost = np.minimum(0.5 * np.square(dev / p.pitch_tolerance), p.off_pitch_cost)
    on = dev <= ON_PITCH_TOLERANCES * p.pitch_tolerance
    return cost * curve.dur[None, :], on * curve.dur[None, :]


def _blend_median(
    mean_cost: np.ndarray | float, median: np.ndarray | float, target, p: AlignParams
) -> np.ndarray | float:
    """A note's pitch cost: its time-weighted mean cost, mixed (``median_weight``) with the
    cost of its median pitch alone."""
    center = np.minimum(0.5 * np.square((median - target) / p.pitch_tolerance), p.off_pitch_cost)
    return (1.0 - p.median_weight) * mean_cost + p.median_weight * center


def _length_cost(log_ratio: np.ndarray, final: bool, p: AlignParams) -> np.ndarray:
    """Cost of a note lasting exp(log_ratio) times its written length at the tempo."""
    slack = LAST_NOTE_SLACK if final else 1.0
    cost = 0.5 * np.square(log_ratio / (p.timing_tolerance * slack))
    return np.where(np.abs(log_ratio) <= np.log(LENGTH_RATIO_LIMIT * slack), cost, np.inf)


def _alignment_cost(
    curve: _Curve, prep: PreparedScore, bounds: list[int], key: float, spb: float, p: AlignParams
) -> float:
    """Total cost of the notes starting at ``bounds`` at one key and tempo. The plain twin of
    the matrices in ``align``: used to rescore an alignment once key and tempo are refitted."""
    total = 0.0
    for i in range(prep.n):
        a, c = bounds[i], bounds[i + 1]
        cost_dur, on_dur = _pitch_terms(curve, np.array([prep.pitches[i] + key]), p)
        voiced = float(curve.voiced[a, c])
        if on_dur[0, a:c].sum() < max(MIN_ON_PITCH_S, MIN_ON_PITCH_FRACTION * voiced) - 1e-9:
            return float("inf")
        mean_cost = float(cost_dur[0, a:c].sum()) / voiced
        if p.median_weight > 0:
            mean_cost = _blend_median(mean_cost, curve.medians()[a, c], prep.pitches[i] + key, p)
        total += float(mean_cost)
        log_ratio = curve.log_span[a, c] - np.log(prep.beats[i] * spb)
        total += float(_length_cost(log_ratio, i == prep.n - 1, p))
        rest = (float(curve.silent[a, c]) - prep.rest_fractions[i]) / p.rest_tolerance
        total += 0.5 * rest**2
        if prep.repeats_next(i) and not curve.articulated[c]:
            total += p.slur_penalty
    return total


def _key_candidates(curve: _Curve, prep: PreparedScore, limit: int = 6) -> np.ndarray:
    """Keys (performed - written pitch, semitones) worth trying: every (piece, written note)
    pair votes for the key that would put that piece on that note, weighted by the piece's
    length. The true key collects the whole tune's votes."""
    votes = (curve.pitch[:, None] - np.array(prep.pitches)[None, :]).ravel()
    weight = np.repeat(curve.dur, prep.n)
    lo = np.floor(votes.min()) - 1.0
    bins = np.round((votes - lo) / 0.25).astype(int)
    hist = np.bincount(bins, weights=weight, minlength=bins.max() + 5)
    hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")  # +-0.5 semitone
    keys: list[float] = []
    for b in np.argsort(hist)[::-1]:
        key = lo + 0.25 * b
        if hist[b] > 0 and all(abs(key - k) >= 0.5 for k in keys):
            keys.append(float(key))
        if len(keys) == limit:
            break
    return np.array(keys)


def align(
    prep: PreparedScore, pieces: list[Step], starts: list[int], params: AlignParams
) -> Alignment | None:
    """Best alignment of the whole score to ``pieces[a:]`` for a first piece ``a`` in
    ``starts``; the last note ends with the last piece. None when there is no alignment."""
    if not pieces or not starts:
        return None
    p = params
    curve = _Curve(pieces)
    n_pieces, n_notes = curve.n, prep.n
    spb_fast = prep.seconds_per_beat / p.max_tempo_ratio
    spb_slow = prep.seconds_per_beat / p.min_tempo_ratio

    # Starts whose whole length could be this tune at an allowed tempo.
    length = curve.bounds[-1] - curve.bounds[starts]
    ok = (length >= 0.5 * prep.total_beats * spb_fast) & (
        length <= 1.5 * prep.total_beats * spb_slow
    )
    starts = [a for a, keep in zip(starts, ok, strict=True) if keep]
    if not starts or n_pieces < n_notes:
        return None

    # Tempo candidates: around what each start implies, or a grid when there are many starts,
    # merged where they are within 8% of each other.
    if len(starts) <= 3:
        implied = (curve.bounds[-1] - curve.bounds[starts]) / prep.total_beats
        tempos = [float(t * f) for t in implied for f in (1.0, 0.8, 1.25)]
    else:
        n_grid = int(np.ceil(np.log2(spb_slow / spb_fast) * 4)) + 1
        tempos = list(np.geomspace(spb_fast, spb_slow, n_grid))
    slots = sorted({round(float(np.log(np.clip(t, spb_fast, spb_slow))) / 0.08) for t in tempos})
    spbs = np.clip(np.exp(np.array(slots) * 0.08), spb_fast, spb_slow)
    keys = _key_candidates(curve, prep)

    n_keys, n_tempos = len(keys), len(spbs)
    size = n_pieces + 1
    need = np.maximum(MIN_ON_PITCH_S, MIN_ON_PITCH_FRACTION * curve.voiced) - 1e-9
    pitch_cost: dict[float, np.ndarray] = {}  # written pitch -> (K, a, c)

    def pitch_matrix(written: float) -> np.ndarray:
        if written not in pitch_cost:
            cost_dur, on_dur = _pitch_terms(curve, written + keys, p)
            zeros = np.zeros((n_keys, 1))
            cost_to = np.concatenate([zeros, np.cumsum(cost_dur, axis=1)], axis=1)
            on_to = np.concatenate([zeros, np.cumsum(on_dur, axis=1)], axis=1)
            on = on_to[:, None, :] - on_to[:, :, None]
            cost = (cost_to[:, None, :] - cost_to[:, :, None]) / curve.voiced[None]
            if p.median_weight > 0:
                target = (written + keys)[:, None, None]
                cost = _blend_median(cost, curve.medians()[None], target, p)
            pitch_cost[written] = np.where(curve.forward[None] & (on >= need[None]), cost, np.inf)
        return pitch_cost[written]

    # best[k, t, c]: cheapest way to have performed the notes so far with the next one
    # starting at boundary c. One move per written note, from its first piece a to the next
    # note's first piece c. (A future "missing notes" allowance is one more move here: stay
    # at the same boundary and pay a fixed penalty, with a count of misses in the state.)
    best = np.full((n_keys, n_tempos, size), np.inf)
    best[:, :, starts] = 0.0
    came_from = np.zeros((n_notes, n_keys, n_tempos, size), np.int32)
    for i in range(n_notes):
        final = i == n_notes - 1
        log_ratio = curve.log_span[None] - np.log(prep.beats[i] * spbs)[:, None, None]
        other = _length_cost(log_ratio, final, p)  # (T, a, c)
        other = other + 0.5 * np.square((curve.silent - prep.rest_fractions[i]) / p.rest_tolerance)
        total = best[:, :, :, None] + pitch_matrix(prep.pitches[i])[:, None] + other[None]
        came_from[i] = np.argmin(total, axis=2)
        best = np.take_along_axis(total, came_from[i][:, :, None, :], axis=2)[:, :, 0, :]
        if final:
            best[:, :, :n_pieces] = np.inf  # the tune ends with the last piece
        else:
            best[:, :, n_pieces] = np.inf  # more notes to come: they need pieces
            if prep.repeats_next(i):
                best = best + p.slur_penalty * ~curve.articulated
        if not np.isfinite(best).any():
            return None

    k, t = np.unravel_index(int(np.argmin(best[:, :, n_pieces])), (n_keys, n_tempos))
    bounds = [n_pieces]
    for i in range(n_notes - 1, -1, -1):
        bounds.append(int(came_from[i][k, t, bounds[-1]]))
    bounds.reverse()

    # Refit key and tempo to this alignment and keep whichever combination is cheapest.
    key0, spb0 = float(keys[k]), float(spbs[t])
    devs: list[float] = []
    weights: list[float] = []
    for i in range(n_notes):
        seg = slice(bounds[i], bounds[i + 1])
        dev = curve.pitch[seg] - prep.pitches[i]
        on = np.abs(dev - key0) <= ON_PITCH_TOLERANCES * p.pitch_tolerance
        devs += list(dev[on])
        weights += list(curve.dur[seg][on])
    order = np.argsort(devs)
    cum = np.cumsum(np.array(weights)[order])
    key1 = float(np.array(devs)[order][np.searchsorted(cum, 0.5 * cum[-1])])
    spb1 = spb0
    if n_notes > 1:
        logs = [
            curve.log_span[bounds[i], bounds[i + 1]] - np.log(prep.beats[i])
            for i in range(n_notes - 1)
        ]
        spb1 = float(np.clip(np.exp(np.mean(logs)), spb_fast, spb_slow))
    options = [(key0, spb0), (key1, spb0), (key0, spb1), (key1, spb1)]
    totals = [_alignment_cost(curve, prep, bounds, ky, sp, p) for ky, sp in options]
    pick = int(np.argmin(totals))
    key, spb = options[pick]
    cost = totals[pick] / n_notes

    return Alignment(
        cost=float(cost),
        score=float(np.exp(-cost)),
        key=key,
        seconds_per_beat=spb,
        start=float(curve.bounds[bounds[0]]),
        end=float(curve.bounds[-1]),
        notes=[
            (float(curve.bounds[bounds[i]]), float(curve.step_ends[bounds[i + 1] - 1]))
            for i in range(n_notes)
        ],
        slurred=sum(
            1
            for i in range(n_notes - 1)
            if prep.repeats_next(i) and not curve.articulated[bounds[i + 1]]
        ),
        bounds=bounds,
    )


# --------------------------------------------------------------------------- recognizer


@dataclass
class _Pending:
    """A tune that matched and is waiting for quiet after its last note."""

    tune: Tune
    alignment: Alignment
    events: list[WhistleEvent]


@dataclass
class _KindState:
    steps: list[Step] = field(default_factory=list)
    events: list[WhistleEvent] = field(default_factory=list)
    last_end: float = -np.inf  # end of the newest step ever seen (kept when steps are dropped)
    pending: _Pending | None = None


@register
class ScoreAlignRecognizer(TuneRecognizer):
    """Aligns the pitch curve to tunes written as sheet music; free in key and tempo."""

    name = "score_align"

    class Config(StageConfig):
        threshold: float = Field(
            0.5, description="Score (0-1) needed to fire, unless the tune overrides"
        )
        pitch_tolerance: float = Field(
            0.6, gt=0, description="How far off pitch a note may be, in semitones (1 std dev)"
        )
        timing_tolerance: float = Field(
            0.25,
            gt=0,
            description="How far off a note's length may be relative to the tempo "
            "(1 std dev; 0.25 is about +-25%)",
        )
        min_tempo_ratio: float = Field(
            0.5, gt=0, description="Slowest performance matched, as a multiple of the tune's tempo"
        )
        max_tempo_ratio: float = Field(
            2.0, gt=0, description="Fastest performance matched, as a multiple of the tune's tempo"
        )
        quiet_before_s: float = Field(
            0.5,
            ge=0,
            description="Silence needed before the first note, so a tune inside a longer "
            "melody does not fire (0 = a tune may start anywhere)",
        )
        quiet_after_s: float = Field(
            0.5,
            ge=0,
            description="Silence needed after the last note before the tune fires "
            "(0 = fire as soon as the last note ends)",
        )
        settle_s: float = Field(
            0.25,
            ge=0,
            description="Extra wait after quiet_after_s for the detector to report a note "
            "that has just started (covers model lookahead)",
        )
        cooldown_s: float = Field(2.0, description="Ignore further matches of a tune for this long")
        step_semitones: float = Field(
            0.5, gt=0, description="Pitch change that starts a new step of the curve"
        )
        slur_penalty: float = Field(
            0.3, ge=0, description="Cost when a repeated note is not re-articulated"
        )
        rest_tolerance: float = Field(
            0.5,
            gt=0,
            description="How loosely rests are judged: allowed error in the silent share of "
            "a note's time (1 std dev)",
        )
        off_pitch_cost: float = Field(
            PITCH_COST_CAP,
            gt=0,
            description="Most that time off a note's pitch can cost (a slide, a scoop, an "
            "overshoot). 2 = strict; about 0.75 makes such moments cheap. At least half of "
            "every note must be on pitch either way",
        )
        median_weight: float = Field(
            0.0,
            ge=0,
            le=1,
            description="How much a note is judged by its central (median) pitch instead of "
            "by every moment of it. 0 = off; 1 = only the central pitch counts, so scoops "
            "and drift within a note cost nothing",
        )
        max_steps: int = Field(
            96,
            ge=8,
            description="Most pieces of the pitch curve aligned at once (bounds the cost; "
            "longer phrases are aligned more coarsely)",
        )

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        if config.min_tempo_ratio > config.max_tempo_ratio:
            raise ValueError("score_align: min_tempo_ratio must not exceed max_tempo_ratio")
        self.params = AlignParams(
            pitch_tolerance=config.pitch_tolerance,
            timing_tolerance=config.timing_tolerance,
            rest_tolerance=config.rest_tolerance,
            slur_penalty=config.slur_penalty,
            min_tempo_ratio=config.min_tempo_ratio,
            max_tempo_ratio=config.max_tempo_ratio,
            off_pitch_cost=config.off_pitch_cost,
            median_weight=config.median_weight,
        )
        self._prepared: dict[str, tuple[str, PreparedScore | None]] = {}
        self.reset()

    def reset(self) -> None:
        self._kinds: dict[str, _KindState] = {}
        self._last_fired: dict[str, float] = {}

    # ---- tunes

    def _scores(self) -> list[tuple[Tune, PreparedScore]]:
        out = []
        for tune in self.ctx.tunes.list() if self.ctx.tunes is not None else []:
            if not tune.enabled or tune.score is None:
                continue
            stamp = tune.score.model_dump_json()
            if self._prepared.get(tune.id, ("", None))[0] != stamp:
                self._prepared[tune.id] = (stamp, prepare_score(tune.score))
            prep = self._prepared[tune.id][1]
            if prep is not None:
                out.append((tune, prep))
        return out

    # ---- curve

    def _add(self, st: _KindState, ev: WhistleEvent) -> None:
        times = np.array([pt.t for pt in ev.contour])
        pitch = hz_to_midi(np.array([pt.hz for pt in ev.contour]))
        frame_s = float(np.median(np.diff(times))) if len(times) > 1 else DEFAULT_FRAME_S
        new = contour_steps(times, pitch, self.config.step_semitones, frame_s)
        # Events of one kind follow each other; clip the frame or two a new track can overlap
        # the one it replaced.
        new = [s for s in new if s.end > st.last_end + 1e-6]
        if not new:
            return
        new[0].start = max(new[0].start, st.last_end)
        gap = new[0].start - st.last_end
        new[0].gap_before = gap if gap > 1.5 * frame_s else 0.0
        st.steps = (st.steps + new)[-4 * self.config.max_steps :]
        st.events.append(ev)
        st.last_end = new[-1].end
        st.events = [e for e in st.events if e.end > st.steps[0].start]

    def _best(self, st: _KindState, now: float) -> _Pending | None:
        """The best-scoring tune that ends with the newest step, if any clears its threshold."""
        c = self.config
        best: _Pending | None = None
        for tune, prep in self._scores():
            if now - self._last_fired.get(tune.id, -np.inf) < c.cooldown_s:
                continue
            # Pieces a quarter of the tune's shortest note at its written tempo, coarser if
            # that would be more pieces than max_steps.
            piece_s = max(0.05, 0.25 * min(prep.beats) * prep.seconds_per_beat)
            # Only the curve the tune could lie in: its length at the slowest tempo, with
            # room for timing errors. Older steps cannot be part of a match that ends now.
            window = 1.5 * prep.total_beats * prep.seconds_per_beat / c.min_tempo_ratio
            steps = [s for s in st.steps if s.end > st.last_end - window][-c.max_steps :]
            pieces = subdivide(steps, piece_s)
            # Coarsening cannot give fewer pieces than steps, so stop once pieces are as
            # long as the window and keep the newest.
            while len(pieces) > c.max_steps and piece_s < window:
                piece_s *= 1.5
                pieces = subdivide(steps, piece_s)
            pieces = pieces[-c.max_steps :]
            if c.quiet_before_s > 0:
                starts = [j for j, s in enumerate(pieces) if s.gap_before >= c.quiet_before_s]
            else:
                starts = list(range(len(pieces)))
            al = align(prep, pieces, starts, self.params)
            threshold = tune.threshold if tune.threshold is not None else c.threshold
            if al is None or al.score < threshold:
                continue
            if best is None or al.score > best.alignment.score:
                best = _Pending(tune, al, [e for e in st.events if e.end > al.start])
        return best

    def _fire(self, kind: str, st: _KindState, now: float) -> TuneMatch:
        pend = st.pending
        assert pend is not None
        al, tune = pend.alignment, pend.tune
        prep = self._prepared[tune.id][1]
        assert prep is not None
        meta = {
            "key": al.key,
            "tempo_bpm": 60.0 / al.seconds_per_beat,
            "start": al.start,
            "end": al.end,
            "slurred": al.slurred,
            "notes": [
                {"start": s, "end": e, "hz": midi_to_hz(written + al.key), "pitch": written}
                for (s, e), written in zip(al.notes, prep.pitches, strict=True)
            ],
        }
        self._last_fired[tune.id] = now
        st.pending = None
        st.steps, st.events = [], []  # consume: these notes must not start another match
        return TuneMatch(tune.id, tune.name, now, al.score, pend.events, kind, meta)

    # ---- stage

    def process(
        self, events: list[WhistleEvent], now: float, active: list[WhistleEvent] | None = None
    ) -> list[TuneMatch]:
        c = self.config
        matches: list[TuneMatch] = []
        for ev in sorted(events, key=lambda e: (e.end, e.start)):
            if not ev.contour:
                continue
            st = self._kinds.setdefault(ev.kind, _KindState())
            if st.pending is not None:
                if ev.start >= st.pending.alignment.end + c.quiet_after_s:
                    matches.append(self._fire(ev.kind, st, now))  # it had its quiet
                st.pending = None  # otherwise the phrase went on: it was not the whole tune
            self._add(st, ev)
            if st.steps:
                st.pending = self._best(st, now)
                if st.pending and c.quiet_after_s == 0:
                    matches.append(self._fire(ev.kind, st, now))
        for kind, st in self._kinds.items():
            if st.pending is None:
                continue
            end = st.pending.alignment.end
            if any(
                a.kind == kind and a.end > end and a.start < end + c.quiet_after_s
                for a in active or []
            ):
                st.pending = None  # the phrase goes on
            elif now >= end + c.quiet_after_s + c.settle_s:
                matches.append(self._fire(kind, st, now))
        return matches

    def flush(self, now: float) -> list[TuneMatch]:
        return [
            self._fire(kind, st, now) for kind, st in self._kinds.items() if st.pending is not None
        ]
