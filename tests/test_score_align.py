"""score_align: tunes written as sheet music, matched against simulated performances.

Performances come from ``data/perform.py``: a pitch curve in some key at some tempo, cut into
note events by the same contour tracker the detectors use, so slides merge notes into one
event and leaps split them, as they do live.
"""

import itertools

import numpy as np
import pytest

from whistle_house.data.perform import (
    FRAME_S,
    Style,
    curve_events,
    detect_curve,
    perform,
    render_audio,
)
from whistle_house.pipeline import Pipeline, PipelineSpec, StageSpec
from whistle_house.stages.base import StageContext
from whistle_house.stages.recognizers.score_align import (
    AlignParams,
    ScoreAlignRecognizer,
    Step,
    _alignment_cost,
    _Curve,
    align,
    contour_steps,
    prepare_score,
    subdivide,
)
from whistle_house.tunes import Score, ScoreNote, Tune, TuneLibrary, fit_note_template

SR = 44100


def score(notes, bpm=100.0) -> Score:
    """[(midi pitch or None for a rest, beats), ...]"""
    return Score(tempo_bpm=bpm, notes=[ScoreNote(pitch=p, beats=b) for p, b in notes])


ARPEGGIO = score([(60, 1), (64, 1), (67, 1), (72, 2)])
TWINKLE = score([(60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 2)])
ODE = score([(64, 1), (64, 1), (65, 1), (67, 1), (67, 1), (65, 1), (64, 1), (62, 1)])
KNOCK = score([(67, 0.5), (67, 0.5), (67, 0.5), (63, 2)])
CALL = score([(72, 1), (None, 1), (67, 0.5), (69, 0.5), (72, 2)])
TUNES = {"arpeggio": ARPEGGIO, "twinkle": TWINKLE, "ode": ODE, "knock": KNOCK, "call": CALL}


LENIENT = {"off_pitch_cost": 0.75, "median_weight": 1.0}


def recognizer(tmp_path, tunes=None, **params) -> ScoreAlignRecognizer:
    lib = TuneLibrary(tmp_path / f"tunes-{len(list(tmp_path.iterdir()))}.json")
    for name, sc in (tunes or TUNES).items():
        lib.put(Tune(name=name, score=sc))
    config = ScoreAlignRecognizer.Config(**params)
    return ScoreAlignRecognizer(config, StageContext(SR, tunes=lib))


def listen(rec, times, curve, kind="whistle", chunk_frames=8):
    """Stream a curve through the contour tracker into the recognizer, as a pipeline would."""
    rec.reset()
    matches, now = [], 0.0
    for now, finished, active in detect_curve(times, curve, kind, chunk_frames):
        matches += rec.process(finished, now, active)
    return matches + rec.flush(now)


def names(matches):
    return [m.tune_name for m in matches]


def sloppy(rng, level=1.0) -> Style:
    """A random human-like performance: any key, 0.6-1.7x tempo, off pitch and off time,
    sometimes legato, sometimes with slides and scoops."""
    return Style(
        tempo_scale=float(np.exp(rng.uniform(np.log(0.6), np.log(1.7)))),
        transpose=float(rng.uniform(-7, 30)),
        intonation_sd=0.3 * level,
        drift=float(rng.normal(0, 0.4)) * level,
        timing_sd=0.12 * level,
        detach=float(rng.uniform(0.05, 0.35)),
        legato=float(rng.choice([0, 0.5, 1])),
        slide_s=float(rng.choice([0, 0.06, 0.15])),
        scoop=float(rng.choice([0, 1.5])) * level,
        hold_rests=bool(rng.random() < 0.3),
    )


# --------------------------------------------------------------------------- pieces


def test_contour_steps_follow_the_pitch():
    t = np.arange(100) * FRAME_S
    held = contour_steps(t, 70 + 0.05 * np.sin(t * 40), 0.5, FRAME_S)
    assert len(held) == 1 and held[0].pitch == pytest.approx(70, abs=0.1)
    assert held[0].end - held[0].start == pytest.approx(100 * FRAME_S)

    glide = contour_steps(t, np.linspace(60, 67, 100), 0.5, FRAME_S)
    pitches = [s.pitch for s in glide]
    assert 6 <= len(glide) <= 16 and pitches == sorted(pitches)
    assert all(a.end == pytest.approx(b.start) for a, b in itertools.pairwise(glide))

    # a single wild frame does not break a step; a bridged dropout does, and is marked
    spike = np.full(100, 70.0)
    spike[50] = 75.0
    assert len(contour_steps(t, spike, 0.5, FRAME_S)) == 1
    gappy = np.delete(t, [50, 51])
    cut = contour_steps(gappy, np.full(98, 70.0), 0.5, FRAME_S)
    assert len(cut) == 2 and cut[0].gap_before == 0 and cut[1].gap_before > 0


def test_subdivide_keeps_time_and_marks_only_real_gaps():
    steps = [Step(0.0, 1.0, 60.0, np.inf), Step(1.2, 1.25, 62.0, 0.2)]
    pieces = subdivide(steps, 0.3)
    assert len(pieces) == 5
    assert pieces[0].start == 0.0 and pieces[3].end == pytest.approx(1.0)
    assert [p.gap_before > 0 for p in pieces] == [True, False, False, False, True]


def test_prepare_score_folds_rests_into_the_note_before():
    prep = prepare_score(score([(None, 1), (60, 1), (None, 0.5), (None, 0.5), (62, 2), (None, 4)]))
    assert prep.pitches == (60.0, 62.0)
    assert prep.beats == (2.0, 2.0)  # note + its rests; trailing and leading rests dropped
    assert prep.rest_fractions == (0.5, 0.0)
    assert prepare_score(score([(None, 1)])) is None
    assert not prep.repeats_next(0)
    assert prepare_score(TWINKLE).repeats_next(0)


# --------------------------------------------------------------------------- invariance


@pytest.mark.parametrize("tempo", [0.55, 1.0, 1.9])
@pytest.mark.parametrize("transpose", [-5.3, 0.0, 26.6])
def test_matches_in_any_key_and_tempo(tmp_path, tempo, transpose):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    for name, sc in TUNES.items():
        style = Style(tempo_scale=tempo, transpose=transpose)
        (m,) = listen(rec, *perform(sc, style, rng))
        assert m.tune_name == name and m.score > 0.7
        assert m.meta["key"] == pytest.approx(transpose, abs=0.3)
        assert m.meta["tempo_bpm"] == pytest.approx(100 * tempo, rel=0.12)


def test_whistled_or_hummed(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    for kind, transpose in (("whistle", 24.0), ("hum", -14.0)):
        times, curve = perform(ODE, Style(transpose=transpose), rng)
        (m,) = listen(rec, times, curve, kind)
        assert (m.tune_name, m.kind) == ("ode", kind)
        assert all(e.kind == kind for e in m.events) and m.events


def test_tempo_outside_the_allowed_range_is_rejected(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    for tempo in (0.3, 3.2):
        assert listen(rec, *perform(TWINKLE, Style(tempo_scale=tempo), rng)) == []
    wide = recognizer(tmp_path, min_tempo_ratio=0.25, max_tempo_ratio=4.0)
    assert names(listen(wide, *perform(TWINKLE, Style(tempo_scale=3.2), rng))) == ["twinkle"]


def test_notes_are_found_where_they_were_performed(tmp_path):
    rec = recognizer(tmp_path)
    style = Style(tempo_scale=0.8, transpose=19.0, lead_s=1.0)
    (m,) = listen(rec, *perform(ARPEGGIO, style, np.random.default_rng(0)))
    beat = 60 / 100 / 0.8
    starts = [n["start"] for n in m.meta["notes"]]
    assert starts == pytest.approx([1.0 + beat * b for b in (0, 1, 2, 3)], abs=0.05)
    assert [n["pitch"] for n in m.meta["notes"]] == [60, 64, 67, 72]
    assert m.meta["notes"][0]["hz"] == pytest.approx(440 * 2 ** ((60 + 19 - 69) / 12), rel=0.01)


# --------------------------------------------------------------------------- slides


def test_slide_through_the_whole_tune_is_one_event_and_still_matches(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    for name in ("ode", "arpeggio"):
        times, curve = perform(TUNES[name], Style(legato=1.0, slide_s=0.15, transpose=20), rng)
        assert len(curve_events(times, curve)) == 1  # nothing separates the notes
        assert names(listen(rec, times, curve)) == [name]


def test_legato_leaps_and_small_steps(tmp_path):
    """Legato without slides: leaps split the events, semitone steps do not."""
    rec = recognizer(tmp_path)
    times, curve = perform(ODE, Style(legato=1.0), np.random.default_rng(0))
    assert 1 < len(curve_events(times, curve)) < 8
    assert names(listen(rec, times, curve)) == ["ode"]


def test_scoops_and_a_broken_note(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    times, curve = perform(ARPEGGIO, Style(scoop=2.0, transpose=15), rng)
    assert names(listen(rec, times, curve)) == ["arpeggio"]
    # a breath of 60 ms in the middle of the long last note
    hole = (times > times[-1] - 0.9) & (times < times[-1] - 0.84)
    assert np.isfinite(curve[hole]).all()
    curve[hole] = np.nan
    assert len(curve_events(times, curve)) == 5
    assert names(listen(rec, times, curve)) == ["arpeggio"]


# --------------------------------------------------------------------------- rests, repeats


def test_rests_are_loose(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    silent = listen(rec, *perform(CALL, Style(), rng))
    held = listen(rec, *perform(CALL, Style(hold_rests=True), rng))
    assert names(silent) == names(held) == ["call"]
    assert silent[0].score > held[0].score  # the written rest was heard


def test_repeated_notes_detached_or_slurred(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(0)
    detached = listen(rec, *perform(TWINKLE, Style(), rng))
    slurred = listen(rec, *perform(TWINKLE, Style(legato=1.0), rng))
    assert names(detached) == names(slurred) == ["twinkle"]
    assert detached[0].meta["slurred"] == 0
    assert slurred[0].meta["slurred"] == 3  # C-C, G-G, A-A ran together
    assert detached[0].score > slurred[0].score


def test_repeats_separated_by_a_dropout_the_detector_bridged(tmp_path):
    """Two frames of silence between repeated notes are bridged by the tracker (one event),
    but the missing frames still count as a re-articulation."""
    rec = recognizer(tmp_path, tunes={"knock": KNOCK})
    times, curve = perform(KNOCK, Style(legato=1.0), np.random.default_rng(0))
    beat = 0.3
    for onset in (0.3 + beat, 0.3 + 2 * beat):
        i = int(np.searchsorted(times, onset))
        curve[i - 2 : i] = np.nan
    assert len(curve_events(times, curve)) == 2
    (m,) = listen(rec, times, curve)
    assert m.meta["slurred"] == 0


# --------------------------------------------------------------------------- rejection


def test_sloppy_performances_are_recognised(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(1)
    hits = total = 0
    for _ in range(25):
        for name, sc in TUNES.items():
            total += 1
            hits += names(listen(rec, *perform(sc, sloppy(rng), rng))) == [name]
    assert hits / total >= 0.95


def test_other_melodies_are_rejected(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(2)
    false = 0
    for _ in range(200):
        notes = [
            (int(rng.integers(55, 80)), float(rng.choice([0.5, 1, 1, 2])))
            for _ in range(rng.integers(3, 10))
        ]
        melody = score(notes, bpm=float(rng.uniform(60, 160)))
        false += bool(listen(rec, *perform(melody, sloppy(rng), rng)))
    assert false <= 2


@pytest.mark.parametrize(
    "variant",
    [
        [(72, 2), (67, 1), (64, 1), (60, 1)],  # backwards
        [(60, 1), (64, 1), (69, 1), (72, 2)],  # third note a tone too high
        [(60, 1), (63, 1), (67, 1), (71, 2)],  # minor, and the top note a semitone flat
        [(60, 2), (64, 0.5), (67, 0.5), (72, 1)],  # right notes, wrong rhythm
        [(60, 1), (64, 1), (67, 1)],  # last note missing
        [(60, 1), (64, 1), (67, 1), (72, 1), (76, 2)],  # one note more
    ],
)
def test_near_misses_are_rejected(tmp_path, variant):
    rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO})
    rng = np.random.default_rng(0)
    for style in (Style(), Style(legato=1.0, slide_s=0.1), Style(tempo_scale=1.4, transpose=7)):
        assert listen(rec, *perform(score(variant), style, rng)) == []
        assert names(listen(rec, *perform(ARPEGGIO, style, rng))) == ["arpeggio"]


def test_threshold_tolerances_and_disabled_tunes(tmp_path):
    rng = np.random.default_rng(0)
    flat_third = score([(60, 1), (63, 1), (67, 1), (72, 2)])  # one note a semitone off
    strict = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, pitch_tolerance=0.4)
    loose = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, pitch_tolerance=1.0)
    assert listen(strict, *perform(flat_third, Style(), rng)) == []
    assert names(listen(loose, *perform(flat_third, Style(), rng))) == ["arpeggio"]

    lib = TuneLibrary(tmp_path / "own.json")
    tune = lib.put(Tune(name="arpeggio", score=ARPEGGIO, threshold=0.99))
    rec = ScoreAlignRecognizer(ScoreAlignRecognizer.Config(), StageContext(SR, tunes=lib))
    assert listen(rec, *perform(ARPEGGIO, Style(), rng)) == []  # per-tune threshold
    lib.put(tune.model_copy(update={"threshold": None, "enabled": False}))
    assert listen(rec, *perform(ARPEGGIO, Style(), rng)) == []
    lib.put(tune.model_copy(update={"threshold": None}))
    assert names(listen(rec, *perform(ARPEGGIO, Style(), rng))) == ["arpeggio"]
    # editing the score takes effect at once
    lib.put(tune.model_copy(update={"threshold": None, "score": ODE}))
    assert listen(rec, *perform(ARPEGGIO, Style(), rng)) == []


# --------------------------------------------------------------------------- leniency

# The pitch steps (start s, end s, MIDI pitch) the live pipeline found in a real hum of
# "Falling" (recording 56484487dcd3, Fifine mic): one legato event that sounds right, but
# the voice dips 1.5 semitones under the last note for a quarter of a second before
# settling on it, and that note then drifts up a semitone.
REAL_HUM = [
    (1.42, 1.45, 49.63),
    (1.45, 1.65, 49.66),
    (1.65, 2.19, 50.26),
    (2.19, 2.21, 49.04),
    (2.21, 2.24, 47.93),
    (2.24, 2.86, 47.20),
    (2.86, 2.88, 45.35),
    (2.88, 2.89, 44.30),
    (2.89, 2.90, 43.36),
    (2.90, 3.15, 41.73),
    (3.15, 3.91, 42.94),
    (3.91, 4.14, 43.99),
]
FALLING = score([(72, 1), (69, 1), (65, 1)], bpm=130)


def real_hum():
    times = np.arange(0, 5.1, FRAME_S)
    curve = np.full(len(times), np.nan)
    for start, end, pitch in REAL_HUM:
        curve[(times >= start) & (times < end)] = pitch
    return times, curve


def test_lenient_options_forgive_a_scoop_and_drift(tmp_path):
    def hum_score(**params):
        rec = recognizer(tmp_path, tunes={"falling": FALLING}, threshold=0.05, **params)
        (m,) = listen(rec, *real_hum(), kind="hum")
        return m.score

    strict = hum_score()
    assert strict == hum_score(off_pitch_cost=2.0, median_weight=0.0)  # the defaults: both off
    assert 0.6 < strict < 0.72
    cheap = hum_score(off_pitch_cost=0.75)
    central = hum_score(median_weight=1.0)
    assert cheap > strict + 0.08
    assert central > 0.9
    assert strict < hum_score(median_weight=0.5) < central


@pytest.mark.parametrize("variant", [[(72, 1), (65, 1), (69, 1)], [(65, 1), (69, 1), (72, 1)]])
def test_lenient_options_still_reject_another_tune(tmp_path, variant):
    rec = recognizer(tmp_path, tunes={"other": score(variant, bpm=130)}, **LENIENT)
    assert listen(rec, *real_hum(), kind="hum") == []
    rng = np.random.default_rng(0)
    rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, **LENIENT)
    flat_third = score([(60, 1), (62, 1), (67, 1), (72, 2)])  # one note a tone off
    for style in (Style(), Style(legato=1.0, slide_s=0.1)):
        assert listen(rec, *perform(flat_third, style, rng)) == []
        assert names(listen(rec, *perform(ARPEGGIO, style, rng))) == ["arpeggio"]


def test_median_matrix_is_the_weighted_median_of_each_span():
    rng = np.random.default_rng(5)
    edges = np.concatenate([[0.0], np.cumsum(rng.uniform(0.02, 0.3, 14))])
    steps = [
        Step(float(a), float(b), float(p))
        for a, b, p in zip(edges[:-1], edges[1:], rng.uniform(50, 60, 14), strict=True)
    ]
    curve = _Curve(steps)
    medians = curve.medians()
    for a in range(curve.n):
        for c in range(a + 1, curve.n + 1):
            order = np.argsort(curve.pitch[a:c], kind="stable")
            upto = np.cumsum(curve.dur[a:c][order])
            want = curve.pitch[a:c][order][np.searchsorted(upto, 0.5 * upto[-1] * (1 - 1e-12))]
            assert medians[a, c] == want


# --------------------------------------------------------------------------- quiet


def test_a_tune_inside_a_longer_melody_does_not_fire(tmp_path):
    """Default: quiet is needed before the first note and after the last."""
    rng = np.random.default_rng(0)
    intro, outro = [(65, 1), (62, 1)], [(77, 1), (74, 2)]
    arp = [(n.pitch, n.beats) for n in ARPEGGIO.notes]
    as_prefix, as_suffix = score(arp + outro), score(intro + arp)
    in_middle = score(intro + arp + outro)
    for style in (Style(), Style(legato=1.0)):
        rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO})
        for longer in (as_prefix, as_suffix, in_middle):
            assert listen(rec, *perform(longer, style, rng)) == []
        anywhere = recognizer(
            tmp_path, tunes={"arpeggio": ARPEGGIO}, quiet_before_s=0, quiet_after_s=0
        )
        for longer in (as_prefix, as_suffix, in_middle):
            assert names(listen(anywhere, *perform(longer, style, rng))) == ["arpeggio"]
        # each requirement on its own
        after_only = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, quiet_before_s=0)
        assert names(listen(after_only, *perform(as_suffix, style, rng))) == ["arpeggio"]
        assert listen(after_only, *perform(as_prefix, style, rng)) == []
        before_only = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, quiet_after_s=0)
        assert names(listen(before_only, *perform(as_prefix, style, rng))) == ["arpeggio"]
        assert listen(before_only, *perform(as_suffix, style, rng)) == []


def test_the_longer_of_two_tunes_wins_when_one_starts_the_other(tmp_path):
    longer = score([(n.pitch, n.beats) for n in ARPEGGIO.notes[:3]] + [(72, 1), (76, 1), (79, 2)])
    rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO, "longer": longer})
    rng = np.random.default_rng(0)
    assert names(listen(rec, *perform(longer, Style(), rng))) == ["longer"]
    assert names(listen(rec, *perform(ARPEGGIO, Style(), rng))) == ["arpeggio"]


def test_fires_only_after_the_quiet_and_twice_for_two_performances(tmp_path):
    rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO}, cooldown_s=0.5)
    rng = np.random.default_rng(0)
    t1, c1 = perform(ARPEGGIO, Style(tail_s=1.5), rng)
    t2, c2 = perform(ARPEGGIO, Style(transpose=5, tail_s=1.5), rng)
    times = np.concatenate([t1, t2 + t1[-1] + FRAME_S])
    matches = listen(rec, times, np.concatenate([c1, c2]))
    assert names(matches) == ["arpeggio", "arpeggio"]
    end = max(e.end for e in matches[0].events)
    assert end + 0.5 <= matches[0].time <= end + 1.0  # quiet_after_s, then settle_s
    assert matches[1].meta["key"] - matches[0].meta["key"] == pytest.approx(5, abs=0.3)


def test_minutes_of_unbroken_sound_before_a_tune(tmp_path):
    """Far more pitch steps than ``max_steps`` (a room hum that never stops, a long
    recording): evaluation stays bounded and a tune after it still matches."""
    rec = recognizer(tmp_path, tunes={"arpeggio": ARPEGGIO})
    rng = np.random.default_rng(3)
    babble = score([(int(p), 0.25) for p in rng.integers(55, 80, 600)])
    t1, c1 = perform(babble, Style(legato=1.0, tail_s=1.5), rng)
    assert len(curve_events(t1, c1)) > 200
    t2, c2 = perform(ARPEGGIO, Style(transpose=3), rng)
    times = np.concatenate([t1, t2 + t1[-1] + FRAME_S])
    matches = listen(rec, times, np.concatenate([c1, c2]))
    assert names(matches)[-1:] == ["arpeggio"]
    assert matches[-1].time > t1[-1]


# --------------------------------------------------------------------------- streaming


def test_streaming_in_any_chunk_size_gives_the_same_matches(tmp_path):
    rec = recognizer(tmp_path)
    rng = np.random.default_rng(3)
    parts, offset = [], 0.0
    for name in ("ode", "knock", "arpeggio", "call"):
        t, c = perform(TUNES[name], sloppy(rng, 0.7), rng)
        t, c = np.append(t, t[-1] + FRAME_S * np.arange(1, 130)), np.append(c, np.full(129, np.nan))
        parts.append((t + offset, c))
        offset += t[-1] + FRAME_S
    times = np.concatenate([p[0] for p in parts])
    curve = np.concatenate([p[1] for p in parts])

    def summary(matches):
        return [(m.tune_name, round(m.score, 6), round(m.meta["key"], 6)) for m in matches]

    reference = summary(listen(rec, times, curve, chunk_frames=1))
    assert [r[0] for r in reference] == ["ode", "knock", "arpeggio", "call"]
    for chunk in (8, 37, 400):
        assert summary(listen(rec, times, curve, chunk_frames=chunk)) == reference


@pytest.mark.parametrize(
    "params",
    [AlignParams(), AlignParams(off_pitch_cost=0.75, median_weight=0.5)],
    ids=["strict", "lenient"],
)
def test_alignment_cost_is_what_its_notes_cost(tmp_path, params):
    """The dynamic programme and the plain per-note cost agree on the alignment returned."""
    rng = np.random.default_rng(4)
    for name, sc in TUNES.items():
        times, curve = perform(sc, sloppy(rng), rng)
        steps: list[Step] = []
        for ev in curve_events(times, curve):
            t = np.array([p.t for p in ev.contour])
            midi = 69 + 12 * np.log2(np.array([p.hz for p in ev.contour]) / 440)
            new = contour_steps(t, midi, 0.5, FRAME_S)
            new[0].gap_before = new[0].start - steps[-1].end if steps else np.inf
            steps += new
        pieces = subdivide(steps, 0.08)
        prep = prepare_score(sc)
        al = align(prep, pieces, [0], params)
        assert al is not None, name
        plain = _alignment_cost(
            _Curve(pieces), prep, al.bounds, al.key, al.seconds_per_beat, params
        )
        assert al.cost * prep.n == pytest.approx(plain)
        assert al.bounds[0] == 0 and al.bounds[-1] == len(pieces)
        assert al.bounds == sorted(set(al.bounds))  # every note has at least one piece


# --------------------------------------------------------------------------- pipeline


def test_whole_pipeline_from_audio(tmp_path):
    """Sine-wave performance -> stft -> peak_tracker -> score_align, streamed and offline."""
    lib = TuneLibrary(tmp_path / "tunes.json")
    lib.put(Tune(name="arpeggio", score=ARPEGGIO))
    lib.put(Tune(name="ode", score=ODE))
    spec = PipelineSpec(recognizer=StageSpec(name="score_align"))
    rng = np.random.default_rng(0)
    for name, style in (
        ("arpeggio", Style(transpose=24, tempo_scale=1.3)),
        ("ode", Style(transpose=17, legato=1.0, slide_s=0.1, tempo_scale=0.8)),
    ):
        audio = render_audio(*perform(TUNES[name], style, rng), SR)
        pipe = Pipeline(spec, tunes=lib)
        offline = pipe.analyze(audio).matches
        assert names(offline) == [name]
        assert offline[0].meta["key"] == pytest.approx(style.transpose, abs=0.4)
        pipe.reset()
        streamed = []
        for i in range(0, len(audio), 777):
            streamed += pipe.step(audio[i : i + 777]).matches
        streamed += pipe.flush()[1]
        assert [(m.tune_name, round(m.score, 6)) for m in streamed] == [
            (m.tune_name, round(m.score, 6)) for m in offline
        ]


def test_each_recognizer_matches_only_its_own_kind_of_tune(tmp_path):
    """A written tune needs score_align, a tune fitted from examples needs note_template."""
    lib = TuneLibrary(tmp_path / "tunes.json")
    rng = np.random.default_rng(0)
    audio = render_audio(*perform(ARPEGGIO, Style(transpose=24), rng), SR)
    examples = [Pipeline(PipelineSpec()).analyze(audio).events]
    lib.put(Tune(name="fitted", template=fit_note_template(examples)))
    lib.put(Tune(name="written", score=ARPEGGIO))
    by_template = Pipeline(PipelineSpec(), tunes=lib).analyze(audio).matches
    by_score = Pipeline(PipelineSpec(recognizer=StageSpec(name="score_align")), tunes=lib).analyze(
        audio
    )
    assert names(by_template) == ["fitted"]
    assert names(by_score.matches) == ["written"]
