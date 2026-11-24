"""Waveform-domain augmentation + synthetic audio.

Clean whistles are recorded in a quiet room, so robustness to background sound has to be
manufactured: every training example is (clean whistle or nothing) + (random interferer at a
random SNR) + (random gain/EQ/reverb). Interferer sources are gathered in
``data/training.py`` (mic negatives, external datasets, ``data/noise/``); this module provides
the signal-level primitives and synthetic audio: whistles, hums (``synth_hum``: held harmonic
stacks with a steep roll-off), and voiced "pseudo-speech" (harmonic stacks + moving formants),
which reproduces the property of speech that fooled v1: strong, gliding harmonic peaks.

All functions take a ``np.random.Generator`` so data generation is reproducible.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


# --------------------------------------------------------------------------- primitives


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + noise scaled so that 20*log10(rms(signal)/rms(noise)) == snr_db."""
    noise = fit_length(noise, len(signal))
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return (signal + scale * noise).astype(np.float32)


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random crop or tile x to exactly n samples."""
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = int(rng.integers(0, len(x) - n + 1)) if rng is not None else 0
    return x[start : start + n].astype(np.float32)


def random_gain(x: np.ndarray, rng: np.random.Generator, lo_db=-20.0, hi_db=6.0) -> np.ndarray:
    return (x * 10 ** (rng.uniform(lo_db, hi_db) / 20)).astype(np.float32)


def _peaking_biquad(f0: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return b / den[0], den / den[0]


def random_eq(x: np.ndarray, sr: int, rng: np.random.Generator, n_bands: int = 3) -> np.ndarray:
    """A few random peaking filters: simulates different mic/room colourations."""
    for _ in range(n_bands):
        b, a = _peaking_biquad(
            rng.uniform(150, min(7000, sr / 2 - 100)), rng.uniform(-8, 8), rng.uniform(0.5, 2.0), sr
        )
        x = lfilter(b, a, x)
    return x.astype(np.float32)


def saturate(
    x: np.ndarray, rng: np.random.Generator, drive_db: tuple[float, float] = (-12.0, 12.0)
) -> np.ndarray:
    """Soft-clip like an overdriven mic/preamp: adds overtones at exact multiples of the pitch.

    This is where a real whistle's loud overtones come from (the whistle itself is nearly a pure
    tone). The pitch does not change, so f0 labels stay valid: the model has to learn that the
    new lines above the note are part of it. ``drive_db`` is the peak level going into the
    ``tanh`` (0 dB puts the 3rd harmonic about 20 dB down, -12 dB about 45 dB down); a random
    bias makes the clipping asymmetric, which adds the even harmonics. Loudness is preserved.
    """
    peak = float(np.abs(x).max())
    if peak < 1e-6:
        return x
    gain = 10 ** (rng.uniform(*drive_db) / 20)
    bias = rng.uniform(0.0, 0.3)
    y = np.tanh(gain * x / peak + bias) - np.tanh(bias)
    return (y * rms(x) / rms(y)).astype(np.float32)


def random_reverb(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying noise impulse response."""
    rt60 = rng.uniform(0.1, 0.8)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct path vs. reverb ratio
    ir /= np.sqrt(np.sum(ir**2))
    return fftconvolve(x, ir)[: len(x)].astype(np.float32)


def channel_response_db(freqs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random smooth frequency response in dB at ``freqs`` (Hz): what a microphone, its
    distance and angle, and a room do to a source.

    A tilt (up to 4 dB per octave either way), one or two shelves (up to 20 dB, corner
    80 Hz-1.5 kHz) and a few broad bumps. The range is set by a measured case: the same voice
    on a USB condenser across a room had 13-18 dB less at 250 Hz-2.5 kHz, relative to its
    fundamental, than on a laptop mic, and ~28 dB less than audiobook speech. That alone made
    a model trained without this call 10-36% of speech frames a hum (notebook 2026-11-23).
    """
    lf = np.log2(np.maximum(np.asarray(freqs, np.float64), 20.0))
    g = rng.uniform(-4.0, 4.0) * (lf - np.log2(rng.uniform(100.0, 1000.0)))
    for _ in range(int(rng.integers(1, 3))):
        fc = rng.uniform(np.log2(80.0), np.log2(1500.0))
        g += rng.uniform(-20.0, 20.0) / (1.0 + np.exp(-(lf - fc) / rng.uniform(0.15, 1.0)))
    for _ in range(int(rng.integers(0, 4))):
        fc = rng.uniform(np.log2(60.0), np.log2(6000.0))
        g += rng.uniform(-10.0, 10.0) * np.exp(-0.5 * ((lf - fc) / rng.uniform(0.15, 0.8)) ** 2)
    band = (freqs > 60) & (freqs < 4000)
    if band.any():
        g -= g[band].mean()
    return np.clip(g, -40.0, 40.0)


def random_channel(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Pass ``x`` through a random ``channel_response_db`` (zero phase). Loudness is kept."""
    if len(x) < 16 or not x.any():
        return x
    spec = np.fft.rfft(x.astype(np.float64))
    g = channel_response_db(np.fft.rfftfreq(len(x), 1.0 / sr), rng)
    y = np.fft.irfft(spec * 10 ** (g / 20), len(x))
    return (y * rms(x) / rms(y)).astype(np.float32)


def room_reverb(
    x: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    rt60_s: tuple[float, float] = (0.15, 0.9),
    drr_db: tuple[float, float] = (-6.0, 18.0),
) -> np.ndarray:
    """A source somewhere in a room: the direct sound plus a decaying tail, with a chosen
    direct-to-reverberant ratio (18 dB = close to the mic, -6 dB = across the room).

    ``random_reverb`` has almost no direct sound (20-35 dB below its tail); this one covers
    near to far. The tail is darker than the direct sound, as rooms absorb highs faster.
    Loudness is kept."""
    if not x.any():
        return x
    rt60 = rng.uniform(*rt60_s)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    b, a = butter(1, rng.uniform(1500, 8000), btype="low", fs=sr)
    tail = lfilter(b, a, tail)
    tail[: int(rng.uniform(0.002, 0.02) * sr)] = 0.0  # the first reflection takes a while
    tail *= 10 ** (-rng.uniform(*drr_db) / 20) / np.sqrt(np.sum(tail**2) + 1e-12)
    tail[0] = 1.0
    y = fftconvolve(x, tail)[: len(x)]
    return (y * rms(x) / rms(y)).astype(np.float32)


def active_level_db(x: np.ndarray, sr: int, frame_s: float = 0.05) -> float:
    """Level of the loud parts of ``x`` in dBFS: the 95th percentile of its 50 ms frame RMS.
    Unlike the plain RMS it does not depend on how much of the clip is silent."""
    n = max(1, int(frame_s * sr))
    m = len(x) // n
    if m == 0:
        return 20 * np.log10(rms(x))
    e = np.mean(np.square(x[: m * n].reshape(m, n), dtype=np.float64), axis=1)
    return float(10 * np.log10(np.percentile(e, 95) + 1e-12))


# --------------------------------------------------------------------------- synthesis


MAINS_HZ = (50.0, 60.0, 100.0, 120.0)  # mains and its second harmonic (transformers, lamps)


def synth_background(sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """A room's steady background, ``n`` samples at unit RMS: coloured noise (often heavy at
    the bottom, as room rumble is) plus up to three steady tonal sources.

    The tonal sources are what a noise gate is for and what a hum detector must not report:
    mains hum, a fan or a fridge motor. Each is a tone or a harmonic series that does not
    move: pitch within 0.1%, level within a few dB (sometimes a slow beat against a weaker
    neighbour a fraction of a Hz away). A hummed note has an onset and wanders far more.
    Measured case (Fifine, 2026-11-24): a 120 Hz tone 16 dB over its neighbours with a
    per-bin std of 1.1 dB over 30 s, carrying about as much power as the rest of the room
    noise; ``humnet-long-room2`` drew a hum note along it."""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    g = channel_response_db(freqs, rng)
    g -= rng.uniform(0.0, 6.0) * np.log2(np.maximum(freqs, 50.0) / 1000.0)  # rumble
    g[freqs < 20.0] = -80.0  # nothing a microphone passes, and it would set the level
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * 10 ** (g / 20), n)
    out = noise / max(rms(noise), 1e-12)
    t = np.arange(n) / sr
    grid = 1.0 + rng.normal(0.0, 0.002)  # everything on the mains is locked to one frequency
    for _ in range(int(rng.integers(0, 4))):
        mains = rng.random() < 0.4
        if mains:
            f0 = float(rng.choice(MAINS_HZ)) * grid
        else:
            f0 = float(2 ** rng.uniform(np.log2(45.0), np.log2(1500.0)))
        n_h = int(rng.integers(1, 9)) if f0 < 500 else int(rng.integers(1, 4))
        top = rng.uniform(-20.0, 15.0)  # strongest partial, dB re the noise's RMS
        roll = rng.uniform(0.0, 12.0)  # dB lost per harmonic
        drift = 1.0 + (0.0 if mains else rng.uniform(0.0, 1e-3)) * np.sin(
            2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 2 * np.pi)
        )
        phase = 2 * np.pi * f0 * np.cumsum(drift) / sr
        beat = rng.uniform(0.2, 3.0) if rng.random() < 0.2 else 0.0
        for k in range(1, n_h + 1):
            if k * f0 > 0.45 * sr:
                break
            amp = 10 ** ((top - roll * (k - 1) + (rng.normal(0.0, 4.0) if k > 1 else 0.0)) / 20)
            tone = np.sin(k * phase + rng.uniform(0, 2 * np.pi))
            if beat:
                tone = tone + 10 ** (-rng.uniform(10.0, 25.0) / 20) * np.sin(
                    k * phase + 2 * np.pi * beat * t + rng.uniform(0, 2 * np.pi)
                )
            out = out + np.sqrt(2.0) * amp * tone
    return (out / max(rms(out), 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------- labelled-clip augmentation
#
# For the few whistle clips recorded on the deployment mic: each function returns a new
# (audio, f0) pair, where f0 is the clip's pitch track at ``label_rate`` Hz (0 = no whistle).
# They make new melodies/pitches/tempos out of the same takes, which plain mixing cannot.
# Used on training examples only (GeneratorConfig.p_mic_*), never for validation.


def _f0_len(n_samples: int, sr: int, label_rate: int) -> int:
    return int(n_samples / sr * label_rate) + 1


def warp_clip(
    audio: np.ndarray,
    f0: np.ndarray,
    sr: int,
    label_rate: int,
    rng: np.random.Generator,
    semitones: tuple[float, float] = (-4.0, 4.0),
    wobble: float = 0.03,
    f0_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Play the clip at a different, slowly wandering speed: shifts pitch and tempo together.

    The base speed is a random shift within ``semitones`` (narrowed so every labelled pitch
    stays inside ``f0_bounds``); on top, a smooth random curve of +-``wobble`` (fraction of
    speed) adds drift like a whistler going slightly sharp/flat. f0 labels follow exactly.
    """
    lo, hi = semitones
    voiced = f0[f0 > 0].astype(np.float64)
    if f0_bounds is not None and voiced.size:
        lo = max(lo, 12 * np.log2(f0_bounds[0] / voiced.min()))
        hi = min(hi, 12 * np.log2(f0_bounds[1] / voiced.max()))
        if lo > hi:  # clip already spans the whole band: leave the pitch alone
            lo = hi = 0.0
    base = 2 ** (rng.uniform(lo, hi) / 12)
    n_in = len(audio)
    n_out = max(1, int(n_in / base))
    knots = max(2, int(n_out / sr / 0.4) + 2)  # one control point per ~0.4 s
    curve = np.interp(
        np.linspace(0, knots - 1, n_out), np.arange(knots), rng.uniform(-wobble, wobble, knots)
    )
    rate = base * (1 + curve)
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])  # input position of each output sample
    keep = pos <= n_in - 1
    pos, rate = pos[keep], rate[keep]
    out = np.interp(pos, np.arange(n_in), audio).astype(np.float32)
    # labels: output label j sits at output sample j*sr/label_rate -> input time pos/sr
    j = np.minimum(
        (np.arange(_f0_len(len(out), sr, label_rate)) * sr / label_rate).astype(int), len(out) - 1
    )
    src = np.clip(np.round(pos[j] / sr * label_rate).astype(int), 0, len(f0) - 1)
    return out, (f0[src].astype(np.float32) * rate[j]).astype(np.float32)


def reverse_clip(audio: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-reverse: a whistled melody played backwards is still a whistled melody."""
    return audio[::-1].copy(), f0[::-1].copy()


def envelope_clip(
    audio: np.ndarray, sr: int, rng: np.random.Generator, depth_db: float = 6.0
) -> np.ndarray:
    """Slow random loudness drift (+-depth_db, one control point per ~0.3 s)."""
    knots = max(2, int(len(audio) / sr / 0.3) + 2)
    db = np.interp(
        np.linspace(0, knots - 1, len(audio)),
        np.arange(knots),
        rng.uniform(-depth_db, depth_db, knots),
    )
    return (audio * 10 ** (db / 20)).astype(np.float32)


def splice_notes(
    audio: np.ndarray,
    f0: np.ndarray,
    sr: int,
    label_rate: int,
    rng: np.random.Generator,
    fade_s: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-order the clip's notes into a new melody.

    The clip is cut in the middle of each unvoiced gap (so every piece holds one note with its
    own onset/decay and room noise around it); a random number of pieces is then drawn with
    replacement and joined with short fades. Clips with a single note are returned unchanged.
    """
    v = f0 > 0
    edges = np.flatnonzero(np.diff(v.astype(np.int8)))  # label index before each on/off change
    starts = [i + 1 for i in edges if not v[i]]  # note onsets (label index)
    ends = [i + 1 for i in edges if v[i]]  # first unvoiced label after a note
    if v[0]:
        starts = [0, *starts]
    cuts = [0]
    for e in ends:  # cut halfway between a note's end and the next note's start
        nxt = [s for s in starts if s >= e]
        if nxt:
            cuts.append((e + nxt[0]) // 2)
    cuts.append(len(f0))
    pieces = [(a, b) for a, b in itertools.pairwise(cuts) if b > a]
    if len(pieces) < 2:
        return audio, f0
    k = int(rng.integers(max(1, len(pieces) // 2), 2 * len(pieces) + 1))
    order = rng.integers(0, len(pieces), k)
    spl = sr // label_rate  # samples per label (label grid is exact for the rates used)
    fade = min(int(fade_s * sr), spl)
    ramp = np.linspace(0, 1, fade, dtype=np.float32) if fade > 1 else None
    out_a, out_f = [], []
    for i in order:
        a, b = pieces[int(i)]
        seg = audio[a * spl : b * spl].astype(np.float32).copy()
        if len(seg) < spl:
            continue
        if ramp is not None and len(seg) > 2 * fade:
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        out_a.append(seg)
        out_f.append(f0[a : a + len(seg) // spl])
    if not out_a:
        return audio, f0
    new_a, new_f = np.concatenate(out_a), np.concatenate(out_f)
    pad = _f0_len(len(new_a), sr, label_rate) - len(new_f)
    return new_a, np.concatenate([new_f, np.zeros(max(0, pad), new_f.dtype)])


@dataclass
class SynthWhistle:
    audio: np.ndarray
    f0: np.ndarray  # per-sample fundamental in Hz, 0 where silent


def synth_whistle(
    sr: int,
    duration: float,
    rng: np.random.Generator,
    n_notes: int | None = None,
    overtones: bool = False,
) -> SynthWhistle:
    """Random whistled phrase: notes with glides/vibrato, breath noise and weak harmonics.

    ``overtones=True`` gives the phrase clearly visible harmonics (h2 10-40 dB down, h3 and h4
    further), as a whistle picked up loud or through a cheap mic has. The label is still the
    fundamental: these examples teach a detector not to report the overtones as notes.
    """
    n = int(duration * sr)
    audio = np.zeros(n, np.float32)
    f0 = np.zeros(n, np.float32)
    # (harmonic number, amplitude, phase) for the phrase; drawn only when asked for
    extra = (
        [
            (k, 10 ** (rng.uniform(lo, hi) / 20), rng.uniform(0, 2 * np.pi))
            for k, (lo, hi) in enumerate(((-40, -10), (-50, -18), (-60, -30)), start=2)
        ]
        if overtones
        else []
    )
    n_notes = n_notes or int(rng.integers(1, 6))
    t = 0.2 * rng.random()
    for _ in range(n_notes):
        dur = rng.uniform(0.08, 0.9)
        s, e = int(t * sr), min(n, int((t + dur) * sr))
        if e - s < int(0.05 * sr):
            break
        m = e - s
        tt = np.arange(m) / sr
        start_hz = rng.uniform(600, 3200)
        glide = rng.choice([0.0, 0.0, rng.uniform(-12, 12)])  # semitones over the note
        curve = start_hz * 2 ** (glide * (tt / max(tt[-1], 1e-3)) ** rng.uniform(0.5, 2) / 12)
        vib = rng.uniform(0, 0.3) * np.sin(2 * np.pi * rng.uniform(4, 7) * tt)  # semitones
        hz = curve * 2 ** (vib / 12)
        phase = 2 * np.pi * np.cumsum(hz) / sr
        env = np.minimum(1, np.minimum(tt / 0.02, (tt[-1] - tt) / 0.03 + 1e-3))
        env *= 1 + 0.15 * rng.standard_normal() * np.sin(2 * np.pi * 3 * tt)
        tone = np.sin(phase) + rng.uniform(0, 0.08) * np.sin(2 * phase)
        for k, a, ph in extra:
            if k * hz.max() < 0.45 * sr:
                tone += a * np.sin(k * phase + ph)
        breath = rng.uniform(0.0, 0.15) * rng.standard_normal(m)
        audio[s:e] += (env * (tone + breath) * rng.uniform(0.2, 0.8)).astype(np.float32)
        f0[s:e] = hz
        t += dur + rng.uniform(0.05, 0.6)
        if t >= duration:
            break
    return SynthWhistle(audio, f0)


def synth_hum(
    sr: int, duration: float, rng: np.random.Generator, n_notes: int | None = None
) -> SynthWhistle:
    """Random hummed phrase: a harmonic stack with a steep spectral roll-off; f0 = fundamental.

    A hum is voiced sound through a closed mouth: a glottal source low-passed by the nasal
    tract. So harmonic levels follow a fixed spectral *envelope* (flat up to a knee, then
    8-20 dB per octave down), not the harmonic number: as the pitch moves, each harmonic slides
    along that envelope. This matches the hums recorded so far (h2 1-9 dB down, h3 10-16 dB,
    h4+ 25 dB or more). Notes are held, joined either by short gaps or by a glide (legato), with
    a little vibrato and pitch jitter. Sometimes the result is high-passed, as a small mic that
    rolls off the fundamental would: the label stays the fundamental.
    """
    n = int(duration * sr)
    f0 = np.zeros(n, np.float64)  # instantaneous pitch (also between legato notes)
    amp = np.zeros(n, np.float64)
    n_notes = n_notes or int(rng.integers(1, 6))
    pitch = np.log2(rng.uniform(85, 380))  # log2 Hz of the first note
    t = 0.2 * rng.random()
    prev_end = None
    for _ in range(n_notes):
        dur = rng.uniform(0.25, 1.4)
        s, e = int(t * sr), min(n, int((t + dur) * sr))
        if e - s < int(0.12 * sr):
            break
        m = e - s
        tt = np.arange(m) / sr
        hz = np.full(m, 2.0**pitch)
        hz *= 2 ** (rng.uniform(0, 0.25) * np.sin(2 * np.pi * rng.uniform(4, 6.5) * tt) / 12)
        hz *= 2 ** (rng.uniform(-0.4, 0.4) * tt / max(tt[-1], 1e-3) / 12)  # slow drift
        env = np.minimum(1, np.minimum(tt / rng.uniform(0.03, 0.09), (tt[-1] - tt) / 0.1 + 1e-3))
        f0[s:e] = hz
        if prev_end is not None and s == prev_end[0]:  # legato: glide in from the last pitch
            g = min(m, int(rng.uniform(0.04, 0.1) * sr))
            f0[s : s + g] = np.exp(np.linspace(np.log(prev_end[1]), np.log(hz[g - 1]), g))
            env[:g] = np.maximum(env[:g], 0.7)
        amp[s:e] = env * rng.uniform(0.6, 1.0)
        legato = rng.random() < 0.4
        t += dur + (0.0 if legato else rng.uniform(0.06, 0.5))
        prev_end = (e, hz[-1]) if legato else None
        pitch = float(np.clip(pitch + rng.integers(-5, 6) / 12, np.log2(80), np.log2(420)))
        if t >= duration:
            break
    voiced = amp > 0
    f0 *= voiced
    inst = np.where(voiced, f0, 0.0) * (1 + 0.003 * rng.standard_normal(n))  # jitter
    phase = 2 * np.pi * np.cumsum(inst) / sr
    knee = rng.uniform(100, 300)  # Hz: the envelope is flat below this
    slope = rng.uniform(8, 20)  # dB per octave above it
    audio = np.zeros(n, np.float64)
    safe = np.where(voiced, f0, 1.0)
    for k in range(1, 13):
        fk = k * safe
        level = -slope * np.maximum(0.0, np.log2(fk / knee)) + rng.normal(0, 3.0) * (k > 1)
        a = 10 ** (level / 20) * (fk < 0.45 * sr)
        audio += a * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    audio = audio * amp + rng.uniform(0.0, 0.01) * rng.standard_normal(n) * amp
    if rng.random() < 0.35:  # a small mic: first-order high-pass eats into the fundamental
        b, a_ = butter(1, rng.uniform(100, 400), btype="high", fs=sr)
        audio = lfilter(b, a_, audio)
    audio = audio / (np.abs(audio).max() + 1e-9) * rng.uniform(0.2, 0.8)
    return SynthWhistle(audio.astype(np.float32), f0.astype(np.float32))


def _resonator(f: float, bw: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.exp(-np.pi * bw / sr)
    return np.array([1 - r]), np.array([1, -2 * r * np.cos(2 * np.pi * f / sr), r * r])


def synth_pseudo_speech(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Crude voiced-speech stand-in: jittery glottal pulse train through moving formants,
    with syllable-rate amplitude modulation. Not intelligible - just speech-like spectra."""
    n = int(duration * sr)
    out = np.zeros(n, np.float32)
    seg = int(0.12 * sr)
    f0_base = rng.uniform(85, 260)
    phase = 0.0
    for s in range(0, n, seg):
        m = min(seg, n - s)
        f0 = f0_base * 2 ** (rng.normal(0, 2) / 12)
        hz = f0 * (1 + 0.01 * rng.standard_normal(m))
        ph = phase + 2 * np.pi * np.cumsum(hz) / sr
        phase = ph[-1] % (2 * np.pi)
        src = (2 * ((ph / (2 * np.pi)) % 1) - 1).astype(np.float32)  # sawtooth: all harmonics
        src += 0.05 * rng.standard_normal(m).astype(np.float32)
        y = np.zeros(m, np.float32)
        for fc, bw in (
            (rng.uniform(300, 900), 80),
            (rng.uniform(900, 2400), 120),
            (rng.uniform(2300, 3300), 160),
        ):
            b, a = _resonator(fc, bw, sr)
            y += lfilter(b, a, src).astype(np.float32)
        out[s : s + m] = y
    syll = np.clip(
        np.sin(2 * np.pi * rng.uniform(3, 6) * np.arange(n) / sr + rng.uniform(0, 6)), 0, 1
    )
    out *= syll.astype(np.float32)
    return out / (np.abs(out).max() + 1e-9) * 0.5
