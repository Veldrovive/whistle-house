// Helpers for tunes written as sheet music: pitch spelling, note lengths, bars and playback.
// A pitch is a MIDI number (60 = middle C); a length is in beats (quarter note = 1).

import type { Score, ScoreNote } from '../api'

export const EMPTY_SCORE: Score = { tempo_bpm: 100, beats_per_bar: 4, notes: [] }

/** Note values offered by the editor. `glyph` is the SMuFL character in the Bravura font. */
export const NOTE_VALUES = [
  { beats: 0.25, vex: '16', glyph: '', name: 'Sixteenth note', key: '1' },
  { beats: 0.5, vex: '8', glyph: '', name: 'Eighth note', key: '2' },
  { beats: 1, vex: 'q', glyph: '', name: 'Quarter note', key: '3' },
  { beats: 2, vex: 'h', glyph: '', name: 'Half note', key: '4' },
  { beats: 4, vex: 'w', glyph: '', name: 'Whole note', key: '5' },
] as const
export const GLYPHS = { rest: '', sharp: '', flat: '', dot: '' }

/** beats -> the note value to draw (the nearest plain or dotted one). */
export function noteValue(beats: number): { base: number; dotted: boolean; vex: string } {
  let best = { base: 1, dotted: false, vex: 'q' }
  let err = Infinity
  for (const v of NOTE_VALUES) {
    for (const dotted of [false, true]) {
      const e = Math.abs(Math.log(beats / (v.beats * (dotted ? 1.5 : 1))))
      if (e < err) {
        err = e
        best = { base: v.beats, dotted, vex: v.vex }
      }
    }
  }
  return best
}

const SHARP = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
const FLAT = ['c', 'db', 'd', 'eb', 'e', 'f', 'gb', 'g', 'ab', 'a', 'bb', 'b']
const NATURAL = [0, 2, 4, 5, 7, 9, 11] // semitone of C D E F G A B
const LETTERS = 'cdefgab'

/** VexFlow key ("c#/4") for a pitch, spelled with a sharp or a flat. */
export function vexKey(n: ScoreNote): string {
  const midi = n.pitch ?? 71
  return `${(n.flat ? FLAT : SHARP)[midi % 12]}/${Math.floor(midi / 12) - 1}`
}

/** "C♯5" */
export function pitchName(n: ScoreNote): string {
  if (n.pitch === null) return 'rest'
  const [name, octave] = vexKey(n).split('/')
  return name[0].toUpperCase() + (name[1] === '#' ? '♯' : name[1] === 'b' ? '♭' : '') + octave
}

/** Staff position of a pitch as written: letters above C0 (C4 = 28, D4 = 29, ...). */
export function staffStep(n: ScoreNote): number {
  const [name, octave] = vexKey(n).split('/')
  return parseInt(octave) * 7 + LETTERS.indexOf(name[0])
}

/** The natural note at a staff position. */
export function stepToMidi(step: number): number {
  return 12 * (Math.floor(step / 7) + 1) + NATURAL[((step % 7) + 7) % 7]
}

export const MIN_STEP = 21 // C3
export const MAX_STEP = 49 // C7
export const clampStep = (s: number) => Math.max(MIN_STEP, Math.min(MAX_STEP, s))

/** The pitch named `letter` (a-g) nearest to `near`. */
export function nearestLetter(letter: string, near: number): number {
  const semitone = NATURAL[LETTERS.indexOf(letter)]
  let best = 60 + semitone
  for (let octave = 2; octave <= 7; octave++) {
    const midi = 12 * (octave + 1) + semitone
    if (Math.abs(midi - near) < Math.abs(best - near)) best = midi
  }
  return best
}

/** Index of the note after which each bar line falls, and bars that a note overflows. */
export function barLines(score: Score): { after: Set<number>; crossed: number[] } {
  const after = new Set<number>()
  const crossed: number[] = []
  let filled = 0
  let bar = 1
  score.notes.forEach((n, i) => {
    filled += n.beats
    if (filled >= score.beats_per_bar - 1e-6) {
      if (filled > score.beats_per_bar + 1e-6) crossed.push(bar)
      filled = filled % score.beats_per_bar < 1e-6 ? 0 : filled % score.beats_per_bar
      bar++
      if (i < score.notes.length - 1) after.add(i)
    }
  })
  return { after, crossed }
}

export const totalBeats = (score: Score) => score.notes.reduce((a, n) => a + n.beats, 0)

/** Play the score with a plain sine (close to a whistle). Returns a function that stops it. */
export function playScore(
  score: Score,
  opts: { transpose?: number; onNote?: (index: number | null) => void } = {},
): () => void {
  const ctx = new AudioContext()
  const out = ctx.createGain()
  out.gain.value = 0.25
  out.connect(ctx.destination)
  const beat = 60 / score.tempo_bpm
  const timers: number[] = []
  let t = 0.08
  score.notes.forEach((n, i) => {
    const length = n.beats * beat
    timers.push(window.setTimeout(() => opts.onNote?.(i), t * 1000))
    if (n.pitch !== null) {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.frequency.value = 440 * 2 ** ((n.pitch + (opts.transpose ?? 0) - 69) / 12)
      const start = ctx.currentTime + t
      const end = start + Math.max(0.05, length * 0.9)
      gain.gain.setValueAtTime(0, start)
      gain.gain.linearRampToValueAtTime(1, start + 0.02)
      gain.gain.setValueAtTime(1, Math.max(start + 0.02, end - 0.04))
      gain.gain.linearRampToValueAtTime(0, end)
      osc.connect(gain).connect(out)
      osc.start(start)
      osc.stop(end + 0.01)
    }
    t += length
  })
  let stopped = false
  const stop = () => {
    if (stopped) return
    stopped = true
    timers.forEach(clearTimeout)
    opts.onNote?.(null)
    ctx.close()
  }
  timers.push(window.setTimeout(stop, t * 1000 + 100))
  return stop
}
