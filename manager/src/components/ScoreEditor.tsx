import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Accidental, BarNote, Beam, Dot, Formatter, Renderer, Stave, StaveNote, Voice, VoiceMode } from 'vexflow/bravura'
import type { Score, ScoreNote } from '../api'
import {
  GLYPHS,
  NOTE_VALUES,
  barLines,
  clampStep,
  nearestLetter,
  noteValue,
  pitchName,
  playScore,
  staffStep,
  stepToMidi,
  totalBeats,
  vexKey,
} from '../lib/score'

const PAPER_INK = '#1d1f24'
const SELECTED = '#1a6fd6'
const PLAYING = '#12915a'
const HIT_PX = 15 // how close (horizontally) a click must be to a note to pick it
const E4_STEP = 30 // staff position of the bottom line of the treble staff

type Layout = {
  xs: number[] // x of each note's head centre
  lineY: (line: number) => number
  yToStep: (y: number) => number
}

/** Sheet-music editor for a tune: click the staff to add notes, drag or use the arrow keys
 *  to change them, and play the result. The staff is drawn by VexFlow. */
export function ScoreEditor({ score, onChange }: { score: Score; onChange: (s: Score) => void }) {
  const paper = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const layout = useRef<Layout | null>(null)
  const drag = useRef<number | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [value, setValue] = useState({ base: 1, dotted: false }) // length of the next note
  const [ghost, setGhost] = useState<{ x: number; y: number } | null>(null)
  const [playing, setPlaying] = useState<number | null>(null)
  const [transpose, setTranspose] = useState(12)
  const [fonts, setFonts] = useState(0)
  const [width, setWidth] = useState(700)
  const stopPlayback = useRef<(() => void) | null>(null)
  const notes = score.notes
  const beats = value.base * (value.dotted ? 1.5 : 1)

  // the music font loads asynchronously: draw again once it is there
  useEffect(() => {
    let live = true
    document.fonts.load('30px Bravura').then(() => live && setFonts((n) => n + 1))
    return () => {
      live = false
    }
  }, [])
  useEffect(() => {
    const el = scroller.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  useEffect(() => () => stopPlayback.current?.(), [])

  // ---- drawing
  useLayoutEffect(() => {
    const host = paper.current
    if (!host) return
    host.querySelectorAll('svg').forEach((s) => s.remove())
    const bars = barLines(score)
    const noteSpace = notes.reduce((a, n) => a + 38 + 16 * Math.sqrt(n.beats), 0) + bars.after.size * 18
    const lead = 96 // clef + time signature
    const w = Math.max(width, lead + noteSpace + 120)
    const renderer = new Renderer(host, Renderer.Backends.SVG)
    renderer.resize(w, 190)
    const ctx = renderer.getContext()
    ctx.setFillStyle(PAPER_INK)
    ctx.setStrokeStyle(PAPER_INK)
    const stave = new Stave(8, 40, w - 16)
    stave.addClef('treble').addTimeSignature(`${score.beats_per_bar}/4`)
    stave.setContext(ctx).draw()

    const staveNotes = notes.map((n, i) => {
      const v = noteValue(n.beats)
      const sn = new StaveNote({
        keys: [n.pitch === null ? 'b/4' : vexKey(n)],
        duration: v.vex + (n.pitch === null ? 'r' : ''),
        dots: v.dotted ? 1 : 0,
        autoStem: true,
      })
      if (v.dotted) Dot.buildAndAttach([sn], { all: true })
      const colour = i === playing ? PLAYING : i === selected ? SELECTED : null
      if (colour) sn.setStyle({ fillStyle: colour, strokeStyle: colour })
      return sn
    })
    if (staveNotes.length) {
      const tickables: (StaveNote | BarNote)[] = []
      const beams: Beam[] = []
      let bar: StaveNote[] = []
      const closeBar = () => {
        beams.push(...Beam.generateBeams(bar))
        bar = []
      }
      staveNotes.forEach((sn, i) => {
        tickables.push(sn)
        bar.push(sn)
        if (bars.after.has(i)) {
          tickables.push(new BarNote())
          closeBar()
        }
      })
      closeBar()
      const voice = new Voice({ numBeats: 4, beatValue: 4 }).setMode(VoiceMode.SOFT)
      voice.addTickables(tickables)
      Accidental.applyAccidentals([voice], 'C')
      new Formatter().joinVoices([voice]).format([voice], Math.max(60, noteSpace))
      voice.draw(ctx, stave)
      beams.forEach((b) => b.setContext(ctx).draw())
    }
    const top = stave.getYForLine(0)
    const spacing = stave.getSpacingBetweenLines()
    layout.current = {
      xs: staveNotes.map((sn) => {
        const box = sn.getBoundingBox()
        return notes[staveNotes.indexOf(sn)].pitch === null ? box.getX() + box.getW() / 2 : sn.getNoteHeadBeginX() + 6
      }),
      lineY: (line) => top + line * spacing,
      yToStep: (y) => clampStep(E4_STEP + Math.round(((top + 4 * spacing - y) / spacing) * 2)),
    }
  }, [score, notes, selected, playing, fonts, width])

  // ---- editing
  const edit = useCallback(
    (next: ScoreNote[], select: number | null) => {
      onChange({ ...score, notes: next })
      setSelected(select)
    },
    [onChange, score],
  )
  /** Insert after the selected note, or at the end. */
  const insert = (note: ScoreNote, after: number | null = selected) => {
    const at = after === null ? notes.length : after + 1
    edit([...notes.slice(0, at), note, ...notes.slice(at)], at)
  }
  const change = (i: number, patch: Partial<ScoreNote>) =>
    edit(notes.map((n, j) => (j === i ? { ...n, ...patch } : n)), i)
  const remove = (i: number) => {
    const next = notes.filter((_, j) => j !== i)
    edit(next, next.length ? Math.min(i, next.length - 1) : null)
  }
  const sel = selected !== null ? notes[selected] : undefined
  const setLength = (base: number, dotted: boolean) => {
    setValue({ base, dotted })
    if (selected !== null) change(selected, { beats: base * (dotted ? 1.5 : 1) })
  }
  const lastPitch = () => {
    for (let i = selected ?? notes.length - 1; i >= 0; i--) if (notes[i].pitch !== null) return notes[i].pitch!
    return 67
  }

  // ---- pointer: click a note to select it (drag to move it), click elsewhere to add one
  const locate = (ev: React.PointerEvent) => {
    const svg = paper.current?.querySelector('svg')
    const lay = layout.current
    if (!svg || !lay) return null
    const r = svg.getBoundingClientRect()
    const x = ev.clientX - r.left
    const y = ev.clientY - r.top
    let near = -1
    lay.xs.forEach((nx, i) => {
      if (Math.abs(nx - x) <= HIT_PX && (near < 0 || Math.abs(nx - x) < Math.abs(lay.xs[near] - x))) near = i
    })
    const before = lay.xs.filter((nx) => nx < x).length // notes to the left of the pointer
    const step = lay.yToStep(y)
    const snapY = lay.lineY(4) - ((step - E4_STEP) * (lay.lineY(1) - lay.lineY(0))) / 2
    const snapX = before === notes.length ? (lay.xs[before - 1] ?? 70) + 52 : (lay.xs[before - 1] + lay.xs[before]) / 2
    return { near, before, step, x: before === 0 && notes.length ? x : snapX, y: snapY, inStaff: y > 0 && y < 190 }
  }
  const onPointerDown = (ev: React.PointerEvent) => {
    const at = locate(ev)
    if (!at) return
    ;(ev.currentTarget as HTMLElement).focus()
    if (at.near >= 0) {
      setSelected(at.near)
      const n = notes[at.near]
      setValue(noteValue(n.beats))
      if (n.pitch !== null) {
        drag.current = at.near
        ;(ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId)
      }
    } else {
      insert({ pitch: stepToMidi(at.step), beats }, at.before === 0 ? -1 : at.before - 1)
    }
    setGhost(null)
  }
  const onPointerMove = (ev: React.PointerEvent) => {
    const at = locate(ev)
    if (!at) return
    if (drag.current !== null) {
      const n = notes[drag.current]
      if (n && n.pitch !== null && staffStep(n) !== at.step) change(drag.current, { pitch: stepToMidi(at.step), flat: false })
    } else {
      setGhost(at.near < 0 && at.inStaff ? { x: at.x, y: at.y } : null)
    }
  }

  // ---- keyboard
  const onKeyDown = (ev: React.KeyboardEvent) => {
    const k = ev.key
    const lower = k.toLowerCase()
    let handled = true
    if (!ev.metaKey && !ev.ctrlKey && 'abcdefg'.includes(lower) && lower.length === 1) {
      insert({ pitch: nearestLetter(lower, lastPitch()), beats })
    } else if (lower === 'r' || k === '0') {
      insert({ pitch: null, beats })
    } else if (NOTE_VALUES.some((v) => v.key === k)) {
      setLength(NOTE_VALUES.find((v) => v.key === k)!.beats, false)
    } else if (k === '.') {
      setLength(value.base, !value.dotted)
    } else if (k === ' ') {
      togglePlay()
    } else if (k === 'ArrowLeft') {
      setSelected(selected === null ? notes.length - 1 : Math.max(0, selected - 1))
    } else if (k === 'ArrowRight') {
      setSelected(selected === null ? null : Math.min(notes.length - 1, selected + 1))
    } else if (k === 'Escape') {
      setSelected(null)
    } else if (selected !== null && sel && (k === 'Backspace' || k === 'Delete')) {
      remove(selected)
    } else if (selected !== null && sel && sel.pitch !== null && (k === 'ArrowUp' || k === 'ArrowDown')) {
      const up = k === 'ArrowUp' ? 1 : -1
      if (ev.metaKey || ev.ctrlKey) change(selected, { pitch: sel.pitch + 12 * up })
      else if (ev.shiftKey) change(selected, { pitch: sel.pitch + up, flat: up < 0 })
      else change(selected, { pitch: stepToMidi(clampStep(staffStep(sel) + up)), flat: false })
    } else if (selected !== null && sel && sel.pitch !== null && (k === '#' || k === '+')) {
      change(selected, { pitch: sel.pitch + 1, flat: false })
    } else if (selected !== null && sel && sel.pitch !== null && k === '-') {
      change(selected, { pitch: sel.pitch - 1, flat: true })
    } else handled = false
    if (handled) ev.preventDefault()
  }

  const togglePlay = () => {
    if (stopPlayback.current) {
      stopPlayback.current()
      return
    }
    if (!notes.length) return
    stopPlayback.current = playScore(score, {
      transpose,
      onNote: (i) => {
        setPlaying(i)
        if (i === null) stopPlayback.current = null
      },
    })
  }

  const bars = barLines(score)
  const sounding = notes.filter((n) => n.pitch !== null).length
  const seconds = (totalBeats(score) * 60) / score.tempo_bpm
  const glyph = (g: string) => <span className="music-glyph">{g}</span>

  return (
    <div className="score-editor">
      <div className="row wrap">
        <div className="row tools" title="Length of the next note (and of the selected one). Keys 1-5, '.' for a dot">
          {NOTE_VALUES.map((v) => (
            <button key={v.vex} title={`${v.name} (${v.key})`} className={value.base === v.beats ? 'sel' : ''} onClick={() => setLength(v.beats, value.dotted)}>
              {glyph(v.glyph)}
            </button>
          ))}
          <button title="Dotted: half as long again (.)" className={value.dotted ? 'sel' : ''} onClick={() => setLength(value.base, !value.dotted)}>
            {glyph(GLYPHS.dot)}
          </button>
        </div>
        <div className="row tools">
          <button title="Add a rest after the selected note (R)" onClick={() => insert({ pitch: null, beats })}>
            {glyph(GLYPHS.rest)}
          </button>
          <button title="Lower the selected note a semitone (-)" disabled={!sel || sel.pitch === null} onClick={() => sel && change(selected!, { pitch: sel.pitch! - 1, flat: true })}>
            {glyph(GLYPHS.flat)}
          </button>
          <button title="Raise the selected note a semitone (+)" disabled={!sel || sel.pitch === null} onClick={() => sel && change(selected!, { pitch: sel.pitch! + 1, flat: false })}>
            {glyph(GLYPHS.sharp)}
          </button>
          <button title="Delete the selected note (Backspace)" disabled={selected === null} onClick={() => selected !== null && remove(selected)}>
            Delete
          </button>
        </div>
        <label title="Quarter notes per minute. The tune is recognised at other tempos too.">
          tempo{' '}
          <input
            type="number"
            min={20}
            max={400}
            style={{ width: 70 }}
            value={score.tempo_bpm}
            onChange={(e) => {
              const bpm = parseFloat(e.target.value)
              if (bpm >= 20 && bpm <= 400) onChange({ ...score, tempo_bpm: bpm })
            }}
          />{' '}
          bpm
        </label>
        <label title="Where the bar lines are drawn. It does not change what is recognised.">
          bar{' '}
          <select value={score.beats_per_bar} onChange={(e) => onChange({ ...score, beats_per_bar: parseInt(e.target.value) })}>
            {[2, 3, 4, 5, 6].map((n) => (
              <option key={n} value={n}>
                {n}/4
              </option>
            ))}
          </select>
        </label>
        <div className="row tools">
          <button className={playing !== null ? 'sel' : ''} disabled={!notes.length} onClick={togglePlay} title="Play at the written tempo (Space)">
            {playing !== null ? '■ Stop' : '▶ Play'}
          </button>
          <select value={transpose} onChange={(e) => setTranspose(parseInt(e.target.value))} title="Octave to play it in">
            <option value={0}>as written</option>
            <option value={12}>1 octave up</option>
            <option value={24}>2 octaves up (whistle)</option>
            <option value={-12}>1 octave down (hum)</option>
          </select>
        </div>
      </div>

      <div className="paper-scroll" ref={scroller}>
        <div
          className="paper"
          ref={paper}
          tabIndex={0}
          role="application"
          aria-label="Sheet music editor"
          onKeyDown={onKeyDown}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => (drag.current = null)}
          onPointerLeave={() => setGhost(null)}
        >
          {ghost && <div className="ghost-note" style={{ left: ghost.x, top: ghost.y }} />}
          {!notes.length && <div className="paper-hint">Click the staff to add the first note</div>}
        </div>
      </div>

      <div className="small muted">
        {sounding} notes · {totalBeats(score)} beats · {seconds.toFixed(1)} s at this tempo
        {sel && ` · selected: ${pitchName(sel)}, ${sel.beats} beat${sel.beats === 1 ? '' : 's'}`}
      </div>
      {bars.crossed.length > 0 && (
        <div className="small warn">
          A note runs over the end of bar {bars.crossed.join(', ')}. The tune still works; only the bar lines are off.
        </div>
      )}
      <div className="small muted">
        Click the staff to add a note where the grey dot shows; click a note to select it and drag it up or down.
        Keys: A–G add a note, R a rest, ↑ ↓ move (Shift = semitone, ⌘/Ctrl = octave), ← → select, 1–5 length, . dot,
        Backspace delete, Space play.
      </div>
    </div>
  )
}
