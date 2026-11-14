import { useEffect, useMemo, useRef } from 'react'
import type { EventJson } from '../api'

// Inferno-ish colormap built from a few stops.
const STOPS: [number, [number, number, number]][] = [
  [0, [0, 0, 4]],
  [0.25, [87, 16, 110]],
  [0.5, [188, 55, 84]],
  [0.75, [249, 142, 9]],
  [1, [252, 255, 164]],
]
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    let k = 0
    while (k < STOPS.length - 2 && x > STOPS[k + 1][0]) k++
    const [x0, c0] = STOPS[k]
    const [x1, c1] = STOPS[k + 1]
    const f = (x - x0) / (x1 - x0)
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + f * (c1[j] - c0[j])
  }
  return lut
})()

function percentile(v: Float32Array, p: number): number {
  if (!v.length) return 0
  const step = Math.max(1, Math.floor(v.length / 5000))
  const s: number[] = []
  for (let i = 0; i < v.length; i += step) s.push(v[i])
  s.sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor(p * s.length))]
}

/** Hz -> fractional bin index by interpolating the axis centre frequencies. */
function hzToBin(hz: number, freqs: number[]): number {
  if (hz <= freqs[0]) return 0
  for (let i = 1; i < freqs.length; i++) {
    if (hz <= freqs[i]) return i - 1 + (hz - freqs[i - 1]) / (freqs[i] - freqs[i - 1])
  }
  return freqs.length - 1
}

export type SpectrogramProps = {
  values: Float32Array // row-major (nFrames x nBins)
  nFrames: number
  nBins: number
  freqs: number[]
  times: number[] // frame centre times
  events?: EventJson[]
  active?: EventJson[]
  scores?: number[]
  segments?: { start: number; end: number; label: string }[]
  playhead?: number | null
  vmin?: number
  vmax?: number
  height?: number
  onSeek?: (t: number) => void
}

export function Spectrogram(p: SpectrogramProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [lo, hi] = useMemo(() => {
    if (p.vmin !== undefined && p.vmax !== undefined) return [p.vmin, p.vmax]
    return [percentile(p.values, 0.02), percentile(p.values, 0.995)]
  }, [p.values, p.vmin, p.vmax])

  useEffect(() => {
    const c = canvas.current
    if (!c || !p.nFrames) return
    c.width = p.nFrames
    c.height = p.nBins
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(p.nFrames, p.nBins)
    const range = hi - lo || 1
    for (let t = 0; t < p.nFrames; t++) {
      for (let b = 0; b < p.nBins; b++) {
        const v = (p.values[t * p.nBins + b] - lo) / range
        const k = Math.max(0, Math.min(255, Math.round(v * 255))) * 3
        const o = ((p.nBins - 1 - b) * p.nFrames + t) * 4
        img.data[o] = LUT[k]
        img.data[o + 1] = LUT[k + 1]
        img.data[o + 2] = LUT[k + 2]
        img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [p.values, p.nFrames, p.nBins, lo, hi])

  const t0 = p.times[0] ?? 0
  const t1 = p.times[p.times.length - 1] ?? 1
  const span = Math.max(t1 - t0, 1e-3)
  const h = p.height ?? 220
  const x = (t: number) => ((t - t0) / span) * 1000
  const y = (hz: number) => (1 - hzToBin(hz, p.freqs) / (p.nBins - 1)) * 1000
  const ticks = [500, 1000, 2000, 3000, 4000, 6000].filter(
    (f) => f >= p.freqs[0] && f <= p.freqs[p.freqs.length - 1],
  )

  const line = (e: EventJson) => e.contour.map(([t, hz]) => `${x(t)},${y(hz)}`).join(' ')

  return (
    <div
      className="spec"
      style={{ height: h }}
      onClick={(ev) => {
        if (!p.onSeek) return
        const r = (ev.currentTarget as HTMLDivElement).getBoundingClientRect()
        p.onSeek(t0 + ((ev.clientX - r.left) / r.width) * span)
      }}
    >
      <canvas ref={canvas} />
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none">
        {p.segments?.map((s, i) => (
          <rect
            key={i}
            x={x(s.start)}
            width={x(s.end) - x(s.start)}
            y={0}
            height={1000}
            className={`seg seg-${s.label}`}
          />
        ))}
        {p.events?.map((e, i) => (
          <polyline key={i} points={line(e)} className="ev" vectorEffect="non-scaling-stroke" />
        ))}
        {p.active?.map((e, i) => (
          <polyline key={`a${i}`} points={line(e)} className="ev active" vectorEffect="non-scaling-stroke" />
        ))}
        {p.scores && p.scores.length === p.times.length && (
          <polyline
            className="score"
            vectorEffect="non-scaling-stroke"
            points={p.scores.map((s, i) => `${x(p.times[i])},${1000 - s * 180}`).join(' ')}
          />
        )}
        {p.playhead != null && (
          <line x1={x(p.playhead)} x2={x(p.playhead)} y1={0} y2={1000} className="playhead" vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      <div className="ticks">
        {ticks.map((f) => (
          <span key={f} style={{ top: `${y(f) / 10}%` }}>
            {f >= 1000 ? `${f / 1000}k` : f}
          </span>
        ))}
      </div>
    </div>
  )
}
