import { useState } from 'react'
import type { FilterInfo } from '../api'
import { profileColor } from '../lib/profileColors'
import { BinChart } from './NoiseProfilePanel'

// Live view of an adaptive_gate filter (Live page, under the spectrogram): which noise
// profile the room is in over the spectrogram's time window, the current belief, and the
// noise the gate is using right now against the latest frame at the filter's input.
// Collapsed (the default; remembered in localStorage) it is one line: the current profile.

const OPEN_KEY = 'whistle-house.adaptiveGateLive.open'

/** Per-frame series buffered on the Live page, aligned with the spectrogram's frames. */
export type GateBuffer = {
  index: number
  profiles: string[]
  probs: number[][] // frame x profile
  fg: number[] // foreground probability per frame
  mu: number[] | null // effective noise mean, latest frame
  input: number[] | null // latest frame at the filter's input
}

const STRIP_H = 60
const FG_H = 16

export function AdaptiveGateLive({ buf, times, info }: { buf: GateBuffer; times: number[]; info?: FilterInfo }) {
  const [open, setOpen] = useState(() => localStorage.getItem(OPEN_KEY) === '1')
  const toggle = () => {
    localStorage.setItem(OPEN_KEY, open ? '0' : '1')
    setOpen(!open)
  }
  const n = Math.min(buf.probs.length, times.length)
  const probs = buf.probs.slice(-n)
  const fg = buf.fg.slice(-n)
  const ts = times.slice(-n)
  const K = buf.profiles.length
  const t0 = times[0] ?? 0
  const span = Math.max((times[times.length - 1] ?? 1) - t0, 1e-3)
  const x = (t: number) => (((t - t0) / span) * 1000).toFixed(1)

  // stacked areas: profile k fills between cum[k] and cum[k+1] (from the bottom)
  const areas = buf.profiles.map((_, k) => {
    const top: string[] = []
    const bottom: string[] = []
    for (let i = 0; i < n; i++) {
      let below = 0
      for (let j = 0; j < k; j++) below += probs[i][j] ?? 0
      const above = below + (probs[i][k] ?? 0)
      top.push(`${x(ts[i])},${((1 - above) * 100).toFixed(1)}`)
      bottom.push(`${x(ts[i])},${((1 - below) * 100).toFixed(1)}`)
    }
    return top.concat(bottom.reverse()).join(' ')
  })
  const fgArea =
    n > 0
      ? [`${x(ts[0])},100`, ...fg.map((v, i) => `${x(ts[i])},${((1 - v) * 100).toFixed(1)}`), `${x(ts[n - 1])},100`].join(
          ' ',
        )
      : ''

  const cur = probs[n - 1] ?? []
  const curFg = fg[n - 1] ?? 0
  const best = cur.length ? cur.indexOf(Math.max(...cur)) : -1
  const freqs = info?.freqs ?? null
  const lines: { values: number[]; color: string; width?: number; opacity?: number; dash?: string }[] = []
  if (freqs) {
    info?.means?.forEach((m, k) =>
      lines.push({ values: m, color: profileColor(k), width: 1, opacity: 0.15 + 0.6 * (cur[k] ?? 0), dash: '3 2' }),
    )
    if (buf.input) lines.push({ values: buf.input, color: 'var(--muted)', width: 1 })
    if (buf.mu) lines.push({ values: buf.mu, color: 'var(--text)', width: 2 })
  }

  return (
    <div className="card" style={{ background: 'var(--panel2)' }}>
      <div className="row wrap">
        <button className="small" onClick={toggle} aria-expanded={open} title={open ? 'Collapse' : 'Show details'}>
          {open ? '▾' : '▸'} Background
        </button>
        {best >= 0 ? (
          <span className="small">
            <b style={{ color: profileColor(best) }}>{buf.profiles[best]}</b> {Math.round(100 * (cur[best] ?? 0))}%
            {curFg > 0.5 && <span className="warn"> · sound on top</span>}
          </span>
        ) : (
          <span className="muted small">waiting for audio…</span>
        )}
        {open && (
          <span className="muted small">
            adaptive_gate, filter {buf.index + 1}: which noise profile the room is in, per frame; the gate uses the
            mix of profiles by these probabilities
          </span>
        )}
      </div>
      {open && (
        <>
      <svg viewBox="0 0 1000 100" preserveAspectRatio="none" style={{ width: '100%', height: STRIP_H, display: 'block', background: '#000', borderRadius: 4 }}>
        <title>Profile probability over the last seconds</title>
        {areas.map((pts, k) => (
          <polygon key={k} points={pts} fill={profileColor(k)} fillOpacity={0.85} />
        ))}
      </svg>
      <svg viewBox="0 0 1000 100" preserveAspectRatio="none" style={{ width: '100%', height: FG_H, display: 'block', background: '#000', borderRadius: 4 }}>
        <title>Foreground probability (some other sound on top of the background)</title>
        {fgArea && <polygon points={fgArea} fill="var(--warn)" fillOpacity={0.8} />}
      </svg>
      <div className="row wrap" style={{ alignItems: 'flex-start', gap: 24 }}>
        <div style={{ minWidth: 240, flex: '1 1 240px', maxWidth: 420 }}>
          {buf.profiles.map((name, k) => (
            <div key={name} className="row small" style={{ gap: 6 }}>
              <span style={{ width: 110, color: profileColor(k), fontWeight: k === best ? 600 : undefined, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {name}
              </span>
              <span style={{ flex: 1, height: 10, background: 'var(--panel)', borderRadius: 3 }}>
                <span style={{ display: 'block', height: '100%', width: `${100 * (cur[k] ?? 0)}%`, background: profileColor(k), borderRadius: 3 }} />
              </span>
              <span style={{ width: 40, textAlign: 'right' }}>{Math.round(100 * (cur[k] ?? 0))}%</span>
            </div>
          ))}
          <div className="row small" style={{ gap: 6, marginTop: 4 }}>
            <span style={{ width: 110 }} className={curFg > 0.5 ? 'warn' : 'muted'} title="probability that some other sound is on top of the background">
              foreground
            </span>
            <span style={{ flex: 1, height: 10, background: 'var(--panel)', borderRadius: 3 }}>
              <span style={{ display: 'block', height: '100%', width: `${100 * curFg}%`, background: 'var(--warn)', borderRadius: 3 }} />
            </span>
            <span style={{ width: 40, textAlign: 'right' }}>{Math.round(100 * curFg)}%</span>
          </div>
          {K === 0 && <div className="muted small">no profiles</div>}
        </div>
        {freqs && lines.length > 0 && (
          <div style={{ flex: '1 1 320px' }}>
            <BinChart
              freqs={freqs}
              title="Noise the gate uses now against the latest input frame"
              lines={lines}
              legend={[
                { text: 'noise mean in use', color: 'var(--text)' },
                { text: 'latest input frame', color: 'var(--muted)' },
              ]}
            />
          </div>
        )}
      </div>
      <div className="muted small">
        Strip: profile probabilities (colours as in the bars) over the spectrogram's time window; thin strip:
        foreground / unexplained (a sound on top that fits no profile; the belief over profiles holds meanwhile).
        Chart: dashed lines are each profile's mean, brighter when more likely.
      </div>
        </>
      )}
    </div>
  )
}
