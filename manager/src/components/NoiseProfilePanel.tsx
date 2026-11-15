import { useState } from 'react'
import { api, type NoiseProfile, type PipelineSpec, type Recording } from '../api'
import { useAction, useLoad } from '../lib/hooks'

// Extra controls shown under a spectral_gate filter in the pipeline editor: pick a noise
// profile, or measure a new one from silence recordings at exactly this filter's position.

type Props = {
  spec: PipelineSpec // the (unsaved) pipeline being edited
  index: number // position of this spectral_gate in spec.filters
  params: Record<string, unknown>
  onParams: (p: Record<string, unknown>) => void
  recordings: Recording[]
}

function upstreamLabel(names: string[], transform: string) {
  return [transform, ...names].join(' → ')
}

function ProfileChart({ p, nStd }: { p: NoiseProfile; nStd: number }) {
  const W = 400
  const H = 110
  const n = p.mean.length
  const thr = p.mean.map((m, i) => m + nStd * p.std[i])
  const lo = Math.min(...p.mean) - 3
  const hi = Math.max(...thr) + 3
  const x = (i: number) => (i / Math.max(1, n - 1)) * W
  const y = (v: number) => H - ((v - lo) / (hi - lo)) * H
  const line = (vals: number[]) => vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const ticks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1]
  return (
    <svg viewBox={`0 0 ${W} ${H + 14}`} style={{ width: '100%', maxWidth: 520 }} role="img">
      <title>Noise mean per bin and gate threshold</title>
      <polyline points={line(thr)} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      <polyline points={line(p.mean)} fill="none" stroke="var(--muted)" strokeWidth={1.5} />
      {ticks.map((i) => (
        <text key={i} x={Math.min(W - 30, x(i))} y={H + 12} fontSize={10} fill="var(--muted)">
          {Math.round(p.freqs[i])} Hz
        </text>
      ))}
      <text x={4} y={10} fontSize={10} fill="var(--accent)">
        gate threshold (mean + {nStd}σ)
      </text>
      <text x={4} y={22} fontSize={10} fill="var(--muted)">
        noise mean · {lo.toFixed(0)}…{hi.toFixed(0)} dB
      </text>
    </svg>
  )
}

export function NoiseProfilePanel({ spec, index, params, onParams, recordings }: Props) {
  const profiles = useLoad(() => api.noiseProfiles(), [])
  const act = useAction()
  const silence = recordings.filter((r) => r.label === 'silence')
  const [picked, setPicked] = useState<string[] | null>(null)
  const chosen = picked ?? silence.map((r) => r.id)
  const [name, setName] = useState('')
  const current = (params.profile as string) || ''
  const prof = profiles.data?.find((p) => p.name === current)
  const nStd = typeof params.n_std === 'number' ? params.n_std : 2

  const here = spec.filters.slice(0, index).map((f) => f.name)
  const measured = prof?.upstream.filters.map((f) => f.name)
  const mismatch =
    prof &&
    (prof.upstream.transform.name !== spec.transform.name ||
      prof.upstream.sample_rate !== spec.sample_rate ||
      JSON.stringify(measured) !== JSON.stringify(here))

  const defaultName = `${spec.name}-silence-${new Date().toISOString().slice(0, 10)}`
  const create = async () => {
    const r = await act.run(() =>
      api.createNoiseProfile({ name: name || defaultName, recording_ids: chosen, spec, position: index }),
    )
    if (r) {
      profiles.reload()
      setName('')
      onParams({ ...params, profile: r.name })
    }
  }

  return (
    <div className="card" style={{ background: 'var(--panel2)' }}>
      <div className="row wrap">
        <b>Noise profile</b>
        <select value={current} onChange={(e) => onParams({ ...params, profile: e.target.value })}>
          <option value="">(none: pass through)</option>
          {profiles.data?.map((p) => (
            <option key={p.name}>{p.name}</option>
          ))}
        </select>
        {prof && (
          <span className="muted small">
            {prof.duration_s.toFixed(1)}s of silence from {prof.sources.length} recording(s), {prof.n_frames} frames
            {prof.n_dropped > 0 && ` (${prof.n_dropped} loud frames ignored)`} · {prof.created.slice(0, 10)}
          </span>
        )}
      </div>
      {mismatch && (
        <div className="error small">
          Measured after {upstreamLabel(measured ?? [], prof.upstream.transform.name)}, but this filter sits after{' '}
          {upstreamLabel(here, spec.transform.name)}. Measure a new profile below (saving will fail otherwise).
        </div>
      )}
      {prof && <ProfileChart p={prof} nStd={nStd} />}

      <div className="muted small">
        Measure a new profile from silence (room tone with nothing happening) recorded on this mic. Record some on the
        Live page with label <code>silence</code>. It is measured at this filter's position in the current, unsaved
        pipeline; save the pipeline afterwards.
      </div>
      {silence.length === 0 ? (
        <div className="muted small">No silence recordings yet.</div>
      ) : (
        <div className="row wrap">
          {silence.map((r) => (
            <label key={r.id} className="small">
              <input
                type="checkbox"
                checked={chosen.includes(r.id)}
                onChange={(e) => setPicked(e.target.checked ? [...chosen, r.id] : chosen.filter((x) => x !== r.id))}
              />{' '}
              {r.id} · {r.duration.toFixed(1)}s
            </label>
          ))}
        </div>
      )}
      <div className="row wrap">
        <input placeholder={defaultName} value={name} onChange={(e) => setName(e.target.value)} style={{ minWidth: 240 }} />
        <button disabled={act.busy || chosen.length === 0} onClick={create}>
          {act.busy ? 'Measuring…' : 'Measure from silence'}
        </button>
        {act.error && <span className="error small">{act.error}</span>}
      </div>
    </div>
  )
}
