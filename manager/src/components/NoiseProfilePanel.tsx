import { useState } from 'react'
import { api, type NoiseProfile, type PipelineSpec, type Recording } from '../api'
import { useAction, useLoad } from '../lib/hooks'
import { profileColor } from '../lib/profileColors'

// Extra controls shown under a gate filter in the pipeline editor.
// spectral_gate (single): pick one noise profile (params.profile).
// adaptive_gate (multi): tick several profiles, one per background the room can be in
// (params.profiles); the filter tracks which one is on.
// Both: measure a new profile from silence recordings at exactly this filter's position.

type Props = {
  spec: PipelineSpec // the (unsaved) pipeline being edited
  index: number // position of this gate in spec.filters
  params: Record<string, unknown>
  onParams: (p: Record<string, unknown>) => void
  recordings: Recording[]
  multi?: boolean // adaptive_gate: several profiles
}

function upstreamLabel(names: string[], transform: string) {
  return [transform, ...names].join(' → ')
}

function hzTicks(freqs: number[], x: (i: number) => number, W: number, H: number) {
  const n = freqs.length
  const ticks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1]
  return ticks.map((i) => (
    <text key={i} x={Math.min(W - 30, x(i))} y={H + 12} fontSize={10} fill="var(--muted)">
      {Math.round(freqs[i])} Hz
    </text>
  ))
}

/** Lines over the bins of one axis (x = bin index), with Hz ticks. */
export function BinChart({
  freqs,
  lines,
  title,
  legend,
}: {
  freqs: number[]
  lines: { values: number[]; color: string; width?: number; opacity?: number; dash?: string }[]
  title: string
  legend?: { text: string; color: string }[]
}) {
  const W = 400
  const H = 110
  const n = freqs.length
  const all = lines.flatMap((l) => l.values)
  if (!n || !all.length) return null
  const lo = Math.min(...all) - 3
  const hi = Math.max(...all) + 3
  const x = (i: number) => (i / Math.max(1, n - 1)) * W
  const y = (v: number) => H - ((v - lo) / (hi - lo)) * H
  const pts = (vals: number[]) => vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H + 14}`} style={{ width: '100%', maxWidth: 520 }} role="img">
      <title>{title}</title>
      {lines.map((l, i) => (
        <polyline
          key={i}
          points={pts(l.values)}
          fill="none"
          stroke={l.color}
          strokeWidth={l.width ?? 1.5}
          strokeOpacity={l.opacity ?? 1}
          strokeDasharray={l.dash}
        />
      ))}
      {hzTicks(freqs, x, W, H)}
      {legend?.map((g, i) => (
        <text key={i} x={4} y={10 + 12 * i} fontSize={10} fill={g.color}>
          {g.text}
        </text>
      ))}
      <text x={W - 4} y={10} fontSize={10} fill="var(--muted)" textAnchor="end">
        {lo.toFixed(0)}…{hi.toFixed(0)} dB
      </text>
    </svg>
  )
}

function ProfileChart({ p, nStd }: { p: NoiseProfile; nStd: number }) {
  const thr = p.mean.map((m, i) => m + nStd * p.std[i])
  return (
    <BinChart
      freqs={p.freqs}
      title="Noise mean per bin and gate threshold"
      lines={[
        { values: thr, color: 'var(--accent)' },
        { values: p.mean, color: 'var(--muted)' },
      ]}
      legend={[
        { text: `gate threshold (mean + ${nStd}σ)`, color: 'var(--accent)' },
        { text: 'noise mean', color: 'var(--muted)' },
      ]}
    />
  )
}

function ProfilesChart({ profs }: { profs: { p: NoiseProfile; color: string }[] }) {
  if (!profs.length) return null
  return (
    <BinChart
      freqs={profs[0].p.freqs}
      title="Noise mean per bin of each ticked profile"
      lines={profs.map(({ p, color }) => ({ values: p.mean, color }))}
      legend={profs.map(({ p, color }) => ({ text: `${p.name} (mean)`, color }))}
    />
  )
}

export function NoiseProfilePanel({ spec, index, params, onParams, recordings, multi }: Props) {
  const profiles = useLoad(() => api.noiseProfiles(), [])
  const act = useAction()
  const silence = recordings.filter((r) => r.label === 'silence')
  const [picked, setPicked] = useState<string[] | null>(null)
  // adaptive_gate measures one background per profile: start with nothing ticked
  const chosen = picked ?? (multi ? [] : silence.map((r) => r.id))
  const [name, setName] = useState('')
  const current = (params.profile as string) || ''
  const ticked = Array.isArray(params.profiles) ? (params.profiles as string[]) : []
  const prof = profiles.data?.find((p) => p.name === current)
  const nStd = typeof params.n_std === 'number' ? params.n_std : 2

  const here = spec.filters.slice(0, index).map((f) => f.name)
  const mismatchOf = (p: NoiseProfile) =>
    p.upstream.transform.name !== spec.transform.name ||
    p.upstream.sample_rate !== spec.sample_rate ||
    JSON.stringify(p.upstream.filters.map((f) => f.name)) !== JSON.stringify(here)
  const mismatch = prof && mismatchOf(prof)

  const setTicked = (names: string[]) => onParams({ ...params, profiles: names })
  const defaultName = multi
    ? `${spec.name}-background-${new Date().toISOString().slice(0, 10)}`
    : `${spec.name}-silence-${new Date().toISOString().slice(0, 10)}`
  const create = async () => {
    const r = await act.run(() =>
      api.createNoiseProfile({ name: name || defaultName, recording_ids: chosen, spec, position: index }),
    )
    if (r) {
      profiles.reload()
      setName('')
      if (multi) {
        setPicked([])
        setTicked(ticked.includes(r.name) ? ticked : [...ticked, r.name])
      } else onParams({ ...params, profile: r.name })
    }
  }

  const tickedProfs = ticked
    .map((n, i) => ({ p: profiles.data?.find((p) => p.name === n), color: profileColor(i) }))
    .filter((x): x is { p: NoiseProfile; color: string } => !!x.p)
  const missing = profiles.data ? ticked.filter((n) => !profiles.data!.some((p) => p.name === n)) : []

  return (
    <div className="card" style={{ background: 'var(--panel2)' }}>
      {multi ? (
        <>
          <div className="row wrap">
            <b>Noise profiles</b>
            <span className="muted small">
              Tick one profile per background the room can be in (quiet, fridge, fan…). The filter estimates which
              one is on, frame by frame, and gates with it. None ticked: pass through.
            </span>
          </div>
          {profiles.data?.length === 0 && <div className="muted small">No noise profiles yet: measure one below.</div>}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {profiles.data?.map((p) => {
              const i = ticked.indexOf(p.name)
              const bad = mismatchOf(p)
              return (
                <label key={p.name} className="small row wrap" style={{ gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={i >= 0}
                    onChange={(e) => setTicked(e.target.checked ? [...ticked, p.name] : ticked.filter((x) => x !== p.name))}
                  />
                  <span style={{ color: i >= 0 ? profileColor(i) : undefined, fontWeight: i >= 0 ? 600 : undefined }}>
                    {p.name}
                  </span>
                  <span className="muted">
                    {p.duration_s.toFixed(1)}s from {p.sources.join(', ')} · {p.created.slice(0, 10)}
                  </span>
                  {bad && (
                    <span className={i >= 0 ? 'error' : 'warn'}>
                      measured after {upstreamLabel(p.upstream.filters.map((f) => f.name), p.upstream.transform.name)},
                      not here
                    </span>
                  )}
                </label>
              )
            })}
          </div>
          {missing.length > 0 && <div className="error small">Missing profiles: {missing.join(', ')}</div>}
          {tickedProfs.some(({ p }) => mismatchOf(p)) && (
            <div className="error small">
              A ticked profile was measured somewhere else than after {upstreamLabel(here, spec.transform.name)}: saving
              will fail. Measure it again below.
            </div>
          )}
          <ProfilesChart profs={tickedProfs} />
        </>
      ) : (
        <>
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
              Measured after{' '}
              {upstreamLabel(
                prof.upstream.filters.map((f) => f.name),
                prof.upstream.transform.name,
              )}
              , but this filter sits after {upstreamLabel(here, spec.transform.name)}. Measure a new profile below
              (saving will fail otherwise).
            </div>
          )}
          {prof && <ProfileChart p={prof} nStd={nStd} />}
        </>
      )}

      <div className="muted small">
        {multi
          ? 'Measure a new profile from silence recordings of one background (tick only takes of the same one: e.g. all "fridge running" clips). '
          : 'Measure a new profile from silence (room tone with nothing happening) recorded on this mic. '}
        Record some on the Live page with label <code>silence</code> (the notes say what was going on). It is
        measured at this filter's position in the current, unsaved pipeline; save the pipeline afterwards.
      </div>
      {silence.length === 0 ? (
        <div className="muted small">No silence recordings yet.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {silence.map((r) => (
            <label key={r.id} className="small">
              <input
                type="checkbox"
                checked={chosen.includes(r.id)}
                onChange={(e) => setPicked(e.target.checked ? [...chosen, r.id] : chosen.filter((x) => x !== r.id))}
              />{' '}
              {r.id} · {r.duration.toFixed(1)}s · {r.created.slice(0, 10)}
              {r.notes ? <span> · {r.notes}</span> : <span className="muted"> · (no notes)</span>}
            </label>
          ))}
        </div>
      )}
      <div className="row wrap">
        <input placeholder={defaultName} value={name} onChange={(e) => setName(e.target.value)} style={{ minWidth: 240 }} />
        <button disabled={act.busy || chosen.length === 0} onClick={create}>
          {act.busy ? 'Measuring…' : multi ? 'Measure and tick' : 'Measure from silence'}
        </button>
        {act.error && <span className="error small">{act.error}</span>}
      </div>
    </div>
  )
}
