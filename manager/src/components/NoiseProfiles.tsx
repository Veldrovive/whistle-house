import { useState } from 'react'
import { api, type NoiseProfile, type PipelineSpec, type Recording } from '../api'
import { fmtS, useAction } from '../lib/hooks'
import { profileColor } from '../lib/profileColors'
import { BinChart } from './NoiseProfilePanel'

// Noise profiles on the Data page: a form on a silence recording that turns it into a profile
// (MakeProfileForm), and the list of existing profiles with what uses them (NoiseProfileManager).
// A profile is measured at a filter position of a pipeline (after its transform and the filters
// before its gate), so both ask which pipeline it is for.

const GATES = ['spectral_gate', 'adaptive_gate']

/** Where a pipeline measures profiles: at its first gate filter, else after all filters. */
function gatePosition(spec: PipelineSpec): number {
  const i = spec.filters.findIndex((f) => GATES.includes(f.name))
  return i < 0 ? spec.filters.length : i
}

const adaptiveGate = (spec: PipelineSpec) => {
  const f = spec.filters[gatePosition(spec)]
  return f?.name === 'adaptive_gate' ? f : null
}

function profilesOf(spec: PipelineSpec): string[] {
  return spec.filters.flatMap((f) => {
    if (f.name === 'spectral_gate') return f.params.profile ? [f.params.profile as string] : []
    if (f.name === 'adaptive_gate') return (f.params.profiles as string[] | undefined) ?? []
    return []
  })
}

const chain = (transform: string, filters: string[]) => [transform, ...filters].join(' → ')
const measuredAfter = (p: NoiseProfile) =>
  chain(
    p.upstream.transform.name,
    p.upstream.filters.map((f) => f.name),
  )
const positionOf = (spec: PipelineSpec) =>
  chain(
    spec.transform.name,
    spec.filters.slice(0, gatePosition(spec)).map((f) => f.name),
  )

/** Was the profile measured where this pipeline's gate sits? (The server checks params too.) */
const fits = (p: NoiseProfile, spec: PipelineSpec) =>
  p.upstream.sample_rate === spec.sample_rate && measuredAfter(p) === positionOf(spec)

/** Copy of the pipeline with the profile added to its adaptive_gate's list. */
function withProfile(spec: PipelineSpec, name: string): PipelineSpec {
  const pos = gatePosition(spec)
  return {
    ...spec,
    filters: spec.filters.map((f, i) => {
      if (i !== pos) return f
      const names = (f.params.profiles as string[] | undefined) ?? []
      return { ...f, params: { ...f.params, profiles: names.includes(name) ? names : [...names, name] } }
    }),
  }
}

const slug = (s: string) =>
  s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)

type Shared = {
  profiles: NoiseProfile[]
  pipelines: PipelineSpec[]
  active: string | null
  onChange: () => void // profiles or pipelines changed: reload them
}

/** On a silence recording: measure a noise profile from it. */
export function MakeProfileForm({ recording, profiles, pipelines, active, onChange }: Shared & { recording: Recording }) {
  const act = useAction()
  const gated = pipelines.filter((p) => p.filters.some((f) => GATES.includes(f.name)))
  const fallback =
    (gated.find((p) => p.name === active) ?? gated[0] ?? pipelines.find((p) => p.name === active) ?? pipelines[0])
      ?.name ?? ''
  const [picked, setPicked] = useState('')
  const [name, setName] = useState('')
  const [attach, setAttach] = useState(true)
  const [done, setDone] = useState<string | null>(null)
  const spec = pipelines.find((p) => p.name === (picked || fallback))
  const defaultName = slug(recording.notes) || `silence-${recording.id.slice(0, 6)}`
  const finalName = name || defaultName
  const taken = profiles.some((p) => p.name === finalName)
  const gate = spec ? adaptiveGate(spec) : null
  const fromThis = profiles.filter((p) => p.sources.includes(recording.id))

  const create = async () => {
    if (!spec) return
    setDone(null)
    const r = await act.run(async () => {
      const prof = await api.createNoiseProfile({
        name: finalName,
        recording_ids: [recording.id],
        spec,
        position: gatePosition(spec),
      })
      if (gate && attach) await api.savePipeline(withProfile(spec, prof.name))
      return prof
    })
    onChange()
    if (r) {
      setDone(gate && attach ? `Made "${r.name}" and added it to ${spec.name}.` : `Made "${r.name}".`)
      setName('')
    }
  }

  return (
    <div className="card" style={{ background: 'var(--panel2)' }}>
      <div className="row wrap">
        <b>Make a noise profile from this recording</b>
        {fromThis.length > 0 && (
          <span className="muted small">already made from it: {fromThis.map((p) => p.name).join(', ')}</span>
        )}
      </div>
      <div className="row wrap">
        <label>
          name <input placeholder={defaultName} value={name} onChange={(e) => {
              setName(e.target.value)
              setDone(null)
            }}
            style={{ minWidth: 200 }} />
        </label>
        <label>
          for pipeline{' '}
          <select value={picked || fallback} onChange={(e) => setPicked(e.target.value)}>
            {pipelines.map((p) => (
              <option key={p.name}>{p.name}</option>
            ))}
          </select>
        </label>
        {gate && (
          <label className="small">
            <input type="checkbox" checked={attach} onChange={(e) => setAttach(e.target.checked)} /> add it to that
            pipeline's adaptive_gate
          </label>
        )}
        <button disabled={act.busy || !spec || taken} onClick={create}>
          {act.busy ? 'Measuring…' : 'Make profile'}
        </button>
      </div>
      <div className="muted small">
        {spec && <>Measured after {positionOf(spec)}, so it fits any gate at that position. </>}
        Use a take with one steady background and nothing else going on ({fmtS(recording.duration)} here; 20 s or more
        is good).
      </div>
      {taken && !done && <div className="warn small">A profile named "{finalName}" exists; profiles cannot be overwritten, pick another name.</div>}
      {act.error && <div className="error small">{act.error}</div>}
      {done && <div className="ok small">{done}</div>}
    </div>
  )
}

/** All noise profiles: where each came from, what uses it, compare, add to a pipeline, delete. */
export function NoiseProfileManager({ profiles, pipelines, onChange, silence }: Shared & { silence: Recording[] }) {
  const act = useAction()
  const [shown, setShown] = useState<string[]>([])
  const notes = (id: string) => silence.find((r) => r.id === id)?.notes || id
  const usedBy = (name: string) => pipelines.filter((p) => profilesOf(p).includes(name)).map((p) => p.name)
  // one chart per frequency axis: profiles measured on different transforms cannot share one
  const charts = new Map<string, NoiseProfile[]>()
  for (const p of profiles.filter((p) => shown.includes(p.name))) {
    const key = `${p.freqs.length}:${p.freqs[0]}:${p.freqs[p.freqs.length - 1]}`
    charts.set(key, [...(charts.get(key) ?? []), p])
  }
  const color = (name: string) => profileColor(shown.indexOf(name))

  return (
    <details className="card">
      <summary>
        <b>Noise profiles</b> <span className="muted small">({profiles.length}) · background noise measured from silence recordings, used by the gate filters</span>
      </summary>
      {profiles.length === 0 ? (
        <div className="muted small">
          None yet. Record the room with nothing going on (label <code>silence</code>), select the recording below and
          use "Make a noise profile".
        </div>
      ) : (
        <table>
          <thead>
            <tr>
              <th title="Tick to compare in the chart">show</th>
              <th>name</th>
              <th>from</th>
              <th>len</th>
              <th>measured after</th>
              <th>used by</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {profiles.map((p) => {
              const users = usedBy(p.name)
              const targets = pipelines.filter((s) => adaptiveGate(s) && fits(p, s) && !profilesOf(s).includes(p.name))
              return (
                <tr key={p.name}>
                  <td>
                    <input
                      type="checkbox"
                      checked={shown.includes(p.name)}
                      onChange={(e) => setShown(e.target.checked ? [...shown, p.name] : shown.filter((n) => n !== p.name))}
                    />
                  </td>
                  <td style={{ color: shown.includes(p.name) ? color(p.name) : undefined }}>
                    <b>{p.name}</b>
                  </td>
                  <td className="small">{p.sources.map(notes).join('; ')}</td>
                  <td>{fmtS(p.duration_s)}</td>
                  <td className="small">{measuredAfter(p)}</td>
                  <td className="small">{users.length ? users.join(', ') : <span className="muted">nothing</span>}</td>
                  <td>
                    <div className="row" style={{ justifyContent: 'flex-end' }}>
                      {targets.length > 0 && (
                        <select
                          value=""
                          disabled={act.busy}
                          title="Add this profile to a pipeline's adaptive_gate"
                          onChange={async (e) => {
                            const spec = pipelines.find((s) => s.name === e.target.value)
                            if (!spec) return
                            await act.run(() => api.savePipeline(withProfile(spec, p.name)))
                            onChange()
                          }}
                        >
                          <option value="">add to…</option>
                          {targets.map((s) => (
                            <option key={s.name}>{s.name}</option>
                          ))}
                        </select>
                      )}
                      <button
                        className="danger"
                        disabled={act.busy || users.length > 0}
                        title={users.length ? `Used by ${users.join(', ')}: remove it from them first (Pipelines page)` : 'Delete this profile'}
                        onClick={async () => {
                          if (!confirm(`Delete noise profile "${p.name}"? The recordings it came from are kept.`)) return
                          await act.run(() => api.deleteNoiseProfile(p.name))
                          setShown(shown.filter((n) => n !== p.name))
                          onChange()
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      {act.error && <div className="error small">{act.error}</div>}
      {[...charts.values()].map((group) => (
        <BinChart
          key={group[0].name}
          freqs={group[0].freqs}
          title="Noise mean per bin of the ticked profiles"
          lines={group.map((p) => ({ values: p.mean, color: color(p.name) }))}
          legend={group.map((p) => ({ text: p.name, color: color(p.name) }))}
        />
      ))}
      <div className="muted small">
        Profiles cannot be edited or renamed (models and label caches refer to them by name): make a new one instead.
        To take a profile out of a pipeline, untick it under the gate on the Pipelines page.
      </div>
    </details>
  )
}
