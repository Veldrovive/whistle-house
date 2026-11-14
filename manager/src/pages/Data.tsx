import { useState } from 'react'
import { api, type Recording } from '../api'
import { AnalysisView } from '../components/AnalysisView'
import { LABELS } from '../lib/labels'
import { fmtDate, fmtS, useAction, useLoad } from '../lib/hooks'

export function Data() {
  const [label, setLabel] = useState('')
  const [selected, setSelected] = useState<Recording | null>(null)
  const [pipeline, setPipeline] = useState<string>('')
  const recs = useLoad(() => api.recordings({ label }), [label])
  const stats = useLoad(() => api.recordingStats(), [])
  const tunes = useLoad(() => api.tunes(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const act = useAction()
  const [upLabel, setUpLabel] = useState('whistle')

  const reload = () => {
    recs.reload()
    stats.reload()
  }
  const update = async (patch: Partial<Recording>) => {
    if (!selected) return
    const r = await act.run(() => api.updateRecording(selected.id, patch))
    if (r) {
      setSelected(r)
      reload()
    }
  }
  const tuneName = (id: string | null) => tunes.data?.find((t) => t.id === id)?.name ?? id

  return (
    <div className="page">
      <div className="row wrap stats">
        {LABELS.map((l) => {
          const s = stats.data?.[l]
          return (
            <button key={l} className={`stat ${label === l ? 'sel' : ''}`} onClick={() => setLabel(label === l ? '' : l)}>
              <b>{l}</b>
              <span>
                {s?.count ?? 0} clips · {fmtS(s?.seconds ?? 0)}
              </span>
            </button>
          )
        })}
      </div>
      <div className="row wrap toolbar">
        <span className="muted">Upload (wav/flac/ogg) as</span>
        <select value={upLabel} onChange={(e) => setUpLabel(e.target.value)}>
          {LABELS.map((l) => (
            <option key={l}>{l}</option>
          ))}
        </select>
        <input
          type="file"
          accept="audio/*"
          multiple
          onChange={async (e) => {
            for (const f of Array.from(e.target.files ?? [])) await act.run(() => api.uploadRecording(f, upLabel))
            e.target.value = ''
            reload()
          }}
        />
        {act.error && <span className="error">{act.error}</span>}
      </div>

      <div className="split">
        <div className="list">
          <table>
            <thead>
              <tr>
                <th>created</th>
                <th>label</th>
                <th>tune</th>
                <th>len</th>
                <th>split</th>
                <th>src</th>
              </tr>
            </thead>
            <tbody>
              {recs.data?.map((r) => (
                <tr key={r.id} className={`clickable ${selected?.id === r.id ? 'sel' : ''}`} onClick={() => setSelected(r)}>
                  <td>{fmtDate(r.created)}</td>
                  <td>
                    <span className={`tag tag-${r.label}`}>{r.label}</span>
                  </td>
                  <td>{tuneName(r.tune_id)}</td>
                  <td>{fmtS(r.duration)}</td>
                  <td>{r.split}</td>
                  <td>{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {recs.data?.length === 0 && <div className="muted">No recordings yet — use the Live page to collect some.</div>}
        </div>

        {selected && (
          <div className="detail card">
            <div className="row wrap">
              <b className="mono">{selected.id}</b>
              <label>
                label{' '}
                <select value={selected.label} onChange={(e) => update({ label: e.target.value })}>
                  {LABELS.map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
              </label>
              <label>
                tune{' '}
                <select value={selected.tune_id ?? ''} onChange={(e) => update({ tune_id: e.target.value || null })}>
                  <option value="">—</option>
                  {tunes.data?.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                split{' '}
                <select value={selected.split ?? ''} onChange={(e) => update({ split: e.target.value as Recording['split'] })}>
                  <option>train</option>
                  <option>val</option>
                  <option>test</option>
                </select>
              </label>
              <button
                className="danger"
                onClick={async () => {
                  if (!confirm('Delete this recording?')) return
                  await act.run(() => api.deleteRecording(selected.id))
                  setSelected(null)
                  reload()
                }}
              >
                Delete
              </button>
            </div>
            <label className="row">
              notes{' '}
              <input
                className="grow"
                defaultValue={selected.notes}
                key={selected.id}
                onBlur={(e) => e.target.value !== selected.notes && update({ notes: e.target.value })}
              />
            </label>
            <label>
              Analyse with{' '}
              <select value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
                <option value="">(active pipeline)</option>
                {pipelines.data?.pipelines.map((p) => (
                  <option key={p.name}>{p.name}</option>
                ))}
              </select>
            </label>
            <AnalysisView recording={selected} pipeline={pipeline || undefined} />
          </div>
        )}
      </div>
    </div>
  )
}
