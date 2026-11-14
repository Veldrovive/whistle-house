import { useState } from 'react'
import { api, type Tune } from '../api'
import { RecordPanel } from '../components/RecordPanel'
import { fmtS, useAction, useLoad } from '../lib/hooks'

function TuneDetail({ tune, onChange }: { tune: Tune; onChange: () => void }) {
  const examples = useLoad(() => api.recordings({ tune_id: tune.id }), [tune.id])
  const outputs = useLoad(() => api.outputs(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const [pipeline, setPipeline] = useState('')
  const [gap, setGap] = useState(1.0)
  const [fitInfo, setFitInfo] = useState<Record<string, number[]> | null>(null)
  const act = useAction()

  const save = async (patch: Partial<Tune>) => {
    await act.run(() => api.updateTune({ ...tune, ...patch }))
    onChange()
  }

  return (
    <div className="card detail">
      <div className="row wrap">
        <h3>{tune.name}</h3>
        <label>
          <input type="checkbox" checked={tune.enabled} onChange={(e) => save({ enabled: e.target.checked })} /> enabled
        </label>
        <label>
          threshold override{' '}
          <input
            type="number"
            step="0.05"
            style={{ width: 80 }}
            defaultValue={tune.threshold ?? ''}
            placeholder="default"
            onBlur={(e) => save({ threshold: e.target.value === '' ? null : parseFloat(e.target.value) })}
          />
        </label>
        <button
          className="danger"
          onClick={async () => {
            if (!confirm(`Delete tune ${tune.name}?`)) return
            await act.run(() => api.deleteTune(tune.id))
            onChange()
          }}
        >
          Delete
        </button>
      </div>
      {act.error && <div className="error">{act.error}</div>}

      <h4>1. Record examples</h4>
      <div className="muted small">
        Whistle the tune several times (one per clip, or many per clip separated by ≥{gap}s of silence). Vary key and
        tempo a little the way you naturally would.
      </div>
      <RecordPanel defaultLabel="whistle" tuneId={tune.id} tunes={[tune]} onSaved={examples.reload} />
      <div className="small">
        {examples.data?.length ?? 0} example clips ·{' '}
        {fmtS((examples.data ?? []).reduce((a, r) => a + r.duration, 0))} total
      </div>

      <h4>2. Fit template</h4>
      <div className="row wrap">
        <label>
          pipeline{' '}
          <select value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
            <option value="">(active)</option>
            {pipelines.data?.pipelines.map((p) => (
              <option key={p.name}>{p.name}</option>
            ))}
          </select>
        </label>
        <label>
          repetition gap (s){' '}
          <input type="number" step="0.1" style={{ width: 70 }} value={gap} onChange={(e) => setGap(parseFloat(e.target.value))} />
        </label>
        <button
          disabled={act.busy}
          onClick={async () => {
            const r = await act.run(() => api.fitTune(tune.id, pipeline || undefined, gap))
            if (r) {
              setFitInfo(r.notes_per_instance)
              onChange()
            }
          }}
        >
          Fit
        </button>
      </div>
      {fitInfo && (
        <div className="small muted">
          notes found per repetition:{' '}
          {Object.entries(fitInfo)
            .map(([id, n]) => `${id.slice(0, 6)}: [${n.join(', ')}]`)
            .join(' · ')}
        </div>
      )}
      {tune.template ? (
        <table className="small">
          <thead>
            <tr>
              <th>note</th>
              <th>pitch (st)</th>
              <th>onset (frac)</th>
              <th>log dur</th>
              <th>glide (st)</th>
            </tr>
          </thead>
          <tbody>
            {tune.template.notes.map((n, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                {[n.pitch, n.onset, n.log_duration, n.glide].map((s, k) => (
                  <td key={k}>
                    {s.mean.toFixed(2)} ± {s.std.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={5} className="muted">
                fitted on {tune.template.n_examples} repetitions · span {Math.exp(tune.template.log_span.mean).toFixed(2)}s
              </td>
            </tr>
          </tfoot>
        </table>
      ) : (
        <div className="warn small">Not fitted yet — this tune can't be recognised.</div>
      )}

      <h4>3. Outputs</h4>
      {outputs.data?.length === 0 && <div className="muted small">No outputs defined — add some on the Outputs page.</div>}
      {outputs.data?.map((o) => (
        <label key={o.id} className="row">
          <input
            type="checkbox"
            checked={tune.outputs.includes(o.id)}
            onChange={(e) =>
              save({ outputs: e.target.checked ? [...tune.outputs, o.id] : tune.outputs.filter((x) => x !== o.id) })
            }
          />
          {o.name} <span className="muted small">({o.type})</span>
        </label>
      ))}
    </div>
  )
}

export function Tunes() {
  const tunes = useLoad(() => api.tunes(), [])
  const [sel, setSel] = useState<string | null>(null)
  const [name, setName] = useState('')
  const act = useAction()
  const selected = tunes.data?.find((t) => t.id === sel)

  return (
    <div className="page split">
      <div className="list">
        <div className="row">
          <input placeholder="new tune name" value={name} onChange={(e) => setName(e.target.value)} />
          <button
            disabled={!name}
            onClick={async () => {
              const t = await act.run(() => api.createTune({ name }))
              setName('')
              tunes.reload()
              if (t) setSel(t.id)
            }}
          >
            Add
          </button>
        </div>
        {act.error && <div className="error">{act.error}</div>}
        {tunes.data?.map((t) => (
          <div key={t.id} className={`item clickable ${sel === t.id ? 'sel' : ''}`} onClick={() => setSel(t.id)}>
            <b>{t.name}</b>{' '}
            <span className="muted small">
              {t.template ? `${t.template.notes.length} notes` : 'unfitted'} · {t.outputs.length} outputs
              {!t.enabled && ' · disabled'}
            </span>
          </div>
        ))}
      </div>
      {selected && <TuneDetail key={selected.id} tune={selected} onChange={tunes.reload} />}
    </div>
  )
}
