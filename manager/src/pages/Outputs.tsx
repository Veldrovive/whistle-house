import { useState } from 'react'
import { api, type OutputInstance } from '../api'
import { SchemaForm } from '../components/SchemaForm'
import { useAction, useLoad } from '../lib/hooks'

function OutputCard({ o, onChange }: { o: OutputInstance; onChange: () => void }) {
  const types = useLoad(() => api.outputTypes(), [])
  const [draft, setDraft] = useState(o)
  const act = useAction()
  const [tested, setTested] = useState(false)
  const t = types.data?.find((x) => x.type === o.type)
  return (
    <div className="card">
      <div className="row wrap">
        <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
        <span className="kind">{o.type}</span>
        <label>
          <input type="checkbox" checked={draft.enabled} onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })} />{' '}
          enabled
        </label>
        <button onClick={() => act.run(() => api.updateOutput(draft)).then(onChange)}>Save</button>
        <button onClick={() => act.run(() => api.testOutput(o.id)).then((r) => setTested(!!r))}>Test fire</button>
        <button className="danger" onClick={() => act.run(() => api.deleteOutput(o.id)).then(onChange)}>
          Delete
        </button>
        {tested && <span className="ok small">fired ✓</span>}
      </div>
      {t && <SchemaForm schema={t.schema} value={draft.params} onChange={(params) => setDraft({ ...draft, params })} />}
      {act.error && <div className="error">{act.error}</div>}
    </div>
  )
}

export function Outputs() {
  const types = useLoad(() => api.outputTypes(), [])
  const outputs = useLoad(() => api.outputs(), [])
  const [type, setType] = useState('home_assistant_event')
  const [name, setName] = useState('')
  const act = useAction()
  return (
    <div className="page">
      <div className="muted">Outputs are what a recognised tune triggers. Bind them to tunes on the Tunes page.</div>
      <div className="row wrap toolbar">
        <input placeholder="name" value={name} onChange={(e) => setName(e.target.value)} />
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {types.data?.map((t) => (
            <option key={t.type} value={t.type}>
              {t.type} — {t.description}
            </option>
          ))}
        </select>
        <button
          disabled={!name}
          onClick={async () => {
            const defaults = types.data?.find((t) => t.type === type)?.defaults ?? {}
            await act.run(() => api.createOutput({ name, type, params: defaults, enabled: true }))
            setName('')
            outputs.reload()
          }}
        >
          Add output
        </button>
        {act.error && <span className="error">{act.error}</span>}
      </div>
      {outputs.data?.map((o) => <OutputCard key={o.id} o={o} onChange={outputs.reload} />)}
    </div>
  )
}
