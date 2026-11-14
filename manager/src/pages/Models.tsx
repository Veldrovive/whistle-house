import { useEffect, useState } from 'react'
import { api } from '../api'
import { useAction, useLoad } from '../lib/hooks'

const METRICS = ['loss', 'f1', 'precision', 'recall', 'false_positive_rate', 'pitch_acc_1st']

export function Models() {
  const models = useLoad(() => api.models(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const [form, setForm] = useState({ name: 'whistlenet', pipeline: '', epochs: 20, steps_per_epoch: 100, batch_size: 16 })
  const [logFor, setLogFor] = useState<string | null>(null)
  const [log, setLog] = useState('')
  const act = useAction()

  const anyRunning = models.data?.some((m) => m.running)
  const reloadModels = models.reload
  useEffect(() => {
    if (!anyRunning && !logFor) return
    const h = setInterval(() => {
      reloadModels()
      if (logFor) api.modelLog(logFor).then((r) => setLog(r.log))
    }, 3000)
    return () => clearInterval(h)
  }, [anyRunning, logFor, reloadModels])

  return (
    <div className="page">
      <div className="card">
        <p className="muted small">
          The learned detector (<code>learned</code> stage) is trained on your recordings with noise/speech mixed in.
          Training is meant for a workstation/GPU: clone the repo, copy <code>data/</code>, run{' '}
          <code>uv run whistle-house train-detector --name whistlenet</code>, then copy{' '}
          <code>data/models/whistlenet.pt</code> + <code>.json</code> back. Starting it here runs it on this machine.
        </p>
        <div className="row wrap">
          <label>
            name <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </label>
          <label>
            front-end from{' '}
            <select value={form.pipeline} onChange={(e) => setForm({ ...form, pipeline: e.target.value })}>
              <option value="">(active pipeline)</option>
              {pipelines.data?.pipelines.map((p) => (
                <option key={p.name}>{p.name}</option>
              ))}
            </select>
          </label>
          {(['epochs', 'steps_per_epoch', 'batch_size'] as const).map((k) => (
            <label key={k}>
              {k}{' '}
              <input type="number" style={{ width: 70 }} value={form[k]} onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
            </label>
          ))}
          <button
            disabled={act.busy}
            onClick={() => act.run(() => api.train({ ...form, pipeline: form.pipeline || undefined })).then(models.reload)}
          >
            Train here
          </button>
        </div>
        {act.error && <div className="error">{act.error}</div>}
      </div>

      {models.data?.length === 0 && <div className="muted">No models trained yet.</div>}
      {models.data?.map((m) => (
        <div key={m.name} className="card">
          <div className="row wrap">
            <b>{m.name}</b>
            <span className={`tag state-${m.state}`}>{m.running ? 'running' : m.state}</span>
            <span className="muted small">
              pipeline {m.pipeline} · {m.device ?? ''} · {m.params ? `${m.params} params` : ''} ·{' '}
              {m.has_checkpoint ? `checkpoint ${m.name}.pt` : 'no checkpoint'}
            </span>
            {m.running && <button onClick={() => act.run(() => api.stopTraining(m.name)).then(models.reload)}>Stop</button>}
            <button
              onClick={() => {
                setLogFor(logFor === m.name ? null : m.name)
                api.modelLog(m.name).then((r) => setLog(r.log))
              }}
            >
              {logFor === m.name ? 'Hide log' : 'Log'}
            </button>
          </div>
          {m.error && <div className="error">{m.error}</div>}
          {m.data && (
            <div className="muted small">
              data: {Object.entries(m.data).map(([k, v]) => `${k}=${v}`).join(' · ')}
            </div>
          )}
          {m.history.length > 0 && (
            <table className="small">
              <thead>
                <tr>
                  <th>epoch</th>
                  {METRICS.map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {m.history.slice(-10).map((h) => (
                  <tr key={h.epoch} className={m.best?.epoch === h.epoch ? 'sel' : ''}>
                    <td>{h.epoch}</td>
                    {METRICS.map((k) => (
                      <td key={k}>{h[k]?.toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {logFor === m.name && <pre className="log">{log}</pre>}
        </div>
      ))}
    </div>
  )
}
