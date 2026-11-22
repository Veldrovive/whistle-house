import { useEffect, useState } from 'react'
import { api, modelReportUrl, type ModelStatus } from '../api'
import { useAction, useLoad } from '../lib/hooks'

const METRICS = ['loss', 'f1', 'precision', 'recall', 'false_positive_rate', 'pitch_acc_1st']
// Shown only when the run has them (training that uses mic whistle clips).
const PROBE_METRICS = [
  'mic_train_f1',
  'mic_val_f1',
  'overfit_gap',
  // per class, for models that detect more than whistles (f1 etc. are then the mean)
  'whistle_f1',
  'hum_f1',
  'whistle_false_positive_rate',
  'hum_false_positive_rate',
  'hum_pitch_acc_1st',
  // share of detected frames whose pitch is the 2nd-4th harmonic of the labelled note
  'overtone_rate',
  'whistle_overtone_rate',
]
const CURVES = [
  { key: 'f1', label: 'val F1 (generalisation; picks the checkpoint)', color: '#4ea1ff' },
  { key: 'whistle_f1', label: 'val F1, whistles', color: '#9fd0ff' },
  { key: 'hum_f1', label: 'val F1, hums', color: '#ffa94d' },
  { key: 'mic_train_f1', label: 'F1 on the mic clips being trained on', color: '#ff9f43' },
  { key: 'mic_val_f1', label: 'F1 on held-out mic clips', color: '#3ddc97' },
]

type History = Record<string, number | null>[]

/** ONNX export of a checkpoint (ml/export.py): the file the learned detector's `onnx` backend
 * runs, about twice as fast on the Pi as PyTorch and usable there without torch. */
function OnnxControls({ m, onChange }: { m: ModelStatus; onChange: () => void }) {
  const act = useAction()
  const o = m.onnx
  const when = o?.exported ? new Date(o.exported * 1000).toLocaleString() : ''
  return (
    <>
      {o?.exists && (
        <span
          className={o.stale ? 'error small' : 'muted small'}
          title={
            o.stale
              ? `${m.name}.onnx (${when}) is older than the checkpoint: export it again`
              : `${m.name}.onnx, exported ${when}. Use it with backend "onnx" on the learned detector`
          }
        >
          {o.stale ? 'ONNX out of date' : 'ONNX ✓'}
        </span>
      )}
      <button
        disabled={act.busy}
        title={`Write ${m.name}.onnx next to the checkpoint and check it against PyTorch. Copy it to the Pi and set the learned detector's backend to "onnx"`}
        onClick={() => act.run(() => api.exportModelOnnx(m.name)).then(onChange)}
      >
        {act.busy ? 'Exporting…' : o?.exists ? 'Re-export ONNX' : 'Export ONNX'}
      </button>
      {act.error && <span className="error small">{act.error}</span>}
    </>
  )
}

/** Layer-by-layer report of a whistle + hum checkpoint on every recording (ml/inspect.py):
 * build it (about a minute for ~40 clips), then open it in a new tab. */
function ReportControls({ m, onChange }: { m: ModelStatus; onChange: () => void }) {
  const act = useAction()
  const r = m.report
  if (r?.running) return <span className="muted small">building report…</span>
  return (
    <>
      {r?.exists && (
        <a
          className="small"
          href={modelReportUrl(m.name)}
          target="_blank"
          rel="noreferrer"
          title={r.built ? `built ${new Date(r.built * 1000).toLocaleString()}` : undefined}
        >
          Report
        </a>
      )}
      <button
        disabled={act.busy}
        title="Run this checkpoint over every recording and show what each layer does (harmonic stack, blocks, heads, decision space)"
        onClick={() => act.run(() => api.buildModelReport(m.name)).then(onChange)}
      >
        {r?.exists ? 'Rebuild report' : 'Build report'}
      </button>
      {act.error && <span className="error small">{act.error}</span>}
    </>
  )
}

/** F1 per validation: val flat or falling while mic-train keeps rising = overfitting. */
function ValidationCurves({ history, bestEpoch }: { history: History; bestEpoch?: number | null }) {
  const curves = CURVES.filter((c) => history.some((h) => typeof h[c.key] === 'number'))
  if (history.length < 2 || curves.length === 0) return null
  const W = 520
  const H = 120
  const P = 6
  const epochs = history.map((h) => h.epoch ?? 0)
  const e0 = Math.min(...epochs)
  const e1 = Math.max(...epochs)
  const vals = history.flatMap((h) => curves.map((c) => h[c.key]).filter((v): v is number => typeof v === 'number'))
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const x = (e: number) => P + ((e - e0) / Math.max(1, e1 - e0)) * (W - 2 * P)
  const y = (v: number) => H - P - ((v - lo) / Math.max(1e-6, hi - lo)) * (H - 2 * P)
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto' }} role="img" aria-label="Validation F1 per epoch">
        {typeof bestEpoch === 'number' && (
          <line x1={x(bestEpoch)} x2={x(bestEpoch)} y1={0} y2={H} stroke="currentColor" strokeOpacity={0.35} strokeDasharray="3 3" />
        )}
        {curves.map((c) => (
          <polyline
            key={c.key}
            fill="none"
            stroke={c.color}
            strokeWidth={1.5}
            points={history
              .filter((h) => typeof h[c.key] === 'number')
              .map((h) => `${x(h.epoch ?? 0).toFixed(1)},${y(h[c.key] as number).toFixed(1)}`)
              .join(' ')}
          />
        ))}
      </svg>
      <div className="muted small">
        {curves.map((c) => (
          <span key={c.key} style={{ marginRight: 12 }}>
            <span style={{ color: c.color }}>━</span> {c.label}
          </span>
        ))}
        <span>
          ┊ best epoch · F1 range {lo.toFixed(3)}–{hi.toFixed(3)}
        </span>
      </div>
    </div>
  )
}

export function Models() {
  const models = useLoad(() => api.models(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const datasets = useLoad(() => api.datasets(), [])
  const [form, setForm] = useState({
    name: 'whistlenet',
    pipeline: '',
    epochs: 20,
    steps_per_epoch: 100,
    batch_size: 16,
    mix: 'mic',
    external: [] as string[],
    init_from: '',
    workers: 2,
    val_examples: 64,
    patience: 0,
    arch: 'small',
    wandb_project: '',
  })
  const prepared = datasets.data?.datasets.filter((d) => d.prepared) ?? []
  const [logFor, setLogFor] = useState<string | null>(null)
  const [log, setLog] = useState('')
  const act = useAction()

  const anyRunning = models.data?.some((m) => m.running || m.report?.running)
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
          The learned detector (<code>learned</code> stage) is trained on recordings with noise/speech mixed in.
          The <code>harmonic</code> model also detects hums (train it on a pipeline with the <code>stft_log</code>{' '}
          transform; <code>hum</code> recordings become positives, labelled by pYIN).
          Training is meant for a workstation/GPU: clone the repo, copy <code>data/</code>, set{' '}
          <code>WHISTLE_HOUSE_EXTERNAL</code> in <code>.env</code>, run{' '}
          <code>uv run whistle-house warmstart</code> (fetch datasets → pretrain → fine-tune on mic data), then copy{' '}
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
          {(['epochs', 'steps_per_epoch', 'batch_size', 'val_examples', 'patience'] as const).map((k) => (
            <label
              key={k}
              title={
                k === 'patience'
                  ? 'Stop after this many validations (epochs) without a better val F1. 0 = never. Use short epochs when fine-tuning on few mic clips.'
                  : k === 'val_examples'
                    ? 'Size of the fixed validation set. Bigger = less noisy.'
                    : undefined
              }
            >
              {k}{' '}
              <input type="number" style={{ width: 70 }} value={form[k]} onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
            </label>
          ))}
          <label title={JSON.stringify(datasets.data?.mixes[form.mix] ?? {})}>
            data mix{' '}
            <select value={form.mix} onChange={(e) => setForm({ ...form, mix: e.target.value })}>
              {Object.keys(datasets.data?.mixes ?? { mic: {} }).map((k) => (
                <option key={k}>{k}</option>
              ))}
            </select>
          </label>
          <label title="small: the ~7k-parameter causal whistle model for the Pi. large: a ~190k-parameter whistle reference model with lookahead, to see how well detection can be done. harmonic: whistles + hums (~31k parameters); needs a pipeline with a log-frequency transform (stft_log) and no noise_floor filter, which swallows held hums. harmonic_deep: the same with twice the blocks (~60k parameters, 1.45 s of context instead of 0.73 s), about twice the cost. Ignored when fine-tuning (the checkpoint's architecture is kept).">
            model{' '}
            <select value={form.arch} onChange={(e) => setForm({ ...form, arch: e.target.value })} disabled={!!form.init_from}>
              <option>small</option>
              <option>large</option>
              <option>harmonic</option>
              <option>harmonic_deep</option>
            </select>
          </label>
          <label title="Also log metrics to this Weights & Biases project (needs `uv sync --extra wandb` and `wandb login` on the machine running the server). Empty = off.">
            wandb project{' '}
            <input
              style={{ width: 110 }}
              value={form.wandb_project}
              placeholder="off"
              onChange={(e) => setForm({ ...form, wandb_project: e.target.value })}
            />
          </label>
          <label>
            fine-tune from{' '}
            <select value={form.init_from} onChange={(e) => setForm({ ...form, init_from: e.target.value })}>
              <option value="">(train from scratch)</option>
              {models.data
                ?.filter((x) => x.has_checkpoint)
                .map((x) => (
                  <option key={x.name}>{x.name}</option>
                ))}
            </select>
          </label>
          {prepared.map((d) => (
            <label key={d.name}>
              <input
                type="checkbox"
                checked={form.external.includes(d.name)}
                onChange={(e) =>
                  setForm({
                    ...form,
                    external: e.target.checked ? [...form.external, d.name] : form.external.filter((x) => x !== d.name),
                  })
                }
              />{' '}
              {d.name}
            </label>
          ))}
          <button
            disabled={act.busy}
            onClick={() =>
              act
                .run(() =>
                  api.train({
                    ...form,
                    pipeline: form.pipeline || undefined,
                    init_from: form.init_from || undefined,
                    wandb_project: form.wandb_project || undefined,
                  }),
                )
                .then(models.reload)
            }
          >
            Train here
          </button>
        </div>
        {act.error && <div className="error">{act.error}</div>}
      </div>

      <div className="card">
        <div className="row wrap">
          <b>External datasets</b>
          <span className="muted small">
            {datasets.data?.root
              ? `WHISTLE_HOUSE_EXTERNAL = ${datasets.data.root}`
              : 'WHISTLE_HOUSE_EXTERNAL not set on this machine (expected on the Pi). Fetch on the training machine: whistle-house datasets fetch all'}
          </span>
        </div>
        <table className="small">
          <thead>
            <tr>
              <th>dataset</th>
              <th>status</th>
              <th>contents</th>
              <th>size</th>
              <th>licence</th>
            </tr>
          </thead>
          <tbody>
            {datasets.data?.datasets.map((d) => (
              <tr key={d.name} title={d.description}>
                <td>
                  <a href={d.homepage} target="_blank" rel="noreferrer">
                    {d.name}
                  </a>
                </td>
                <td>{d.prepared ? 'prepared' : d.downloaded ? 'downloaded' : '—'}</td>
                <td>
                  {d.stats
                    ? Object.entries(d.stats)
                        .map(([k, v]) => `${k}: ${v.clips} / ${v.hours}h`)
                        .join(', ')
                    : d.description}
                </td>
                <td>{d.download_size}</td>
                <td>{d.license}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {models.data?.length === 0 && <div className="muted">No models trained yet.</div>}
      {models.data?.map((m) => (
        <div key={m.name} className="card">
          <div className="row wrap">
            <b>{m.name}</b>
            <span className={`tag state-${m.state}`}>{m.running ? 'running' : m.state}</span>
            <span className="muted small">
              pipeline {m.pipeline} · {m.classes ? `detects ${m.classes.join(' + ')} · ` : ''}
              {m.device ?? ''} · {m.params ? `${m.params} params` : ''} ·{' '}
              {m.has_checkpoint ? `checkpoint ${m.name}.pt` : 'no checkpoint'}
            </span>
            {m.running && <button onClick={() => act.run(() => api.stopTraining(m.name)).then(models.reload)}>Stop</button>}
            {m.can_resume && (
              <button
                title={`Continue from the last finished epoch (${m.history.at(-1)?.epoch ?? 0}) with the same settings`}
                onClick={() => act.run(() => api.resumeTraining(m.name)).then(models.reload)}
              >
                Resume
              </button>
            )}
            {m.resumed?.length ? <span className="muted small">resumed {m.resumed.length}×</span> : null}
            {m.wandb && (
              <a className="small" href={m.wandb.url} target="_blank" rel="noreferrer">
                wandb
              </a>
            )}
            <button
              onClick={() => {
                setLogFor(logFor === m.name ? null : m.name)
                api.modelLog(m.name).then((r) => setLog(r.log))
              }}
            >
              {logFor === m.name ? 'Hide log' : 'Log'}
            </button>
            {m.has_checkpoint && !m.running && <OnnxControls m={m} onChange={models.reload} />}
            {m.has_checkpoint && m.classes?.includes('hum') && <ReportControls m={m} onChange={models.reload} />}
          </div>
          {m.report?.failed && (
            <details className="error small">
              <summary>Report build failed</summary>
              <pre style={{ whiteSpace: 'pre-wrap' }}>{m.report.log}</pre>
            </details>
          )}
          {m.error && <div className="error">{m.error}</div>}
          {m.data && (
            <div className="muted small">
              {m.init_from && <>fine-tuned from {m.init_from} · </>}
              {Object.entries(m.data)
                .map(([split, counts]) => `${split}: ${Object.entries(counts).map(([k, v]) => `${k}=${v}`).join(', ')}`)
                .join(' · ')}
            </div>
          )}
          {m.history.length > 0 &&
            (() => {
              const cols = [...METRICS, ...PROBE_METRICS.filter((k) => m.history.some((h) => typeof h[k] === 'number'))]
              const last = m.history[m.history.length - 1].epoch ?? 0
              const bestEpoch = m.best?.epoch
              // Always show the best row and (for fine-tunes) the epoch-0 baseline, then the latest.
              const rows = m.history.filter(
                (h, i) => i >= m.history.length - 10 || h.epoch === bestEpoch || h.epoch === 0,
              )
              return (
                <>
                  <div className="muted small">
                    {typeof bestEpoch === 'number' && (
                      <>
                        best val F1 {m.best?.f1?.toFixed(3)} at epoch {bestEpoch}
                        {bestEpoch === 0 ? ' (the starting checkpoint: fine-tuning has not improved on it)' : ''} ·{' '}
                        {last - bestEpoch} validation(s) since
                      </>
                    )}
                    {m.stopped_early ? ` · stopped early at epoch ${m.stopped_early} (patience)` : ''}
                  </div>
                  <ValidationCurves history={m.history} bestEpoch={bestEpoch} />
                  <table className="small">
                    <thead>
                      <tr>
                        <th>epoch</th>
                        {cols.map((k) => (
                          <th key={k}>{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((h) => (
                        <tr key={h.epoch} className={bestEpoch === h.epoch ? 'sel' : ''}>
                          <td>{h.epoch === 0 ? '0 (init)' : h.epoch}</td>
                          {cols.map((k) => (
                            <td key={k}>{h[k]?.toFixed(3)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )
            })()}
          {logFor === m.name && <pre className="log">{log}</pre>}
        </div>
      ))}
    </div>
  )
}
