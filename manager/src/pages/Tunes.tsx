import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { api, type Recording, type Score, type Tune } from '../api'
import { AnalysisView } from '../components/AnalysisView'
import { RecordPanel } from '../components/RecordPanel'
import { fmtDate, fmtS, useAction, useLoad } from '../lib/hooks'
import { EMPTY_SCORE } from '../lib/score'

// the notation library is large: load it only when a tune is opened
const ScoreEditor = lazy(() => import('../components/ScoreEditor').then((m) => ({ default: m.ScoreEditor })))

function TuneDetail({ tune, onChange }: { tune: Tune; onChange: () => void }) {
  const examples = useLoad(() => api.recordings({ tune_id: tune.id }), [tune.id])
  const outputs = useLoad(() => api.outputs(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const [pipeline, setPipeline] = useState('')
  const [gap, setGap] = useState(1.0)
  const [fitInfo, setFitInfo] = useState<Record<string, number[]> | null>(null)
  const [tryClip, setTryClip] = useState<Recording | null>(null)
  const act = useAction()

  // The score is edited here and saved shortly after the last change; every other save
  // sends the current score along, so a stale copy never overwrites it.
  const [score, setScore] = useState<Score>(tune.score ?? EMPTY_SCORE)
  const latest = useRef({ tune, score, dirty: false })
  latest.current.tune = tune
  const [saved, setSaved] = useState(true)
  const save = async (patch: Partial<Tune> = {}) => {
    const { tune: t, score: sc, dirty } = latest.current
    latest.current.dirty = false
    await act.run(() => api.updateTune({ ...t, score: dirty || t.score ? sc : null, ...patch }))
    setSaved(!latest.current.dirty)
    onChange()
  }
  const editScore = (sc: Score) => {
    latest.current.score = sc
    latest.current.dirty = true
    setScore(sc)
    setSaved(false)
  }
  useEffect(() => {
    if (saved) return
    const timer = setTimeout(() => save(), 600)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [score, saved])
  // leaving the tune with an unsaved edit: save it now
  useEffect(
    () => () => {
      const { tune: t, score: sc, dirty } = latest.current
      if (dirty) api.updateTune({ ...t, score: sc }).then(onChange)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const active = pipelines.data?.pipelines.find((p) => p.name === pipelines.data?.active)
  const sounding = score.notes.filter((n) => n.pitch !== null).length

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

      <h4>
        1. Sheet music{' '}
        <span className="muted small">{act.busy ? 'saving…' : saved ? (tune.score ? 'saved' : '') : 'edited'}</span>
      </h4>
      <Suspense fallback={<div className="muted">loading the editor…</div>}>
        <ScoreEditor score={score} onChange={editScore} />
      </Suspense>
      <div className="muted small">
        Only the shape of the tune is matched: it is recognised in any key or octave, whistled or hummed, from half
        to twice this tempo, and with notes slid together. By default it must stand alone, with a moment of quiet
        before and after (the <code>score_align</code> settings on the Pipelines page).
      </div>
      {sounding > 0 && sounding < 4 && (
        <div className="warn small">
          Short tunes are easy to whistle by accident. Five or more notes with a leap or two are much safer.
        </div>
      )}
      {active && active.recognizer.name !== 'score_align' && (
        <div className="warn small row wrap">
          The active pipeline ({active.name}) uses the {active.recognizer.name} recognizer, which ignores written
          tunes.
          <button
            disabled={act.busy}
            onClick={async () => {
              await act.run(() => api.savePipeline({ ...active, recognizer: { name: 'score_align', params: {} } }))
              pipelines.reload()
            }}
          >
            Switch {active.name} to score_align
          </button>
        </div>
      )}

      <h4>2. Try it</h4>
      <div className="muted small">
        Record yourself whistling or humming the tune, then click the clip to see what the active pipeline heard and
        where it found each note.
      </div>
      <RecordPanel defaultLabel="whistle" tuneId={tune.id} tunes={[tune]} onSaved={examples.reload} />
      {(examples.data ?? []).map((r) => (
        <div key={r.id} className={`item clickable small ${tryClip?.id === r.id ? 'sel' : ''}`} onClick={() => setTryClip(tryClip?.id === r.id ? null : r)}>
          <span className={`tag tag-${r.label}`}>{r.label}</span> {fmtDate(r.created)} · {fmtS(r.duration)}
        </div>
      ))}
      {tryClip && <AnalysisView recording={tryClip} />}

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

      <details>
        <summary className="muted">Or learn the tune from recorded examples (note_template recognizer)</summary>
        <div className="muted small">
          Record the tune several times under "Try it" (one per clip, or many per clip separated by ≥{gap}s of
          silence), then fit. {examples.data?.length ?? 0} clips ·{' '}
          {fmtS((examples.data ?? []).reduce((a, r) => a + r.duration, 0))} total.
        </div>
        <h4>Fit a template to the clips above</h4>
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
          <div className="muted small">No template fitted.</div>
        )}
      </details>
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
              {t.score?.notes.length
                ? `${t.score.notes.filter((n) => n.pitch !== null).length} notes`
                : t.template
                  ? `${t.template.notes.length} notes (fitted)`
                  : 'empty'}{' '}
              · {t.outputs.length} outputs
              {!t.enabled && ' · disabled'}
            </span>
          </div>
        ))}
      </div>
      {selected && <TuneDetail key={selected.id} tune={selected} onChange={tunes.reload} />}
    </div>
  )
}
