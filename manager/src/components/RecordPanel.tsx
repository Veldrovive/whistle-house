import { useEffect, useState } from 'react'
import { api, type EngineStatus, type Recording, type Tune } from '../api'
import { useAction } from '../lib/hooks'

import { LABELS } from '../lib/labels'

const HINTS: Record<string, string> = {
  whistle: 'Clean whistling, quiet room. These become positives (pitch auto-labelled).',
  hum: 'Clean humming, quiet room. Positives for the whistle + hum model (pitch auto-labelled); a whistle-only model uses them as hard negatives.',
  speech: 'Talk normally, no whistling. The #1 false-trigger in v1 — collect lots.',
  noise: 'TV, kitchen, doors, typing… anything that is not whistling.',
  music: 'Music playing in the room.',
  silence: 'Room tone.',
  other: 'Unsorted; excluded from training.',
}

/** Labels that can be a performance of a tune: it is whistled or hummed. */
const TUNE_LABELS = ['whistle', 'hum']

/** Record clips from the server mic into the dataset. */
export function RecordPanel({
  defaultLabel = 'whistle',
  tuneId = null,
  tunes,
  onSaved,
}: {
  defaultLabel?: string
  tuneId?: string | null
  tunes?: Tune[]
  onSaved?: (r: Recording) => void
}) {
  const [label, setLabel] = useState(defaultLabel)
  const [tune, setTune] = useState<string | null>(tuneId)
  const [status, setStatus] = useState<EngineStatus | null>(null)
  const [last, setLast] = useState<Recording | null>(null)
  const { busy, error, run } = useAction()

  useEffect(() => {
    const tick = () => api.status().then((s) => setStatus(s.engine)).catch(() => {})
    tick()
    const h = setInterval(tick, 500)
    return () => clearInterval(h)
  }, [])

  const saved = (r: Recording | null | undefined) => {
    if (r) {
      setLast(r)
      onSaved?.(r)
    }
  }
  const tuneful = TUNE_LABELS.includes(label)
  const meta = { label, tune_id: tuneful ? tune : null }
  const live = status?.state === 'running'

  return (
    <div className="card record">
      <div className="row wrap">
        <label>
          Label{' '}
          <select value={label} onChange={(e) => setLabel(e.target.value)} disabled={status?.recording}>
            {LABELS.map((l) => (
              <option key={l}>{l}</option>
            ))}
          </select>
        </label>
        {tunes && tuneful && (
          <label>
            Tune{' '}
            <select value={tune ?? ''} onChange={(e) => setTune(e.target.value || null)}>
              <option value="">(none — generic {label})</option>
              {tunes.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {!status?.recording ? (
          <button className="rec" disabled={!live || busy} onClick={() => run(() => api.recordStart(meta))}>
            ● Record
          </button>
        ) : (
          <>
            <button className="rec on" onClick={() => run(api.recordStop).then(saved)}>
              ■ Stop & save ({status.recording_seconds.toFixed(1)}s)
            </button>
            <button onClick={() => run(api.recordCancel)}>Discard</button>
          </>
        )}
        <span className="muted">or save the last</span>
        {[3, 5, 10].map((s) => (
          <button
            key={s}
            disabled={!live || busy}
            onClick={() => run(() => api.capture({ seconds: s, ...meta })).then(saved)}
          >
            {s}s
          </button>
        ))}
      </div>
      <div className="muted small">{HINTS[label]}</div>
      {!live && <div className="warn small">Live engine not running — recording disabled.</div>}
      {error && <div className="error">{error}</div>}
      {last && (
        <div className="ok small">
          Saved {last.label} clip {last.id} ({last.duration.toFixed(1)}s, split {last.split})
        </div>
      )}
    </div>
  )
}
