import { useEffect, useRef, useState } from 'react'
import {
  api,
  decodeBlock,
  type Axis,
  type EngineStatus,
  type EventJson,
  type KindScores,
  type MatchJson,
  type PipelineSpec,
  type Tune,
} from '../api'
import { RecordPanel } from '../components/RecordPanel'
import { Spectrogram } from '../components/Spectrogram'
import { useAction, useLoad } from '../lib/hooks'

const WINDOW_S = 10

type Buf = { rows: Float32Array[]; times: number[] }

function append(buf: Buf, data: Float32Array, nFrames: number, nBins: number, times: number[]) {
  for (let i = 0; i < nFrames; i++) {
    buf.rows.push(data.subarray(i * nBins, (i + 1) * nBins))
    buf.times.push(times[i])
  }
  const cutoff = buf.times[buf.times.length - 1] - WINDOW_S
  let k = 0
  while (k < buf.times.length && buf.times[k] < cutoff) k++
  if (k) {
    buf.rows.splice(0, k)
    buf.times.splice(0, k)
  }
}

function flatten(buf: Buf, nBins: number): Float32Array {
  const out = new Float32Array(buf.rows.length * nBins)
  buf.rows.forEach((r, i) => out.set(r, i * nBins))
  return out
}

export function Live() {
  const [view, setView] = useState<'filtered' | 'raw'>('filtered')
  const [cfg, setCfg] = useState<{ pipeline: PipelineSpec; axis: Axis; raw_axis: Axis } | null>(null)
  const [, setTick] = useState(0)
  const [connected, setConnected] = useState(false)
  const [matches, setMatches] = useState<MatchJson[]>([])
  const raw = useRef<Buf>({ rows: [], times: [] })
  const filt = useRef<Buf>({ rows: [], times: [] })
  const scores = useRef<number[]>([])
  const kindScores = useRef<KindScores>({})
  const events = useRef<EventJson[]>([])
  const active = useRef<EventJson[]>([])

  const status = useLoad(() => api.status(), [])
  const pipelines = useLoad(() => api.pipelines(), [])
  const devices = useLoad(() => api.devices().catch(() => []), [])
  const tunes = useLoad<Tune[]>(() => api.tunes(), [])
  const act = useAction()

  useEffect(() => {
    api.matches().then(setMatches).catch(() => {})
    let ws: WebSocket | null = null
    let closed = false
    const connect = () => {
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) setTimeout(connect, 1500)
      }
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data)
        if (msg.type === 'config') {
          setCfg(msg.pipeline ? msg : null)
          raw.current = { rows: [], times: [] }
          filt.current = { rows: [], times: [] }
          scores.current = []
          kindScores.current = {}
          events.current = []
        } else if (msg.type === 'frames') {
          append(raw.current, decodeBlock(msg.raw), msg.raw.n_frames, msg.raw.n_bins, msg.raw.times)
          append(filt.current, decodeBlock(msg.filtered), msg.filtered.n_frames, msg.filtered.n_bins, msg.filtered.times)
          scores.current = scores.current.concat(msg.scores).slice(-filt.current.times.length)
          const ks: KindScores = {}
          for (const [kind, s] of Object.entries((msg.kind_scores ?? {}) as KindScores)) {
            ks[kind] = (kindScores.current[kind] ?? []).concat(s).slice(-filt.current.times.length)
          }
          kindScores.current = ks
          const t0 = filt.current.times[0] ?? 0
          events.current = events.current.concat(msg.events).filter((e) => e.end >= t0)
          active.current = msg.active
          setTick((t) => t + 1)
        } else if (msg.type === 'match') {
          setMatches((ms) => [msg.match, ...ms].slice(0, 50))
        }
      }
    }
    connect()
    return () => {
      closed = true
      ws?.close()
    }
  }, [])

  const engine: EngineStatus | undefined = status.data?.engine
  const buf = view === 'raw' ? raw.current : filt.current
  const axis = view === 'raw' ? cfg?.raw_axis : cfg?.axis
  const nBins = buf.rows[0]?.length ?? 0

  const refresh = () => {
    status.reload()
    pipelines.reload()
  }

  return (
    <div className="page">
      <div className="row wrap toolbar">
        <span className={`dot ${engine?.state}`} /> engine <b>{engine?.state ?? '…'}</b>
        {engine?.state === 'running' ? (
          <button onClick={() => act.run(api.liveStop).then(refresh)}>Stop</button>
        ) : (
          <button onClick={() => act.run(api.liveStart).then(refresh)}>Start</button>
        )}
        <label>
          Mic{' '}
          <select
            value={engine?.device ?? ''}
            onChange={(e) => act.run(() => api.setDevice(e.target.value || null)).then(refresh)}
          >
            <option value="">(system default)</option>
            {devices.data?.map((d) => (
              <option key={d.index} value={d.name}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Pipeline{' '}
          <select
            value={pipelines.data?.active ?? ''}
            onChange={(e) => act.run(() => api.activatePipeline(e.target.value)).then(refresh)}
          >
            {pipelines.data?.pipelines.map((p) => (
              <option key={p.name}>{p.name}</option>
            ))}
          </select>
        </label>
        <label>
          View{' '}
          <select value={view} onChange={(e) => setView(e.target.value as 'raw' | 'filtered')}>
            <option value="filtered">filtered + detections</option>
            <option value="raw">raw transform</option>
          </select>
        </label>
        <span className={connected ? 'ok small' : 'warn small'}>{connected ? 'stream connected' : 'stream disconnected'}</span>
      </div>
      {engine?.error && <div className="error">{engine.error}</div>}
      {act.error && <div className="error">{act.error}</div>}

      {axis && nBins > 0 ? (
        <Spectrogram
          values={flatten(buf, nBins)}
          nFrames={buf.rows.length}
          nBins={nBins}
          freqs={axis.freqs}
          times={buf.times}
          events={view === 'filtered' ? events.current : undefined}
          active={view === 'filtered' ? active.current : undefined}
          scores={view === 'filtered' ? scores.current : undefined}
          kindScores={view === 'filtered' ? kindScores.current : undefined}
          height={300}
        />
      ) : (
        <div className="spec empty">waiting for audio…</div>
      )}
      <div className="muted small">
        Lines: finished notes (<span className="kind-whistle-text">whistled</span>,{' '}
        <span className="kind-hum-text">hummed</span>) · dashed: in progress · trace along the bottom: detector
        score (one per kind, in its colour, for detectors that tell them apart).
      </div>

      <h3>Collect data</h3>
      <RecordPanel tunes={tunes.data ?? []} />

      <h3>Recent tune matches</h3>
      {matches.length === 0 && <div className="muted">none yet</div>}
      {matches.map((m, i) => (
        <div key={i} className="match">
          <b>{m.tune_name}</b> · score {m.score.toFixed(2)}
          {m.kind && m.kind !== 'whistle' ? ` · ${m.kind}` : ''}
          {m.wall_time && <span className="muted"> · {new Date(m.wall_time * 1000).toLocaleTimeString()}</span>}
        </div>
      ))}
    </div>
  )
}
