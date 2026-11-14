import { useEffect, useMemo, useRef, useState } from 'react'
import { api, decodeBlock, type Analysis, type PipelineSpec, type Recording } from '../api'
import { useAction } from '../lib/hooks'
import { Spectrogram } from './Spectrogram'

/** Run a pipeline over a recording and show every stage's output. */
export function AnalysisView({
  recording,
  pipeline,
}: {
  recording: Recording
  pipeline?: string | PipelineSpec
}) {
  const [res, setRes] = useState<Analysis | null>(null)
  const [playhead, setPlayhead] = useState<number | null>(null)
  const audio = useRef<HTMLAudioElement>(null)
  const { busy, error, run } = useAction()
  const key = typeof pipeline === 'object' ? JSON.stringify(pipeline) : pipeline

  useEffect(() => {
    setRes(null)
    run(() => api.analyze(recording.id, pipeline)).then((r) => r && setRes(r))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording.id, key])

  const raw = useMemo(() => (res ? decodeBlock(res.raw) : null), [res])
  const filt = useMemo(() => (res ? decodeBlock(res.filtered) : null), [res])
  const seek = (t: number) => {
    if (audio.current) {
      audio.current.currentTime = Math.max(0, t)
      audio.current.play()
    }
  }

  return (
    <div className="analysis">
      <audio
        ref={audio}
        controls
        src={api.audioUrl(recording.id)}
        onTimeUpdate={(e) => setPlayhead((e.target as HTMLAudioElement).currentTime)}
      />
      {busy && <div className="muted">analysing…</div>}
      {error && <div className="error">{error}</div>}
      {res && raw && filt && (
        <>
          <h4>
            Transform <span className="muted small">{res.pipeline.transform.name}</span>
          </h4>
          <Spectrogram
            values={raw}
            nFrames={res.raw.n_frames}
            nBins={res.raw.n_bins}
            freqs={res.raw_axis.freqs}
            times={res.raw.times}
            segments={recording.segments}
            playhead={playhead}
            onSeek={seek}
          />
          <h4>
            Filtered + detections{' '}
            <span className="muted small">
              {res.pipeline.filters.map((f) => f.name).join(' → ') || 'no filters'} ·{' '}
              {res.pipeline.detector.name} · {res.events.length} events
            </span>
          </h4>
          <Spectrogram
            values={filt}
            nFrames={res.filtered.n_frames}
            nBins={res.filtered.n_bins}
            freqs={res.axis.freqs}
            times={res.filtered.times}
            events={res.events}
            scores={res.scores}
            playhead={playhead}
            onSeek={seek}
          />
          <div className="row wrap">
            <div>
              <h4>Tune matches ({res.pipeline.recognizer.name})</h4>
              {res.matches.length === 0 && <div className="muted">none</div>}
              {res.matches.map((m, i) => (
                <div key={i} className="match">
                  <b>{m.tune_name}</b> at {m.time.toFixed(2)}s · score {m.score.toFixed(2)}
                </div>
              ))}
            </div>
            <div className="grow">
              <h4>Events</h4>
              <table className="small">
                <thead>
                  <tr>
                    <th>start</th>
                    <th>dur</th>
                    <th>median Hz</th>
                    <th>conf</th>
                  </tr>
                </thead>
                <tbody>
                  {res.events.map((e, i) => {
                    const hz = e.contour.map((c) => c[1]).sort((a, b) => a - b)
                    return (
                      <tr key={i} onClick={() => seek(e.start - 0.2)} className="clickable">
                        <td>{e.start.toFixed(2)}</td>
                        <td>{(e.end - e.start).toFixed(2)}</td>
                        <td>{hz[Math.floor(hz.length / 2)].toFixed(0)}</td>
                        <td>{e.confidence.toFixed(2)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
