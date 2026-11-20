import { useEffect, useMemo, useRef, useState } from 'react'
import { api, decodeBlock, type Analysis, type HumPitch, type PipelineSpec, type Recording } from '../api'
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
  const [source, setSource] = useState<'original' | 'processed'>('original')
  const [processedUrl, setProcessedUrl] = useState<string | null>(null)
  const render = useAction()
  const [playhead, setPlayhead] = useState<number | null>(null)
  const audio = useRef<HTMLAudioElement>(null)
  const { busy, error, run } = useAction()
  const key = typeof pipeline === 'object' ? JSON.stringify(pipeline) : pipeline
  // hum recordings: overlay the offline pYIN pitch labels (what a hum head would train on)
  const [hum, setHum] = useState<HumPitch | null>(null)
  const humAction = useAction()
  useEffect(() => {
    setHum(null)
    if (recording.label !== 'hum') return
    let stale = false
    humAction.run(() => api.humPitch(recording.id)).then((h) => !stale && h && setHum(h))
    return () => {
      stale = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording.id, recording.label])
  const humTrack = useMemo(
    () => (hum ? hum.f0.map((hz, i) => [i / hum.rate, hz] as [number, number]) : undefined),
    [hum],
  )

  useEffect(() => {
    setRes(null)
    run(() => api.analyze(recording.id, pipeline)).then((r) => r && setRes(r))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording.id, key])

  // processed audio is rendered on demand and invalidated when the recording/pipeline changes
  useEffect(() => {
    setSource('original')
    setProcessedUrl(null)
  }, [recording.id, key])
  useEffect(() => {
    if (source !== 'processed' || processedUrl) return
    let stale = false
    render.run(() => api.renderProcessed(recording.id, pipeline)).then((blob) => {
      if (stale) return
      if (blob) setProcessedUrl(URL.createObjectURL(blob))
      else setSource('original')
    })
    return () => {
      stale = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, processedUrl, recording.id, key])
  // free the blob when it is replaced or the view unmounts
  useEffect(
    () => () => {
      if (processedUrl) URL.revokeObjectURL(processedUrl)
    },
    [processedUrl],
  )
  const switchTo = (s: 'original' | 'processed') => {
    // keep the playback position when flipping between the two
    const t = audio.current?.currentTime ?? 0
    const playing = audio.current ? !audio.current.paused : false
    setSource(s)
    requestAnimationFrame(() => {
      if (!audio.current) return
      const restore = () => {
        audio.current!.currentTime = t
        if (playing) audio.current!.play()
      }
      if (audio.current.readyState >= 1) restore()
      else audio.current.addEventListener('loadedmetadata', restore, { once: true })
    })
  }
  const src = source === 'processed' ? processedUrl : api.audioUrl(recording.id)

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
      <div className="row wrap">
        <audio
          ref={audio}
          controls
          src={src ?? undefined}
          onTimeUpdate={(e) => setPlayhead((e.target as HTMLAudioElement).currentTime)}
        />
        <div
          className="row"
          title="Processed = the recording after the pipeline's transform + filters, turned back into sound: only the transform's frequency range, with whatever the filters removed taken out. Roughly what the detector hears."
        >
          <button className={source === 'original' ? 'sel' : ''} onClick={() => switchTo('original')}>
            Original
          </button>
          <button className={source === 'processed' ? 'sel' : ''} onClick={() => switchTo('processed')}>
            {render.busy ? 'Rendering…' : 'Processed'}
          </button>
        </div>
        {render.error && <span className="error small">{render.error}</span>}
      </div>
      {recording.label === 'hum' && (
        <div className="small" title="Offline pYIN pitch labels (dashed green on the transform spectrogram). Harmonic levels are relative to the strongest harmonic in each frame.">
          <b>Hum pitch (pYIN)</b>{' '}
          {humAction.busy && <span className="muted">labelling…</span>}
          {humAction.error && <span className="error">{humAction.error}</span>}
          {hum && (
            <span className="muted">
              {(hum.voiced_fraction * 100).toFixed(0)}% voiced
              {hum.f0_median !== undefined &&
                ` · ${hum.f0_p5!.toFixed(0)}–${hum.f0_p95!.toFixed(0)} Hz (median ${hum.f0_median.toFixed(0)})`}
              {' · harmonics h1…: '}
              {hum.harmonics_db.map((d) => (d === null ? '·' : d.toFixed(0))).join(' ')} dB
              {hum.f0_p5 !== undefined && res && hum.f0_p5 < res.raw_axis.freqs[0] &&
                ` · fundamental is below this transform's ${res.raw_axis.freqs[0].toFixed(0)} Hz floor`}
            </span>
          )}
        </div>
      )}
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
            pitch={humTrack}
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
            kindScores={res.kind_scores}
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
                  {m.kind && m.kind !== 'whistle' ? ` · ${m.kind}` : ''}
                </div>
              ))}
            </div>
            <div className="grow">
              <h4>Events</h4>
              <table className="small">
                <thead>
                  <tr>
                    <th>kind</th>
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
                        <td className={`kind-${e.kind ?? 'whistle'}-text`}>{e.kind ?? 'whistle'}</td>
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
