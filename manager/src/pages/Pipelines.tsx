import { useEffect, useState } from 'react'
import { api, type PipelineSpec, type StageInfo, type StageKind, type StageSpec } from '../api'
import { AnalysisView } from '../components/AnalysisView'
import { NoiseProfilePanel } from '../components/NoiseProfilePanel'
import { SchemaForm } from '../components/SchemaForm'
import { useAction, useLoad } from '../lib/hooks'

const SLOT_HELP: Record<StageKind, string> = {
  transform: 'Audio → time/frequency frames',
  filter: 'Frames → frames (applied in order)',
  detector: 'Frames → whistle events',
  recognizer: 'Whistle events → tune matches',
}

// params edited by the panel under the stage rather than by the generic form
const PANEL_FIELDS: Record<string, string[]> = { adaptive_gate: ['profiles'] }

function StageEditor({
  kind,
  spec,
  options,
  onChange,
  onRemove,
  onMove,
  children,
}: {
  kind: StageKind
  spec: StageSpec
  options: StageInfo[]
  onChange: (s: StageSpec) => void
  onRemove?: () => void
  onMove?: (d: -1 | 1) => void
  children?: React.ReactNode
}) {
  const info = options.find((o) => o.name === spec.name)
  return (
    <div className="stage card">
      <div className="row wrap">
        <span className="kind">{kind}</span>
        <select value={spec.name} onChange={(e) => onChange({ name: e.target.value, params: {} })}>
          {options.map((o) => (
            <option key={o.name}>{o.name}</option>
          ))}
        </select>
        <span className="muted small grow">{info?.description}</span>
        {onMove && (
          <>
            <button onClick={() => onMove(-1)}>↑</button>
            <button onClick={() => onMove(1)}>↓</button>
          </>
        )}
        {onRemove && (
          <button className="danger" onClick={onRemove}>
            ✕
          </button>
        )}
      </div>
      {info ? (
        <SchemaForm
          schema={info.schema}
          value={spec.params}
          onChange={(params) => onChange({ ...spec, params })}
          hide={PANEL_FIELDS[spec.name]}
        />
      ) : (
        <div className="error">unknown stage {spec.name}</div>
      )}
      {children}
    </div>
  )
}

export function Pipelines() {
  const stages = useLoad(() => api.stages(), [])
  const list = useLoad(() => api.pipelines(), [])
  const recs = useLoad(() => api.recordings(), [])
  const [edit, setEdit] = useState<PipelineSpec | null>(null)
  const [testRec, setTestRec] = useState('')
  const [testSpec, setTestSpec] = useState<PipelineSpec | null>(null)
  const act = useAction()

  useEffect(() => {
    if (!edit && list.data?.pipelines.length) {
      setEdit(list.data.pipelines.find((p) => p.name === list.data!.active) ?? list.data.pipelines[0])
    }
  }, [list.data, edit])

  const st = stages.data
  if (!st || !edit) return <div className="page muted">loading…</div>
  const set = (patch: Partial<PipelineSpec>) => setEdit({ ...edit, ...patch })
  const save = async (spec: PipelineSpec) => {
    const r = await act.run(() => api.savePipeline(spec))
    if (r) {
      setEdit(r)
      list.reload()
    }
  }
  const recording = recs.data?.find((r) => r.id === testRec)

  return (
    <div className="page">
      <div className="row wrap toolbar">
        {list.data?.pipelines.map((p) => (
          <button key={p.name} className={edit.name === p.name ? 'sel' : ''} onClick={() => setEdit(p)}>
            {p.name}
            {list.data?.active === p.name && ' ★'}
          </button>
        ))}
      </div>

      <div className="card">
        <div className="row wrap">
          <label>
            name <input value={edit.name} onChange={(e) => set({ name: e.target.value })} />
          </label>
          <label>
            sample rate{' '}
            <input type="number" style={{ width: 90 }} value={edit.sample_rate} onChange={(e) => set({ sample_rate: +e.target.value })} />
          </label>
          <label title="Samples per pipeline step. Larger = less CPU per second for model detectors, more latency.">
            chunk{' '}
            <input type="number" style={{ width: 80 }} value={edit.chunk_size} onChange={(e) => set({ chunk_size: +e.target.value })} />
          </label>
          <input
            className="grow"
            placeholder="description"
            value={edit.description}
            onChange={(e) => set({ description: e.target.value })}
          />
        </div>
        <div className="row wrap">
          <button disabled={act.busy} onClick={() => save(edit)}>
            Save
          </button>
          <button disabled={act.busy} onClick={() => act.run(() => api.activatePipeline(edit.name)).then(list.reload)}>
            Make active (live)
          </button>
          <button
            className="danger"
            onClick={async () => {
              if (!confirm(`Delete pipeline ${edit.name}?`)) return
              await act.run(() => api.deletePipeline(edit.name))
              setEdit(null)
              list.reload()
            }}
          >
            Delete
          </button>
          {act.error && <span className="error">{act.error}</span>}
        </div>
      </div>

      <div className="muted small">{SLOT_HELP.transform}</div>
      <StageEditor kind="transform" spec={edit.transform} options={st.transform} onChange={(transform) => set({ transform })} />
      <div className="muted small">{SLOT_HELP.filter}</div>
      {edit.filters.map((f, i) => (
        <StageEditor
          key={i}
          kind="filter"
          spec={f}
          options={st.filter}
          onChange={(s) => set({ filters: edit.filters.map((x, j) => (j === i ? s : x)) })}
          onRemove={() => set({ filters: edit.filters.filter((_, j) => j !== i) })}
          onMove={(d) => {
            const fs = [...edit.filters]
            const j = i + d
            if (j < 0 || j >= fs.length) return
            ;[fs[i], fs[j]] = [fs[j], fs[i]]
            set({ filters: fs })
          }}
        >
          {(f.name === 'spectral_gate' || f.name === 'adaptive_gate') && (
            <NoiseProfilePanel
              key={f.name}
              spec={edit}
              index={i}
              params={f.params}
              onParams={(params) => set({ filters: edit.filters.map((x, j) => (j === i ? { ...x, params } : x)) })}
              recordings={recs.data ?? []}
              multi={f.name === 'adaptive_gate'}
            />
          )}
        </StageEditor>
      ))}
      <button onClick={() => set({ filters: [...edit.filters, { name: 'identity', params: {} }] })}>+ filter</button>
      <div className="muted small">{SLOT_HELP.detector}</div>
      <StageEditor kind="detector" spec={edit.detector} options={st.detector} onChange={(detector) => set({ detector })} />
      <div className="muted small">{SLOT_HELP.recognizer}</div>
      <StageEditor kind="recognizer" spec={edit.recognizer} options={st.recognizer} onChange={(recognizer) => set({ recognizer })} />

      <h3>Test on a recording</h3>
      <div className="row wrap">
        <select value={testRec} onChange={(e) => setTestRec(e.target.value)}>
          <option value="">choose…</option>
          {recs.data?.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label} · {r.duration.toFixed(1)}s · {r.id}
            </option>
          ))}
        </select>
        <button disabled={!testRec} onClick={() => setTestSpec(structuredClone(edit))}>
          Run current (unsaved) settings
        </button>
      </div>
      {recording && testSpec && <AnalysisView recording={recording} pipeline={testSpec} />}
    </div>
  )
}
