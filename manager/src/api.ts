// Typed client for the Python backend (src/whistle_house/server/app.py).
// Keep in sync with the server: every route used by the manager is wrapped here.

export type JsonSchema = {
  type?: string
  title?: string
  description?: string
  default?: unknown
  enum?: unknown[]
  anyOf?: JsonSchema[]
  items?: JsonSchema
  prefixItems?: JsonSchema[]
  properties?: Record<string, JsonSchema>
  additionalProperties?: JsonSchema | boolean
  $ref?: string
  $defs?: Record<string, JsonSchema>
}

export type StageKind = 'transform' | 'filter' | 'detector' | 'recognizer'
export type StageInfo = {
  name: string
  description: string
  schema: JsonSchema
  defaults: Record<string, unknown>
}
export type StageSpec = { name: string; params: Record<string, unknown> }
export type PipelineSpec = {
  name: string
  description: string
  sample_rate: number
  chunk_size: number
  transform: StageSpec
  filters: StageSpec[]
  detector: StageSpec
  recognizer: StageSpec
}

export type Segment = { start: number; end: number; label: string }
export type Recording = {
  id: string
  created: string
  label: string
  tune_id: string | null
  sample_rate: number
  duration: number
  source: string
  device: string | null
  notes: string
  tags: string[]
  segments: Segment[]
  split: 'train' | 'val' | 'test' | null
}

export type FeatureStat = { mean: number; std: number }
export type NoteTemplate = {
  pitch: FeatureStat
  onset: FeatureStat
  log_duration: FeatureStat
  glide: FeatureStat
}
export type Tune = {
  id: string
  name: string
  description: string
  enabled: boolean
  examples: string[]
  template: { notes: NoteTemplate[]; log_span: FeatureStat; n_examples: number } | null
  outputs: string[]
  threshold: number | null
}

export type OutputType = {
  type: string
  description: string
  schema: JsonSchema
  defaults: Record<string, unknown>
}
export type OutputInstance = {
  id: string
  name: string
  type: string
  params: Record<string, unknown>
  enabled: boolean
}

export type Axis = { freqs: number[]; scale: string }
export type EncodedBlock = { values: string; n_frames: number; n_bins: number; times: number[] }
export type EventJson = {
  start: number
  end: number
  confidence: number
  contour: [number, number, number][]
  meta: Record<string, unknown>
}
export type MatchJson = {
  tune_id: string
  tune_name: string
  time: number
  score: number
  events: EventJson[]
  wall_time?: number
}
export type Analysis = {
  pipeline: PipelineSpec
  raw_axis: Axis
  axis: Axis
  raw: EncodedBlock
  filtered: EncodedBlock
  scores: number[]
  events: EventJson[]
  matches: MatchJson[]
}

export type EngineStatus = {
  state: 'stopped' | 'running' | 'error'
  error: string | null
  device: string | null
  pipeline: string | null
  sample_rate: number | null
  recording: boolean
  recording_seconds: number
  buffered_seconds: number
}

export type ModelStatus = {
  name: string
  state: 'preparing' | 'training' | 'done' | 'failed'
  pipeline: string
  config: Record<string, number | string>
  history: Record<string, number>[]
  best?: Record<string, number>
  data?: Record<string, Record<string, number>> // {train: {...counts}, val: {...counts}}
  init_from?: string
  error?: string
  device?: string
  params?: number
  running: boolean
  has_checkpoint: boolean
}

export type DatasetStatus = {
  name: string
  description: string
  homepage: string
  license: string
  download_size: string
  downloaded: boolean
  prepared: boolean
  stats: Record<string, { clips: number; hours: number }> | null
}
export type DatasetsInfo = {
  root: string | null // WHISTLE_HOUSE_EXTERNAL; usually null on the Pi
  datasets: DatasetStatus[]
  mixes: Record<string, Record<string, number>>
}

/** Per-bin noise statistics measured from silence recordings (spectral_gate filter). */
export type NoiseProfile = {
  name: string
  freqs: number[]
  mean: number[]
  std: number[]
  upstream: { sample_rate: number; transform: StageSpec; filters: StageSpec[] }
  sources: string[]
  n_frames: number
  n_dropped: number
  duration_s: number
  created: string
}

export class ApiError extends Error {}

async function req<T>(method: string, path: string, body?: unknown, as: 'json' | 'blob' = 'json'): Promise<T> {
  const init: RequestInit = { method }
  if (body instanceof FormData) init.body = body
  else if (body !== undefined) {
    init.body = JSON.stringify(body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  const r = await fetch(path, init)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* not json */
    }
    throw new ApiError(msg)
  }
  return (as === 'blob' ? r.blob() : r.json()) as Promise<T>
}

export const api = {
  status: () =>
    req<{ engine: EngineStatus; labels: string[]; data_dir: string }>('GET', '/api/status'),
  devices: () =>
    req<{ index: number; name: string; default_samplerate: number }[]>('GET', '/api/devices'),
  setDevice: (device: string | null) => req<EngineStatus>('POST', '/api/live/device', { device }),
  liveStart: () => req<EngineStatus>('POST', '/api/live/start'),
  liveStop: () => req<EngineStatus>('POST', '/api/live/stop'),

  stages: () => req<Record<StageKind, StageInfo[]>>('GET', '/api/stages'),
  pipelines: () => req<{ active: string | null; pipelines: PipelineSpec[] }>('GET', '/api/pipelines'),
  savePipeline: (spec: PipelineSpec) =>
    req<PipelineSpec>('PUT', `/api/pipelines/${encodeURIComponent(spec.name)}`, spec),
  deletePipeline: (name: string) => req('DELETE', `/api/pipelines/${encodeURIComponent(name)}`),
  activatePipeline: (name: string) =>
    req<EngineStatus>('POST', `/api/pipelines/${encodeURIComponent(name)}/activate`),

  recordings: (q: { label?: string; tune_id?: string } = {}) => {
    const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v) as [string, string][])
    return req<Recording[]>('GET', `/api/recordings?${p}`)
  },
  recordingStats: () =>
    req<Record<string, { count: number; seconds: number }>>('GET', '/api/recordings/stats'),
  updateRecording: (id: string, upd: Partial<Recording>) =>
    req<Recording>('PATCH', `/api/recordings/${id}`, upd),
  deleteRecording: (id: string) => req('DELETE', `/api/recordings/${id}`),
  uploadRecording: (file: File, label: string, tune_id?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('label', label)
    if (tune_id) fd.append('tune_id', tune_id)
    return req<Recording>('POST', '/api/recordings/upload', fd)
  },
  audioUrl: (id: string) => `/api/recordings/${id}/audio`,

  recordStart: (meta: { label: string; tune_id?: string | null; notes?: string }) =>
    req<EngineStatus>('POST', '/api/record/start', meta),
  recordStop: () => req<Recording | null>('POST', '/api/record/stop'),
  recordCancel: () => req<EngineStatus>('POST', '/api/record/cancel'),
  capture: (meta: { seconds: number; label: string; tune_id?: string | null }) =>
    req<Recording>('POST', '/api/record/capture', meta),

  /** The recording after the pipeline's transform + filters, resynthesised (WAV blob). */
  renderProcessed: (recording_id: string, pipeline?: string | PipelineSpec) =>
    req<Blob>(
      'POST',
      '/api/render',
      typeof pipeline === 'object' ? { recording_id, spec: pipeline } : { recording_id, pipeline },
      'blob',
    ),
  analyze: (recording_id: string, pipeline?: string | PipelineSpec) =>
    req<Analysis>(
      'POST',
      '/api/analyze',
      typeof pipeline === 'object' ? { recording_id, spec: pipeline } : { recording_id, pipeline },
    ),

  tunes: () => req<Tune[]>('GET', '/api/tunes'),
  createTune: (t: Partial<Tune>) => req<Tune>('POST', '/api/tunes', t),
  updateTune: (t: Tune) => req<Tune>('PUT', `/api/tunes/${t.id}`, t),
  deleteTune: (id: string) => req('DELETE', `/api/tunes/${id}`),
  fitTune: (id: string, pipeline: string | undefined, split_gap_s: number) =>
    req<{ tune: Tune; notes_per_instance: Record<string, number[]> }>(
      'POST',
      `/api/tunes/${id}/fit`,
      { pipeline, split_gap_s },
    ),

  outputTypes: () => req<OutputType[]>('GET', '/api/output-types'),
  outputs: () => req<OutputInstance[]>('GET', '/api/outputs'),
  createOutput: (o: Omit<OutputInstance, 'id'>) => req<OutputInstance>('POST', '/api/outputs', o),
  updateOutput: (o: OutputInstance) => req<OutputInstance>('PUT', `/api/outputs/${o.id}`, o),
  deleteOutput: (id: string) => req('DELETE', `/api/outputs/${id}`),
  testOutput: (id: string) => req('POST', `/api/outputs/${id}/test`),
  matches: () => req<MatchJson[]>('GET', '/api/matches'),

  models: () => req<ModelStatus[]>('GET', '/api/models'),
  train: (body: {
    name: string
    pipeline?: string
    epochs: number
    steps_per_epoch: number
    batch_size: number
    mix: string
    external: string[]
    init_from?: string
    workers: number
  }) => req('POST', '/api/models/train', body),
  datasets: () => req<DatasetsInfo>('GET', '/api/datasets'),

  noiseProfiles: () => req<NoiseProfile[]>('GET', '/api/noise-profiles'),
  createNoiseProfile: (body: { name: string; recording_ids: string[]; spec: PipelineSpec; position?: number }) =>
    req<NoiseProfile>('POST', '/api/noise-profiles', body),
  deleteNoiseProfile: (name: string) => req('DELETE', `/api/noise-profiles/${encodeURIComponent(name)}`),
  stopTraining: (name: string) => req('POST', `/api/models/${name}/stop`),
  modelLog: (name: string) => req<{ log: string }>('GET', `/api/models/${name}/log`),
}

export function decodeBlock(b: EncodedBlock): Float32Array {
  const bin = atob(b.values)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Float32Array(bytes.buffer)
}
