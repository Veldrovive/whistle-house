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
/** One written note (pitch = MIDI number) or rest (pitch null). A quarter note is 1 beat. */
export type ScoreNote = { pitch: number | null; beats: number; flat?: boolean }
/** A tune as sheet music; matched by the score_align recognizer in any key and tempo. */
export type Score = { tempo_bpm: number; beats_per_bar: number; notes: ScoreNote[] }
export type Tune = {
  id: string
  name: string
  description: string
  enabled: boolean
  examples: string[]
  template: { notes: NoteTemplate[]; log_span: FeatureStat; n_examples: number } | null
  score: Score | null
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
  /** how the note was made: 'whistle' or 'hum' (absent in old data = whistle) */
  kind?: string
}
/** Per-kind detector scores per frame, from detectors that tell kinds apart (learned). */
export type KindScores = Record<string, number[]>
export type MatchJson = {
  tune_id: string
  tune_name: string
  time: number
  score: number
  /** whether the tune was whistled or hummed */
  kind?: string
  events: EventJson[]
  wall_time?: number
  /** score_align: the key (semitones above the written pitch) and tempo it was performed
   *  in, and where each written note was found */
  meta?: {
    key?: number
    tempo_bpm?: number
    slurred?: number
    notes?: { start: number; end: number; hz: number; pitch: number }[]
  }
}
/** pYIN pitch track of a recording (offline hum labeller); f0 = 0 where unvoiced. */
export type HumPitch = {
  rate: number
  f0: number[]
  harmonics_db: (number | null)[]
  voiced_fraction: number
  f0_median?: number
  f0_p5?: number
  f0_p95?: number
}

export type Analysis = {
  pipeline: PipelineSpec
  raw_axis: Axis
  axis: Axis
  raw: EncodedBlock
  filtered: EncodedBlock
  scores: number[]
  kind_scores?: KindScores
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
  // One entry per validation (epoch). Fine-tunes start with epoch 0 = the init checkpoint
  // (loss null). mic_train_* / mic_val_* / overfit_gap appear when mic whistle clips are used.
  // Models with several classes add <class>_<metric> (whistle_f1, hum_f1, ...); the plain
  // names are then the mean over classes.
  history: Record<string, number | null>[]
  classes?: string[] // what the model detects: ['whistle'] or ['whistle', 'hum']
  best?: Record<string, number | null>
  stopped_early?: number // epoch at which `patience` ended the run
  data?: Record<string, Record<string, number>> // {train: {...counts}, val: {...counts}}
  init_from?: string
  error?: string
  device?: string
  wandb?: { id: string; url: string; project: string } | null // Weights & Biases run, if logged
  params?: number
  running: boolean
  has_checkpoint: boolean
  // not running, not done, and a resume state exists: POST /resume continues it
  can_resume: boolean
  resumed?: { epoch: number; time: number }[] // one entry per resume: the epoch it continued after
  report: ModelReportStatus
  onnx: ModelOnnxStatus
}

// ONNX export of a checkpoint (<name>.onnx), run by the learned detector's `onnx` backend
export type ModelOnnxStatus = {
  exists: boolean
  exported: number | null // unix time of the export
  stale: boolean // the checkpoint changed after the export
}

// Layer-by-layer report of a whistle + hum checkpoint (whistle-house inspect-model)
export type ModelReportStatus = {
  running: boolean
  exists: boolean
  built: number | null // unix time of the last build
  failed: boolean
  log?: string // tail of the build log, when it failed
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
  humPitch: (id: string) => req<HumPitch>('GET', `/api/recordings/${id}/hum-pitch`),
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
    val_examples: number
    patience: number
    arch: string
    wandb_project?: string
  }) => req('POST', '/api/models/train', body),
  datasets: () => req<DatasetsInfo>('GET', '/api/datasets'),

  noiseProfiles: () => req<NoiseProfile[]>('GET', '/api/noise-profiles'),
  createNoiseProfile: (body: { name: string; recording_ids: string[]; spec: PipelineSpec; position?: number }) =>
    req<NoiseProfile>('POST', '/api/noise-profiles', body),
  deleteNoiseProfile: (name: string) => req('DELETE', `/api/noise-profiles/${encodeURIComponent(name)}`),
  stopTraining: (name: string) => req('POST', `/api/models/${name}/stop`),
  // continue an interrupted run with the settings it was started with
  resumeTraining: (name: string) => req('POST', `/api/models/${name}/resume`),
  modelLog: (name: string) => req<{ log: string }>('GET', `/api/models/${name}/log`),
  // (re)build a model's report; open it at modelReportUrl when status.report.exists
  buildModelReport: (name: string) => req('POST', `/api/models/${name}/report`),
  // export a checkpoint to <name>.onnx (a few seconds; verified against PyTorch)
  exportModelOnnx: (name: string) => req<ModelOnnxStatus>('POST', `/api/models/${name}/export-onnx`),
}

export const modelReportUrl = (name: string) => `/api/models/${encodeURIComponent(name)}/report/index.html`

export function decodeBlock(b: EncodedBlock): Float32Array {
  const bin = atob(b.values)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Float32Array(bytes.buffer)
}
