import { useState } from 'react'
import type { JsonSchema } from '../api'

// Renders a form for a pydantic model's JSON schema (stage/output Config classes).
// Simple field types get native inputs; anything else falls back to a JSON editor,
// so new stages work in the manager without frontend changes.

type Props = {
  schema: JsonSchema
  value: Record<string, unknown>
  onChange: (v: Record<string, unknown>) => void
  /** fields edited by a dedicated panel instead (e.g. adaptive_gate's `profiles`) */
  hide?: string[]
}

function nullableOf(s: JsonSchema): JsonSchema | null {
  if (!s.anyOf || s.anyOf.length !== 2) return null
  const nonNull = s.anyOf.filter((x) => x.type !== 'null')
  return nonNull.length === 1 ? nonNull[0] : null
}

function JsonField({ value, onChange }: { value: unknown; onChange: (v: unknown) => void }) {
  const [text, setText] = useState(JSON.stringify(value ?? null))
  const [bad, setBad] = useState(false)
  return (
    <input
      className={bad ? 'bad mono' : 'mono'}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        try {
          onChange(JSON.parse(e.target.value))
          setBad(false)
        } catch {
          setBad(true)
        }
      }}
    />
  )
}

function Field({
  schema,
  value,
  onChange,
}: {
  schema: JsonSchema
  value: unknown
  onChange: (v: unknown) => void
}) {
  const inner = nullableOf(schema)
  const s = inner ?? schema
  if (s.enum) {
    return (
      <select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
        {s.enum.map((o) => (
          <option key={String(o)}>{String(o)}</option>
        ))}
      </select>
    )
  }
  if (s.type === 'boolean') {
    return <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
  }
  if (s.type === 'number' || s.type === 'integer') {
    return (
      <input
        type="number"
        step={s.type === 'integer' ? 1 : 'any'}
        value={value === null || value === undefined ? '' : String(value)}
        placeholder={inner ? 'null' : ''}
        onChange={(e) => {
          if (e.target.value === '') return onChange(inner ? null : 0)
          onChange(s.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))
        }}
      />
    )
  }
  if (s.type === 'string') {
    return <input value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
  }
  return <JsonField value={value} onChange={onChange} />
}

export function SchemaForm({ schema, value, onChange, hide }: Props) {
  const props = Object.fromEntries(Object.entries(schema.properties ?? {}).filter(([k]) => !hide?.includes(k)))
  if (!Object.keys(props).length) return <div className="muted small">no parameters</div>
  return (
    <div className="form-grid">
      {Object.entries(props).map(([k, s]) => (
        <label key={k} title={s.description}>
          <span>
            {s.title ?? k}
            {s.description && <em className="muted small"> — {s.description}</em>}
          </span>
          <Field
            schema={s}
            value={value[k] !== undefined ? value[k] : s.default}
            onChange={(v) => onChange({ ...value, [k]: v })}
          />
        </label>
      ))}
    </div>
  )
}
