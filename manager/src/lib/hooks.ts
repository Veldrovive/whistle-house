import { useCallback, useEffect, useState } from 'react'

/** Load data with a reload() callback and error capture. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const load = useCallback(fn, deps)
  const reload = useCallback(() => {
    load()
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [load])
  useEffect(reload, [reload])
  return { data, error, reload, setData }
}

/** Wrap an async action: tracks busy + error for buttons. */
export function useAction() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true)
    setError(null)
    try {
      return await fn()
    } catch (e) {
      setError(String((e as Error).message ?? e))
    } finally {
      setBusy(false)
    }
  }, [])
  return { busy, error, run, setError }
}

export const fmtS = (s: number) => `${s.toFixed(1)}s`
export const fmtDate = (iso: string) => new Date(iso).toLocaleString()
