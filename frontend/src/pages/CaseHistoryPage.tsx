// Case history list (GET /cases), paginated via limit/offset with a
// "Załaduj więcej" pattern -- kept in the same slate/teal, rounded-2xl
// visual language as UploadPredictPage ("Przewodnik").
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { ApiError, listCases, resolveStorageUrl } from '../api/client'
import { CLASS_LABELS, type CaseListItem } from '../api/types'

const PAGE_SIZE = 20

const dateFormatter = new Intl.DateTimeFormat('pl-PL', {
  dateStyle: 'medium',
  timeStyle: 'short',
})

export function CaseHistoryPage() {
  const [cases, setCases] = useState<CaseListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [error, setError] = useState(false)

  const loadPage = async (offset: number) => {
    try {
      const page = await listCases(PAGE_SIZE, offset)
      setCases((prev) => (offset === 0 ? page : [...prev, ...page]))
      setHasMore(page.length === PAGE_SIZE)
      setError(false)
    } catch (err) {
      setError(true)
      toast.error(err instanceof ApiError ? err.message : 'Nie udało się pobrać historii przypadków')
    }
  }

  useEffect(() => {
    loadPage(0).finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleLoadMore = async () => {
    setLoadingMore(true)
    await loadPage(cases.length)
    setLoadingMore(false)
  }

  return (
    <div className="mx-auto max-w-2xl px-6 py-10 font-sans text-slate-800">
      <header className="mb-8">
        <h1 className="text-lg font-semibold text-slate-900">Historia przypadków</h1>
        <p className="mt-1 text-sm text-slate-400">Wcześniej przeanalizowane zdjęcia zmian skórnych.</p>
      </header>

      {loading && <p className="text-sm text-slate-400">Ładuję…</p>}

      {!loading && error && cases.length === 0 && (
        <p className="text-sm text-slate-400">Nie udało się pobrać historii przypadków.</p>
      )}

      {!loading && !error && cases.length === 0 && (
        <p className="text-sm text-slate-400">Brak przeanalizowanych przypadków.</p>
      )}

      <ul className="flex flex-col gap-2">
        {cases.map((c) => (
          <li key={c.id}>
            <Link
              to={`/cases/${c.id}`}
              className="flex items-center gap-4 rounded-2xl border border-slate-200 bg-white p-3 transition-colors duration-150 hover:bg-slate-50"
            >
              <img
                src={resolveStorageUrl(c.image_url)}
                alt=""
                className="h-14 w-14 shrink-0 rounded-xl object-cover"
              />
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium text-slate-800">{CLASS_LABELS[c.predicted_class]}</p>
                <p className="text-xs text-slate-400">{dateFormatter.format(new Date(c.created_at))}</p>
              </div>
              <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-500">
                {c.predicted_class}
              </span>
            </Link>
          </li>
        ))}
      </ul>

      {!loading && hasMore && cases.length > 0 && (
        <button
          type="button"
          onClick={handleLoadMore}
          disabled={loadingMore}
          className="mt-6 w-full rounded-full border border-slate-300 py-2 text-sm font-medium text-slate-600 transition-[transform] duration-150 hover:bg-slate-50 active:scale-[0.98] disabled:opacity-50"
        >
          {loadingMore ? 'Ładuję…' : 'Załaduj więcej'}
        </button>
      )}
    </div>
  )
}
