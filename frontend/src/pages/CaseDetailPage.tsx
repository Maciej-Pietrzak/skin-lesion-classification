// Single case detail (GET /cases/{id}) -- same Oryginał/Grad-CAM toggle and
// probability list styling as UploadPredictPage ("Przewodnik"), but the full
// breakdown is shown directly (no progressive disclosure -- this is already
// a deliberate "show me everything" view).
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { ApiError, getCase, resolveStorageUrl } from '../api/client'
import { CLASS_LABELS, type CaseRead } from '../api/types'

const dateFormatter = new Intl.DateTimeFormat('pl-PL', {
  dateStyle: 'medium',
  timeStyle: 'short',
})

export function CaseDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [caseData, setCaseData] = useState<CaseRead | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [showImage, setShowImage] = useState<'original' | 'gradcam'>('gradcam')

  useEffect(() => {
    if (!id) return
    setLoading(true)
    getCase(id)
      .then((data) => {
        setCaseData(data)
        setShowImage(data.gradcam_url ? 'gradcam' : 'original')
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true)
        } else {
          toast.error(err instanceof ApiError ? err.message : 'Nie udało się pobrać przypadku')
        }
      })
      .finally(() => setLoading(false))
  }, [id])

  const ranked = caseData
    ? Object.entries(caseData.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="mx-auto max-w-2xl px-6 py-10 font-sans text-slate-800">
      <Link to="/cases" className="text-sm text-slate-400 transition-colors duration-150 hover:text-slate-600">
        ← Powrót do historii
      </Link>

      {loading && <p className="mt-6 text-sm text-slate-400">Ładuję…</p>}

      {!loading && notFound && (
        <p className="mt-6 text-sm text-slate-400">Nie znaleziono przypadku o podanym identyfikatorze.</p>
      )}

      {!loading && caseData && (
        <div className="mt-6 w-[420px] max-w-full">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
            <img
              src={resolveStorageUrl(
                showImage === 'original' || !caseData.gradcam_url ? caseData.image_url : caseData.gradcam_url,
              )}
              alt=""
              className="aspect-square w-full object-cover"
            />
            {caseData.gradcam_url && (
              <div className="flex justify-center gap-1 border-t border-slate-100 p-2">
                <button
                  type="button"
                  onClick={() => setShowImage('original')}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors duration-150 ${
                    showImage === 'original' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100'
                  }`}
                >
                  Oryginał
                </button>
                <button
                  type="button"
                  onClick={() => setShowImage('gradcam')}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors duration-150 ${
                    showImage === 'gradcam' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100'
                  }`}
                >
                  Grad-CAM
                </button>
              </div>
            )}
          </div>

          <div className="mt-6">
            <p className="text-2xl font-semibold text-slate-900">{CLASS_LABELS[caseData.predicted_class]}</p>
            <p className="mt-1 text-sm text-slate-400">{dateFormatter.format(new Date(caseData.created_at))}</p>
          </div>

          <ul className="mt-6 space-y-1.5 text-sm">
            {ranked.map(([cls, prob]) => (
              <li key={cls} className="flex items-center gap-2">
                <span className="w-40 truncate text-slate-500">
                  {CLASS_LABELS[cls as keyof typeof CLASS_LABELS]}
                </span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                  <span
                    className={`block h-full rounded-full ${cls === caseData.predicted_class ? 'bg-teal-500' : 'bg-slate-300'}`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </span>
                <span className="w-12 text-right font-mono text-xs text-slate-400">
                  {(prob * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
