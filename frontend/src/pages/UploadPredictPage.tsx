// Main screen: upload a lesion image, show the prediction + Grad-CAM overlay.
// One focus at a time (upload -> analiza -> wynik), analysis starts
// automatically on file selection, full probability breakdown is
// progressively disclosed behind "Pokaż pełny rozkład prawdopodobieństw".
//
// Chosen from 3 variants explored via the `prototype` skill (Raport,
// Przewodnik, Stanowisko) -- this was "Przewodnik".
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { ApiError, predict, resolveStorageUrl } from '../api/client'
import { CLASS_LABELS, type PredictionResponse } from '../api/types'

type Step = 'upload' | 'analyzing' | 'result'

// Entrance transition for step changes: opacity + a few px of translateY,
// ease-out, under 200ms -- transform/opacity only, per emil-design-eng.
function FadeIn({ children }: { children: React.ReactNode }) {
  const [entered, setEntered] = useState(false)
  useEffect(() => {
    const raf = requestAnimationFrame(() => setEntered(true))
    return () => cancelAnimationFrame(raf)
  }, [])
  return (
    <div
      style={{
        opacity: entered ? 1 : 0,
        transform: entered ? 'translateY(0)' : 'translateY(6px)',
        transition: 'opacity 200ms cubic-bezier(0.23,1,0.32,1), transform 200ms cubic-bezier(0.23,1,0.32,1)',
      }}
    >
      {children}
    </div>
  )
}

export function UploadPredictPage() {
  const [step, setStep] = useState<Step>('upload')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [showImage, setShowImage] = useState<'original' | 'gradcam'>('gradcam')
  const [showAll, setShowAll] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const runAnalysis = async (file: File) => {
    setPreviewUrl(URL.createObjectURL(file))
    setStep('analyzing')
    try {
      const response = await predict(file)
      setResult(response)
      setShowImage('gradcam')
      setShowAll(false)
      setStep('result')
    } catch (err) {
      setStep('upload')
      toast.error(err instanceof ApiError ? err.message : 'Nie udało się przeanalizować obrazu')
    }
  }

  const steps: { key: Step; label: string }[] = [
    { key: 'upload', label: 'Zdjęcie' },
    { key: 'analyzing', label: 'Analiza' },
    { key: 'result', label: 'Wynik' },
  ]
  const stepIndex = steps.findIndex((s) => s.key === step)

  const ranked = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []
  const top = ranked[0]

  return (
    <div className="flex min-h-screen flex-col items-center bg-slate-50 px-6 pt-16 pb-28 font-sans text-slate-800">
      <ol className="mb-12 flex items-center gap-3 text-sm">
        {steps.map((s, i) => (
          <li key={s.key} className="flex items-center gap-3">
            <span
              className={`flex h-6 w-6 items-center justify-center rounded-full text-xs transition-colors duration-150 ${
                i <= stepIndex ? 'bg-teal-600 text-white' : 'bg-slate-200 text-slate-400'
              }`}
            >
              {i + 1}
            </span>
            <span className={i <= stepIndex ? 'text-slate-700' : 'text-slate-400'}>{s.label}</span>
            {i < steps.length - 1 && <span className="ml-3 h-px w-8 bg-slate-200" />}
          </li>
        ))}
      </ol>

      {step === 'upload' && (
        <FadeIn>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setIsDragging(false)
              const f = e.dataTransfer.files[0]
              if (f) runAnalysis(f)
            }}
            onClick={() => inputRef.current?.click()}
            className={`flex w-[420px] cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-10 py-14 text-center transition-colors duration-150 ${
              isDragging ? 'border-teal-400 bg-teal-50' : 'border-slate-300 bg-white'
            }`}
          >
            <svg viewBox="0 0 24 24" className="h-10 w-10 text-slate-300" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 16.5V19a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2.5M12 15V4m0 0-4 4m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="font-medium text-slate-700">Upuść zdjęcie zmiany skórnej</p>
            <p className="text-sm text-slate-400">lub kliknij, aby wybrać plik — analiza rozpocznie się automatycznie</p>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) runAnalysis(f)
              }}
            />
          </div>
        </FadeIn>
      )}

      {step === 'analyzing' && (
        <FadeIn>
          <div className="flex flex-col items-center gap-5 py-10">
            {previewUrl && (
              <img src={previewUrl} alt="" className="h-40 w-40 rounded-2xl object-cover opacity-70" />
            )}
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-200 border-t-teal-600" />
            <p className="text-slate-500">Analizuję zdjęcie…</p>
          </div>
        </FadeIn>
      )}

      {step === 'result' && result && top && (
        <FadeIn>
          <div className="w-[420px]">
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <img
                src={showImage === 'original' ? (previewUrl ?? '') : resolveStorageUrl(result.gradcam_url)}
                alt=""
                className="aspect-square w-full object-cover"
              />
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
            </div>

            <div className="mt-6 text-center">
              <p className="text-2xl font-semibold text-slate-900">{CLASS_LABELS[top[0] as keyof typeof CLASS_LABELS]}</p>
              <p className="mt-1 text-teal-600">{(top[1] * 100).toFixed(1)}% pewności modelu</p>
            </div>

            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="mt-5 w-full text-center text-sm text-slate-500 underline-offset-2 hover:underline"
            >
              {showAll ? 'Ukryj pełny rozkład' : 'Pokaż pełny rozkład prawdopodobieństw'}
            </button>

            {showAll && (
              <FadeIn>
                <ul className="mt-3 space-y-1.5 text-sm">
                  {ranked.map(([cls, prob]) => (
                    <li key={cls} className="flex items-center gap-2">
                      <span className="w-40 truncate text-slate-500">
                        {CLASS_LABELS[cls as keyof typeof CLASS_LABELS]}
                      </span>
                      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                        <span
                          className="block h-full rounded-full bg-teal-500"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </span>
                      <span className="w-12 text-right font-mono text-xs text-slate-400">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
                </ul>
              </FadeIn>
            )}

            <button
              type="button"
              onClick={() => {
                setStep('upload')
                setResult(null)
                setPreviewUrl(null)
              }}
              className="mt-8 w-full rounded-full border border-slate-300 py-2 text-sm font-medium text-slate-600 transition-[transform] duration-150 hover:bg-slate-50 active:scale-[0.98]"
            >
              Nowa analiza
            </button>
          </div>
        </FadeIn>
      )}
    </div>
  )
}
