// Mirrors backend/app/schemas/{prediction,case}.py -- keep in sync by hand,
// there's no shared codegen between FastAPI and this frontend (yet).

// Fixed order matches ml/src/dataset.py:CLASSES -- the 7 HAM10000 diagnoses.
export const CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] as const

export type LesionClass = (typeof CLASSES)[number]

// Human-readable labels for clinicians, keyed by the class codes above.
export const CLASS_LABELS: Record<LesionClass, string> = {
  akiec: 'Rogowacenie słoneczne / rak kolczystokomórkowy in situ',
  bcc: 'Rak podstawnokomórkowy',
  bkl: 'Łagodne zmiany keratynocytowe',
  df: 'Włókniak twardy',
  mel: 'Czerniak',
  nv: 'Znamię melanocytowe',
  vasc: 'Zmiana naczyniowa',
}

export interface PredictionResponse {
  predicted_class: LesionClass
  probabilities: Record<LesionClass, number>
  gradcam_url: string
}

export interface CaseListItem {
  id: string
  image_path: string
  gradcam_image_path: string | null
  predicted_class: LesionClass
  created_at: string
  image_url: string
  gradcam_url: string | null
}

export interface CaseRead extends CaseListItem {
  probabilities: Record<LesionClass, number>
}
