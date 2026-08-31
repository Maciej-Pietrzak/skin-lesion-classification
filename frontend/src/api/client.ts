// Thin wrapper around the FastAPI backend -- see backend/app/api/routes/{predict,cases}.py
// for the routes this mirrors.
import type { CaseListItem, CaseRead, PredictionResponse } from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// image_url/gradcam_url in API responses are paths relative to the backend
// (e.g. "/storage/uploads/<uuid>.jpg"), not full URLs -- resolve against
// BASE_URL so <img src> works regardless of which origin the frontend runs on.
export function resolveStorageUrl(path: string): string {
  return `${BASE_URL}${path}`
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init)
  if (!response.ok) {
    // FastAPI's default error shape is {"detail": "..."} -- fall back to the
    // status text if the body isn't JSON or doesn't have that shape.
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined)
    throw new ApiError(response.status, detail ?? response.statusText)
  }
  // 204 No Content (DELETE /cases/{id}) has no body to parse.
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export function predict(file: File): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return request<PredictionResponse>('/predict', { method: 'POST', body: formData })
}

export function listCases(limit = 50, offset = 0): Promise<CaseListItem[]> {
  return request<CaseListItem[]>(`/cases?limit=${limit}&offset=${offset}`)
}

export function getCase(id: string): Promise<CaseRead> {
  return request<CaseRead>(`/cases/${id}`)
}

export function deleteCase(id: string): Promise<void> {
  return request<void>(`/cases/${id}`, { method: 'DELETE' })
}
