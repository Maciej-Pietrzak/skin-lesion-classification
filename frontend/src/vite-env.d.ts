/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI backend. Defaults to http://127.0.0.1:8000 (see src/api/client.ts). */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
