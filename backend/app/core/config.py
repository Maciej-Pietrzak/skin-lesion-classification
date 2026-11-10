"""Env-driven app settings (pydantic-settings): DATABASE_URL, STORAGE_DIR, MODEL_CHECKPOINT_PATH, DEVICE, CORS origins."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, so relative paths in .env (e.g. MODEL_CHECKPOINT_PATH) resolve
# consistently regardless of the working directory the app is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    database_url: str = "postgresql://user:password@localhost:5432/skin_lesion_db"
    storage_dir: Path = BASE_DIR / "storage"
    model_checkpoint_path: Path = (
        BASE_DIR.parent / "ml" / "models" / "efficientnet_b0_ham10000_v3.pt"
    )
    device: str = "cpu"
    # Comma-separated origins in .env, e.g. "http://localhost:3000,http://localhost:5173"
    cors_origins: str = "http://localhost:3000"

    @field_validator("storage_dir", "model_checkpoint_path", mode="after")
    @classmethod
    def _resolve_relative_to_base_dir(cls, v: Path) -> Path:
        # .env may supply a relative path (e.g. MODEL_CHECKPOINT_PATH=../ml/...,
        # written assuming CWD is backend/). Anchor it to BASE_DIR instead of
        # the process's actual CWD, so behavior doesn't depend on where the
        # app/a script happens to be launched from -- matches ml/src's
        # Path(__file__)-based path handling elsewhere in the repo.
        return v if v.is_absolute() else (BASE_DIR / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    # Cached so Settings() (which reads/validates the .env file) runs once
    # per process, not on every request that depends on it.
    return Settings()
