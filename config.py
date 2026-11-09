"""Configuration management for iRacing Setup RAG & Telemetry Engineering MCP Server."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Application configuration loaded from environment variables or .env file."""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        GARAGE61_API_KEY: str = ""
        GARAGE61_BASE_URL: str = "https://garage61.net/api/v1"
        CHROMA_DB_PATH: str = "./data/chroma_db"
        GITBOOK_CACHE_DIR: str = "./data/gitbook_docs"
        GITBOOK_ROOT_URL: str = "https://learnsetup.gitbook.io/learn-car-setup-for-iracing"
        EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

        @property
        def chroma_path(self) -> Path:
            return Path(self.CHROMA_DB_PATH).resolve()

        @property
        def gitbook_dir(self) -> Path:
            return Path(self.GITBOOK_CACHE_DIR).resolve()

except ImportError:
    # Graceful fallback if pydantic-settings is not installed in a minimal test environment
    class Settings:  # type: ignore[no-redef]
        """Fallback Settings loader reading directly from environment and .env file."""

        def __init__(self, **kwargs: str) -> None:
            env_file = Path(".env")
            file_vars: dict[str, str] = {}
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        file_vars[k.strip()] = v.strip().strip("'\"")

            def _get(key: str, default: str) -> str:
                return kwargs.get(key) or os.environ.get(key) or file_vars.get(key, default)

            self.GARAGE61_API_KEY: str = _get("GARAGE61_API_KEY", "")
            self.GARAGE61_BASE_URL: str = _get("GARAGE61_BASE_URL", "https://garage61.net/api/v1")
            self.CHROMA_DB_PATH: str = _get("CHROMA_DB_PATH", "./data/chroma_db")
            self.GITBOOK_CACHE_DIR: str = _get("GITBOOK_CACHE_DIR", "./data/gitbook_docs")
            self.GITBOOK_ROOT_URL: str = _get(
                "GITBOOK_ROOT_URL",
                "https://learnsetup.gitbook.io/learn-car-setup-for-iracing",
            )
            self.EMBEDDING_MODEL_NAME: str = _get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

        @property
        def chroma_path(self) -> Path:
            return Path(self.CHROMA_DB_PATH).resolve()

        @property
        def gitbook_dir(self) -> Path:
            return Path(self.GITBOOK_CACHE_DIR).resolve()


settings = Settings()
