"""Application settings loaded from .env via Pydantic."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field, validator


logger = logging.getLogger(__name__)


class SimulatorSettings(BaseSettings):
    asset_root: Path = Field(default=Path("templates"), env="ROT_GAME_ASSETS")
    output_root: Path = Field(default=Path("outputs"), env="ROT_GAME_OUTPUTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("asset_root", "output_root", pre=True)
    def _expand_root(cls, value: Path) -> Path:  # type: ignore[override]
        return Path(value).expanduser().resolve()

    @validator("asset_root")
    def _validate_asset_root(cls, value: Path) -> Path:  # type: ignore[override]
        if not value.exists():
            raise ValueError(f"Asset root directory does not exist: {value}")
        if not (value / "backgrounds").exists() or not (value / "cell_lines").exists():
            raise ValueError(
                "Asset root must contain 'backgrounds/' and 'cell_lines/' subdirectories."
            )
        return value

    @validator("output_root")
    def _ensure_output_root(cls, value: Path) -> Path:  # type: ignore[override]
        value.mkdir(parents=True, exist_ok=True)
        return value


_settings: Optional[SimulatorSettings] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_settings() -> SimulatorSettings:
    global _settings
    if _settings is None:
        env_path = _project_root() / ".env"
        if not env_path.exists():
            logger.warning(
                "No .env file found at %s. Create one with:\n"
                "ROT_GAME_ASSETS=/absolute/path/to/assets\n"
                "ROT_GAME_OUTPUTS=/absolute/path/to/outputs",
                env_path,
            )
        _settings = SimulatorSettings()
    return _settings


def asset_root() -> Path:
    return get_settings().asset_root


def output_root() -> Path:
    return get_settings().output_root


def resolve_asset_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (asset_root() / path).resolve()


def reset_settings_cache() -> None:
    global _settings
    _settings = None
