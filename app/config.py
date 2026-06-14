from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    api_key: str = Field(default="dev-key", description="X-API-Key for protected endpoints")
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # Database
    timescaledb_url: str = Field(
        default="postgresql://sensorwatch:sensorwatch@localhost:5432/sensorwatch"
    )

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_alert_channel: str = "sensorwatch:alerts"

    # ML model
    artifacts_dir: Path = Path("artifacts/")
    window_size: int = 30
    n_features: int = 14

    # Ingest
    max_batch_size: int = 500

    # Simulator
    simulator_hz: float = 10.0
    simulator_engine_unit: int = 1

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
