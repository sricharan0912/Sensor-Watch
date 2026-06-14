from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
