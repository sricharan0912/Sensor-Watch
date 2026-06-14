# ── Stage 1: dependency resolver ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

WORKDIR /app
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app ./app
COPY ml ./ml
COPY simulator ./simulator

# artifacts/ is mounted at runtime; create the dir so the path is valid
RUN mkdir -p artifacts

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health/liveness || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
