# The Triage API, for an ARM (aarch64) Linux server: Oracle Cloud Ampere A1. Built and run by docker-compose.yml;
# setup and updates: DEPLOY.md. Builds on x86_64 too (requirements.lock has CPU-only torch for both).
FROM python:3.14.7-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# Every package at the exact tested version, CPU-only torch
COPY requirements.lock .
RUN pip install --root-user-action=ignore -r requirements.lock

# The embedding model, baked in at build time (pinned revision), checked to load with the network off
COPY app/ app/
COPY scripts/__init__.py scripts/fetch_embedding_model.py scripts/
RUN python -m scripts.fetch_embedding_model

COPY config/ config/
COPY migrations/ migrations/
COPY alembic.ini .
COPY models/ models/

# Runs as a plain user. It owns only the response cache's folder; the code, models and embedding model are read-only.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin triage \
    && mkdir -p /app/data/cache && chown -R triage:triage /app/data
USER triage

# Never download at runtime; small memory footprint (DEPLOY.md, "Memory"); routing among the providers on this host
ENV HF_HUB_OFFLINE=1 \
    MALLOC_ARENA_MAX=2 \
    TRIAGE_EMBEDDING_THREADS=1 \
    TRIAGE_SERVER_PROVIDERS=groq,gemini

EXPOSE 8000

# Schema first, then ONE worker (the demo, sign-in and per-minute limits, breakers and caches live in this process).
# --no-access-log: uvicorn's access log prints client IPs, and the privacy policy says Triage logs none.
CMD alembic upgrade head && exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
