# The Triage API, for a small Linux server on either architecture: linux/amd64 (e.g. Google Cloud e2-micro) or
# linux/arm64 (e.g. Oracle Cloud Ampere A1). Nothing here is architecture-specific: the base image is multi-arch and
# every package in requirements.lock has wheels for both. Built and run by docker-compose.yml; setup: DEPLOY.md.
FROM python:3.14.7-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# Every package at the exact tested version (no torch: the embedding model runs on onnxruntime)
COPY requirements.lock .
RUN pip install --root-user-action=ignore -r requirements.lock

# The embedding model, baked in at build time (pinned revision), checked to load with the network off
COPY app/ app/
COPY scripts/__init__.py scripts/fetch_embedding_model.py scripts/
RUN python -m scripts.fetch_embedding_model

COPY config/ config/
COPY migrations/ migrations/
COPY alembic.ini .
# The schema's head revision, so startup can check the database with one SELECT and import Alembic only when it
# must upgrade (app/db.py, migrate_or_check)
RUN python -c "from app import db; db.SCHEMA_HEAD_FILE.write_text(db.head_revision() + '\n')"
# Bytecode for our own code, compiled once here: the filesystem is read-only at runtime, so without it every start
# would compile app/ from source (about 10 s at a tenth of a CPU). Installed packages already have theirs.
RUN python -m compileall -q app migrations
COPY models/ models/

# Runs as a plain user. It owns only the response cache's folder; the code, models and embedding model are read-only.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin triage \
    && mkdir -p /app/data/cache && chown -R triage:triage /app/data
USER triage

# Never download at runtime; no onnxruntime telemetry to Microsoft (on by default in its builds); small memory footprint (DEPLOY.md, "Memory"); routing among the providers on this host
ENV HF_HUB_OFFLINE=1 \
    ORT_DISABLE_TELEMETRY=1 \
    MALLOC_ARENA_MAX=2 \
    TRIAGE_EMBEDDING_THREADS=1 \
    TRIAGE_SERVER_PROVIDERS=groq,gemini

EXPOSE 8000

# ONE worker (the demo, sign-in and per-minute limits, breakers and caches live in this process). It brings the
# database schema up to date itself at startup, only when it's behind (app/db.py), so no separate alembic run.
# --no-access-log: uvicorn's access log prints client IPs, and the privacy policy says Triage logs none.
# The port: $PORT where the host sets it (Render: 10000), else 8000 (docker-compose.yml).
CMD exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log
