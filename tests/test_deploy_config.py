"""The single-server deploy (docker-compose.yml, Dockerfile, deploy/): the shape DEPLOY.md relies on."""
import re

import yaml

from app.registry import PROJECT_ROOT

COMPOSE = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
SERVER_ENV = PROJECT_ROOT / "deploy" / ".env.example"
DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text()

# What production must be able to set (DEPLOY.md, section 4)
PRODUCTION_VARS = {
    "API_DOMAIN", "ACME_EMAIL", "DATABASE_URL", "DATABASE_URL_UNPOOLED", "TRIAGE_DB_PREPARED_STATEMENTS",
    "TRIAGE_HASH_SECRET", "TRIAGE_ENCRYPTION_KEYS", "TRIAGE_API_KEYS", "TRIAGE_ADMIN_KEYS", "GROQ_API_KEY",
    "GEMINI_API_KEY", "TRIAGE_PUBLIC", "TRIAGE_ACCOUNTS", "TRIAGE_CORS_ORIGINS", "TRIAGE_APP_URL",
    "TRIAGE_PUBLIC_API_URL", "RESEND_API_KEY", "TRIAGE_EMAIL_FROM", "TRIAGE_EMAIL_LOGO_URL",
}
# Local development only: never in the server's template
DEV_ONLY = {"TRIAGE_FAKE_MODELS", "TRIAGE_FAKE_DB", "TRIAGE_DEV_EMAIL_LOG", "TRIAGE_TEST_DATABASE_URL",
            "TRIAGE_CACHE_PATH", "TRIAGE_LOG", "OLLAMA_BASE_URL"}


def server_env_lines():
    return [l.strip() for l in SERVER_ENV.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def test_the_server_env_template_has_names_only():
    for line in server_env_lines():
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*=", line), f"a value in deploy/.env.example: {line!r}"


def test_the_server_env_template_lists_every_production_setting_and_nothing_for_dev():
    names = {line[:-1] for line in server_env_lines()}
    assert PRODUCTION_VARS <= names, sorted(PRODUCTION_VARS - names)
    assert not names & DEV_ONLY
    assert "TRIAGE_TRUSTED_PROXY_HOPS" not in names          # fixed by docker-compose.yml's layout


def test_only_caddy_is_reachable_from_outside():
    api, caddy = COMPOSE["services"]["api"], COMPOSE["services"]["caddy"]
    assert "ports" not in api and api["expose"] == ["8000"]
    assert sorted(caddy["ports"]) == ["443:443", "443:443/udp", "80:80"]


def test_one_proxy_hop_behind_caddy():
    assert COMPOSE["services"]["api"]["environment"]["TRIAGE_TRUSTED_PROXY_HOPS"] == "1"


def test_both_services_restart_log_with_limits_and_the_api_is_health_checked():
    for name, service in COMPOSE["services"].items():
        assert service["restart"] == "unless-stopped", name
        assert service["logging"]["options"] == {"max-size": "10m", "max-file": "3"}, name
    check = " ".join(COMPOSE["services"]["api"]["healthcheck"]["test"])
    assert "/healthz" in check and "/readyz" not in check       # liveness only: never wakes the database
    assert COMPOSE["services"]["caddy"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_caddy_gets_only_its_two_settings_not_the_secrets():
    caddy = COMPOSE["services"]["caddy"]
    assert "env_file" not in caddy and set(caddy["environment"]) == {"API_DOMAIN", "ACME_EMAIL"}


def test_images_are_pinned_by_digest():
    assert re.search(r"^FROM python:3\.14\.7-slim@sha256:[0-9a-f]{64}$", DOCKERFILE, re.M)
    assert re.fullmatch(r"caddy:[0-9.]+@sha256:[0-9a-f]{64}", COMPOSE["services"]["caddy"]["image"])


def test_the_image_runs_as_a_plain_user_offline_with_the_memory_settings():
    assert re.search(r"^USER triage$", DOCKERFILE, re.M)
    for setting in ("HF_HUB_OFFLINE=1", "MALLOC_ARENA_MAX=2", "TRIAGE_EMBEDDING_THREADS=1"):
        assert setting in DOCKERFILE
    cmd = [l for l in DOCKERFILE.splitlines() if l.startswith("CMD ")][0]
    assert cmd.index("alembic upgrade head") < cmd.index("uvicorn")
    assert "--workers 1" in cmd and "--no-access-log" in cmd and "--host 0.0.0.0" in cmd


def test_the_image_gets_the_live_model_files_and_never_the_env_file():
    ignore = (PROJECT_ROOT / ".dockerignore").read_text().splitlines()
    assert ignore[[l.strip() for l in ignore].index("*")] == "*"            # allow-list
    allowed = {l[1:] for l in ignore if l.startswith("!")}
    from app.router.predictor import load_classifier_config
    name, version = load_classifier_config()
    for tier in ("low", "medium"):
        for suffix in (".joblib", ".json", ".npz"):
            assert f"models/{name}_{tier}_v{version}{suffix}" in allowed
    assert not any(".env" in a for a in allowed)


def test_caddy_streams_and_keeps_no_access_log():
    caddyfile = (PROJECT_ROOT / "deploy" / "Caddyfile").read_text()
    directives = [l.strip().split()[0] for l in caddyfile.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert "flush_interval" in directives and "log" not in directives and "encode" not in directives


def test_there_is_one_deploy_config():
    assert not (PROJECT_ROOT / "render.yaml").exists()


def test_the_api_memory_cap_is_set_per_host():
    assert COMPOSE["services"]["api"]["mem_limit"] == "${TRIAGE_API_MEM_LIMIT:-1536m}"
    assert re.search(r"^TRIAGE_API_MEM_LIMIT=$", (PROJECT_ROOT / "deploy" / ".env.example").read_text(), re.M)
