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
    assert "alembic" not in cmd                         # the app upgrades the schema itself, only when behind
    assert "--workers 1" in cmd and "--no-access-log" in cmd and "--host 0.0.0.0" in cmd
    assert "db.SCHEMA_HEAD_FILE.write_text(db.head_revision()" in DOCKERFILE        # recorded at build time
    assert "RUN python -m compileall -q app migrations" in DOCKERFILE                # no compiling at each start


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


# ---------- Render (the API's host): render.yaml ----------

RENDER = yaml.safe_load((PROJECT_ROOT / "render.yaml").read_text())
RENDER_API = RENDER["services"][0]


def test_render_runs_the_tested_image_on_the_free_plan_in_oregon():
    assert len(RENDER["services"]) == 1
    assert RENDER_API["type"] == "web" and RENDER_API["runtime"] == "image"
    assert RENDER_API["image"] == {"url": "ghcr.io/karansanghvi/triage-api:main"}       # public: no credentials
    assert RENDER_API["plan"] == "free" and RENDER_API["region"] == "oregon"
    assert RENDER_API["healthCheckPath"] == "/healthz"


def test_render_runs_the_images_own_command():
    """No start command in render.yaml: a dockerCommand once reached sh as one word (exit 127), and Render doesn't
    document how it splits one. The image's CMD is what Render runs, and what CI's smoke test runs."""
    assert "dockerCommand" not in RENDER_API and "startCommand" not in RENDER_API


def test_the_images_command_listens_on_the_hosts_port_through_a_shell():
    cmd = [l for l in DOCKERFILE.splitlines() if l.startswith("CMD ")]
    assert len(cmd) == 1
    command = cmd[0][len("CMD "):]
    assert not command.startswith("[")          # shell form: /bin/sh -c expands ${PORT:-8000} (exec form wouldn't)
    assert command.startswith("exec uvicorn app.api.main:app ")
    for part in ("--host 0.0.0.0", "--port ${PORT:-8000}", "--workers 1", "--no-access-log"):
        assert part in command


def test_the_smoke_test_runs_that_command_with_renders_port():
    script = (PROJECT_ROOT / "deploy" / "smoke_test_image.sh").read_text()
    start = script[script.index('echo "== startup smoke test"'):]
    run = start[:start.index('"$image" >/dev/null')]
    assert "--entrypoint" not in run                                  # the image's own CMD
    assert "-e PORT=10000" in run and ':10000"' in run                # reached only on $PORT, as on Render


def test_render_lists_variable_names_only():
    names = set()
    for var in RENDER_API["envVars"]:
        assert set(var) == {"key", "sync"} and var["sync"] is False, var        # no values in the repo
        names.add(var["key"])
    assert PRODUCTION_VARS - {"API_DOMAIN", "ACME_EMAIL"} <= names               # no Caddy on Render
    assert "TRIAGE_TRUSTED_PROXY_HOPS" in names and not names & DEV_ONLY




def test_the_api_memory_cap_is_set_per_host():
    assert COMPOSE["services"]["api"]["mem_limit"] == "${TRIAGE_API_MEM_LIMIT:-1536m}"
    assert re.search(r"^TRIAGE_API_MEM_LIMIT=$", (PROJECT_ROOT / "deploy" / ".env.example").read_text(), re.M)


# ---------- images built by GitHub Actions, pulled by the server ----------

WORKFLOW = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "image.yml").read_text())
REGISTRY_IMAGE = "ghcr.io/karansanghvi/triage-api"


def _steps(job):
    return WORKFLOW["jobs"][job]["steps"]


def test_images_are_built_on_pushes_to_main_only():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))          # PyYAML reads the key `on` as True
    assert triggers == {"push": {"branches": ["main"]}}


def test_actions_are_pinned_by_commit_sha():
    uses = [s["uses"] for job in WORKFLOW["jobs"].values() for s in job["steps"] if "uses" in s]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_the_token_can_write_packages_only_where_images_are_pushed():
    assert WORKFLOW["permissions"] == {}
    assert WORKFLOW["jobs"]["build"]["permissions"] == {"contents": "read", "packages": "write"}
    assert WORKFLOW["jobs"]["publish"]["permissions"] == {"packages": "write"}
    assert WORKFLOW["jobs"]["deploy"]["permissions"] == {}
    for job in WORKFLOW["jobs"]:
        for step in _steps(job):
            mentions_token = "GITHUB_TOKEN" in yaml.safe_dump(step)
            assert mentions_token == ("docker login" in step.get("run", "")), step.get("name")
    checkout = next(s for s in _steps("build") if "uses" in s)
    assert checkout["with"]["persist-credentials"] is False


def test_each_architecture_builds_natively_and_is_tested_before_it_is_pushed():
    matrix = WORKFLOW["jobs"]["build"]["strategy"]["matrix"]["include"]
    assert {(m["arch"], m["runner"]) for m in matrix} == {("amd64", "ubuntu-24.04"), ("arm64", "ubuntu-24.04-arm")}
    runs = [s.get("run", "") for s in _steps("build")]
    build = next(i for i, r in enumerate(runs) if "docker build" in r)
    test = next(i for i, r in enumerate(runs) if "deploy/smoke_test_image.sh" in r)
    push = next(i for i, r in enumerate(runs) if "docker push" in r)
    assert build < test < push
    assert "org.opencontainers.image.revision=${GITHUB_SHA}" in runs[build]


def test_the_published_image_is_tagged_with_the_commit_and_main_for_both_architectures():
    assert WORKFLOW["env"]["IMAGE"] == REGISTRY_IMAGE
    assert WORKFLOW["jobs"]["publish"]["needs"] == "build"
    run = _steps("publish")[0]["run"]
    assert '-t "$IMAGE:$GITHUB_SHA" -t "$IMAGE:main"' in run
    assert '"$IMAGE:$GITHUB_SHA-amd64" "$IMAGE:$GITHUB_SHA-arm64"' in run
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False      # :main never moves backwards


def test_the_server_pulls_the_registry_image_and_can_still_build():
    api = COMPOSE["services"]["api"]
    assert api["image"] == "${TRIAGE_IMAGE_REPO:-" + REGISTRY_IMAGE + "}:${TRIAGE_IMAGE_TAG:-main}"
    assert api["build"]["context"] == "."
    assert re.search(r"^TRIAGE_IMAGE_REPO=$", SERVER_ENV.read_text(), re.M)
    update = (PROJECT_ROOT / "deploy" / "update.sh").read_text()
    rollback = (PROJECT_ROOT / "deploy" / "rollback.sh").read_text()
    for script in (update, rollback):
        assert "IMAGE=${IMAGE:-" + REGISTRY_IMAGE + "}" in script
        assert "docker compose up -d --no-build" in script
    assert 'docker pull -q "$IMAGE:$new"' in update and "org.opencontainers.image.revision" in update
    assert "--build) build=yes" in update                              # the documented fallback
    assert 'target=${1:-$(get TRIAGE_PREVIOUS_IMAGE_TAG)}' in rollback  # any earlier commit, or the previous one
    assert "alembic downgrade" in rollback


def test_the_smoke_test_script_is_executable():
    import os
    script = PROJECT_ROOT / "deploy" / "smoke_test_image.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text()
    assert "--network none" in text and "/readyz" in text and "TRIAGE_FAKE_MODELS=1" in text


def test_each_green_push_deploys_its_own_image_to_render():
    job = WORKFLOW["jobs"]["deploy"]
    assert job["needs"] == "publish"                                   # only after both images passed and joined
    step = job["steps"][0]
    assert step["env"] == {"HOOK": "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"}
    assert '"$IMAGE:$GITHUB_SHA"' in step["run"] and "imgURL=" in step["run"]
    assert '[ -z "$HOOK" ]' in step["run"]                             # skipped until the secret exists
    manual = (PROJECT_ROOT / "deploy" / "render_deploy.sh").read_text()
    assert "RENDER_DEPLOY_HOOK_URL" in manual and "imgURL=" in manual
