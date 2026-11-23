#!/usr/bin/env bash
# Checks a built API image before it is pushed (.github/workflows/image.yml runs this) or deployed by hand:
#
#   deploy/smoke_test_image.sh triage-api:ci
#
# 1. Offline model check, with networking off: the embedding model loads from the image's own cache and gives a
#    384-number vector; onnxruntime's telemetry is off; torch and transformers aren't installed; it runs as uid 10001.
# 2. Startup smoke test: the image's own CMD with PORT=10000, exactly as Render runs it (so a CMD that ignored $PORT
#    fails here), read-only with no capabilities, as docker-compose.yml runs it. Fake models, a throwaway SQLite
#    database and throwaway secrets (nothing real, nothing sent anywhere). /healthz and /readyz report ok, a short
#    and a long (~18,000-character) prompt are routed and answered, and the logs show no traceback.
set -euo pipefail
image=${1:?usage: deploy/smoke_test_image.sh <image>}
name="triage-smoke-$$"
port=${SMOKE_PORT:-18000}
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== offline model check ($image)"
docker run --rm --network none --entrypoint python "$image" -c '
import importlib.util, os, sys
from app.router.embeddings import get_model, embed_one
assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE is not 1"
v = embed_one("hello from the smoke test")
assert v.shape == (384,), v.shape
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1", "onnxruntime telemetry is not off"
assert get_model().session.get_providers() == ["CPUExecutionProvider"]
for module in ("torch", "transformers", "sentence_transformers"):
    assert importlib.util.find_spec(module) is None, module + " is installed"
assert os.getuid() == 10001, os.getuid()
import app.api.main, pathlib
assert app.api.main.__cached__ and pathlib.Path(app.api.main.__cached__).exists(), "app/ has no precompiled bytecode"
print("offline model check ok")
'

echo "== startup smoke test"
docker run -d --name "$name" --read-only --tmpfs /tmp --tmpfs /app/data/cache:uid=10001,gid=10001 \
  --cap-drop ALL --security-opt no-new-privileges:true -e PORT=10000 -p "127.0.0.1:$port:10000" \
  -e TRIAGE_FAKE_MODELS=1 -e TRIAGE_FAKE_DB=/app/data/cache/fake.sqlite3 \
  -e DATABASE_URL=sqlite:////app/data/cache/smoke.sqlite3 \
  -e TRIAGE_HASH_SECRET="$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')" \
  -e TRIAGE_API_KEYS=smoke-test-key -e TRIAGE_PUBLIC=on -e TRIAGE_ACCOUNTS=off \
  "$image" >/dev/null

base="http://127.0.0.1:$port"
for _ in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "$base/healthz" || true)" = 200 ] && break
  if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]; then
    docker logs "$name" 2>&1 | tail -30; echo "the API exited during startup" >&2; exit 1
  fi
  sleep 2
done
check() {   # check <what> <expected> <actual>
  if [ "$2" != "$3" ]; then docker logs "$name" 2>&1 | tail -30; echo "FAILED: $1: expected $2, got $3" >&2; exit 1; fi
  echo "ok: $1"
}
check "/healthz" 200 "$(curl -s -o /dev/null -w '%{http_code}' "$base/healthz")"
check "/readyz" '"ok"' "$(curl -s "$base/readyz" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["database"]))')"

long=$(python3 -c "print('Compare transformers, RNNs and CNNs in a detailed table. ' + 'Background on models, attention, latency and cost. ' * 350)")
for prompt in "What is 2+3?" "$long"; do
  body=$(python3 -c 'import json,sys; print(json.dumps({"model": "auto", "messages": [{"role": "user", "content": sys.argv[1]}]}))' "$prompt")
  headers=$(mktemp)
  answer=$(curl -s -D "$headers" "$base/v1/chat/completions" -H "Authorization: Bearer smoke-test-key" \
    -H "Content-Type: application/json" -d "$body")
  check "a ${#prompt}-character prompt is answered" yes \
    "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print("yes" if d["choices"][0]["message"]["content"] else "no")' "$answer" 2>/dev/null || echo "no: $answer")"
  check "it was routed by the classifier" yes "$(grep -qi '^x-triage-routed-by: classifier' "$headers" && echo yes || echo no)"
  rm -f "$headers"
done

check "no traceback in the logs" 0 "$(docker logs "$name" 2>&1 | grep -c -E 'Traceback|refusing to start' || true)"
echo "smoke test passed: $image"
