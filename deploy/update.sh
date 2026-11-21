#!/usr/bin/env bash
# Deploy or update Triage on the server: pull main, build an image tagged with the commit, restart, wait until the
# API is healthy, then delete old images (keeping the previous one for deploy/rollback.sh). DEPLOY.md, section 6.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env here: create it from deploy/.env.example first (DEPLOY.md)" >&2; exit 1; }

get() { sed -n "s/^$1=//p" .env | tail -1; }
put() {   # set NAME=value in .env, replacing the line or adding it
  if grep -q "^$1=" .env; then sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak; else echo "$1=$2" >> .env; fi
}

current=$(get TRIAGE_IMAGE_TAG)
git checkout -q main
git pull --ff-only
new=$(git rev-parse --short=12 HEAD)
echo "building triage-api:$new (running now: ${current:-none})"
TRIAGE_IMAGE_TAG=$new docker compose build api

if [ -n "$current" ] && [ "$current" != "$new" ]; then put TRIAGE_PREVIOUS_IMAGE_TAG "$current"; fi
put TRIAGE_IMAGE_TAG "$new"
docker compose up -d

echo "waiting for the API to report healthy..."
for _ in $(seq 1 40); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q api)" 2>/dev/null || echo starting)
  [ "$status" = healthy ] && break
  sleep 5
done
if [ "$status" != healthy ]; then
  echo "the API is '$status' after 200 s: check 'docker compose logs api'; to go back, run deploy/rollback.sh" >&2
  exit 1
fi
echo "healthy: triage-api:$new is live"

# Keep the running image and the previous one; delete other triage-api images and unused layers
keep="$new $(get TRIAGE_PREVIOUS_IMAGE_TAG)"
for tag in $(docker images triage-api --format '{{.Tag}}'); do
  case " $keep " in *" $tag "*) ;; *) docker image rm "triage-api:$tag" >/dev/null && echo "removed triage-api:$tag" ;; esac
done
docker image prune -f >/dev/null
docker builder prune -f --keep-storage 2GB >/dev/null
