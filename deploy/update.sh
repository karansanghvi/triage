#!/usr/bin/env bash
# Deploy or update Triage on the server (DEPLOY.md, section 6):
#
#   deploy/update.sh            pull main, wait until GitHub Actions has published the image for that commit
#                               (.github/workflows/image.yml builds and tests it), pull it, restart, wait for a
#                               healthy API, then delete old images, keeping the previous one for a quick rollback
#   deploy/update.sh --build    the fallback: build the image on this server instead (needs swap on a 1 GB VM)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "no .env here: create it from deploy/.env.example first (DEPLOY.md)" >&2; exit 1; }

WAIT_MINUTES=${WAIT_MINUTES:-30}
build=no
case "${1:-}" in --build) build=yes ;; "") ;; *) echo "usage: deploy/update.sh [--build]" >&2; exit 2 ;; esac

get() { sed -n "s/^$1=//p" .env | tail -1; }
IMAGE=$(get TRIAGE_IMAGE_REPO); IMAGE=${IMAGE:-ghcr.io/karansanghvi/triage-api}   # .env may override it
put() {   # set NAME=value in .env, replacing the line or adding it
  if grep -q "^$1=" .env; then sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak; else echo "$1=$2" >> .env; fi
}

current=$(get TRIAGE_IMAGE_TAG)
before=$(git rev-parse HEAD)
give_up() {   # stop before deploying: leave the checkout matching what's still running
  for line in "$@"; do echo "$line" >&2; done
  git checkout -q "$before"
  exit 1
}
git checkout -q main
git pull --ff-only
new=$(git rev-parse HEAD)
echo "deploying $IMAGE:$new (running now: ${current:-none})"

if [ "$build" = yes ]; then
  TRIAGE_IMAGE_TAG=$new docker compose build api || give_up "the build failed; nothing was changed"
else
  echo "waiting for GitHub Actions to publish it (usually a few minutes after the push; up to $WAIT_MINUTES min)..."
  deadline=$((SECONDS + WAIT_MINUTES * 60))
  until out=$(docker pull -q "$IMAGE:$new" 2>&1); do
    if grep -qiE "denied|unauthorized" <<<"$out"; then
      give_up "the registry refused the pull: make the package public on GitHub (DEPLOY.md, section 6)"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      give_up "no image for $new after $WAIT_MINUTES min ($out). Check the Actions tab: a failed test stops the push." \
        "To build on this server instead: deploy/update.sh --build"
    fi
    sleep 30
  done
  revision=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE:$new")
  [ "$revision" = "$new" ] || give_up "the pulled image says it is commit '$revision', not $new: not deploying"
fi

if [ -n "$current" ] && [ "$current" != "$new" ]; then put TRIAGE_PREVIOUS_IMAGE_TAG "$current"; fi
put TRIAGE_IMAGE_TAG "$new"
docker compose up -d --no-build

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
echo "healthy: $IMAGE:$new is live"

# Keep the running image and the previous one locally (any other commit's image can be pulled again by
# deploy/rollback.sh); delete this repository's other images, and unused layers
keep="$new $(get TRIAGE_PREVIOUS_IMAGE_TAG)"
for ref in $(docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}}'); do
  case " $keep " in *" ${ref##*:} "*) ;; *) docker image rm "$ref" >/dev/null && echo "removed $ref" ;; esac
done
docker image prune -f >/dev/null
docker builder prune -f --keep-storage 1GB >/dev/null
