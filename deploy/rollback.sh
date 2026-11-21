#!/usr/bin/env bash
# Go back to the previous image (kept by deploy/update.sh), without rebuilding. DEPLOY.md, section 6.
# The database is NOT rolled back: if the release being undone added a migration, run its `alembic downgrade` first.
set -euo pipefail
cd "$(dirname "$0")/.."
get() { sed -n "s/^$1=//p" .env | tail -1; }
put() { sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak; }

current=$(get TRIAGE_IMAGE_TAG); previous=$(get TRIAGE_PREVIOUS_IMAGE_TAG)
[ -n "$previous" ] || { echo "no previous image recorded in .env" >&2; exit 1; }
docker image inspect "triage-api:$previous" >/dev/null 2>&1 || { echo "image triage-api:$previous is gone" >&2; exit 1; }

git checkout -q "$previous"             # the compose file and Caddyfile that went with that image
put TRIAGE_IMAGE_TAG "$previous"; put TRIAGE_PREVIOUS_IMAGE_TAG "$current"
docker compose up -d --no-build
echo "rolled back to triage-api:$previous (the code is checked out at that commit; deploy/update.sh returns to main)"
