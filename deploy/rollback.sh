#!/usr/bin/env bash
# Go back to an earlier image, without building (DEPLOY.md, section 6):
#
#   deploy/rollback.sh                      the previous image (recorded in .env by deploy/update.sh)
#   deploy/rollback.sh <commit>             any earlier commit on main that GitHub Actions published (pulled if needed)
#   deploy/rollback.sh <commit> --migrations-done
#                                           after downgrading the database yourself (see below)
#
# The database is NOT rolled back by itself. If the commit you're going back to has an older schema than the one
# running, this stops and prints the `alembic downgrade` to run first (with the newer image, which knows the way down).
set -euo pipefail
cd "$(dirname "$0")/.."
get() { sed -n "s/^$1=//p" .env | tail -1; }
IMAGE=$(get TRIAGE_IMAGE_REPO); IMAGE=${IMAGE:-ghcr.io/karansanghvi/triage-api}   # .env may override it
put() {
  if grep -q "^$1=" .env; then sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak; else echo "$1=$2" >> .env; fi
}
schema_head() { git ls-tree --name-only "$1" migrations/versions/ | sed -n 's|.*/\([0-9]\{4\}\)_.*\.py$|\1|p' | sort | tail -1; }

current=$(get TRIAGE_IMAGE_TAG)
target=${1:-$(get TRIAGE_PREVIOUS_IMAGE_TAG)}
migrations_done=no; [ "${2:-}" = "--migrations-done" ] && migrations_done=yes
[ -n "$target" ] || { echo "no previous image recorded in .env: name a commit, deploy/rollback.sh <commit>" >&2; exit 1; }
git fetch -q origin main || echo "(couldn't fetch from GitHub; using the commits already here)" >&2
target=$(git rev-parse --verify -q "$target^{commit}") || { echo "unknown commit: $1" >&2; exit 1; }
[ "$target" != "$current" ] || { echo "$target is already running" >&2; exit 1; }
git merge-base --is-ancestor "$target" origin/main ||
  echo "warning: $target isn't on main, so GitHub Actions may never have built it" >&2

if ! docker image inspect "$IMAGE:$target" >/dev/null 2>&1; then
  echo "pulling $IMAGE:$target..."
  docker pull -q "$IMAGE:$target" >/dev/null || {
    echo "no image for $target in the registry (only pushes to main get one, and a push replaced by a newer one" >&2
    echo "while waiting has none). Pick another commit, or check it out and run: docker compose build api" >&2
    exit 1; }
fi

if [ -n "$current" ] && [ "$migrations_done" = no ]; then
  now=$(schema_head "$current"); then_=$(schema_head "$target")
  if [ "$now" != "$then_" ]; then
    echo "the database schema differs: running $current has migration $now, $target expects $then_." >&2
    echo "First, with the current image still running:  docker compose exec api alembic downgrade $then_" >&2
    echo "then: deploy/rollback.sh $target --migrations-done" >&2
    exit 1
  fi
fi

git checkout -q "$target"             # the compose file and Caddyfile that went with that image
put TRIAGE_IMAGE_TAG "$target"; put TRIAGE_PREVIOUS_IMAGE_TAG "$current"
docker compose up -d --no-build
echo "rolled back to $IMAGE:$target (the code is checked out at that commit; deploy/update.sh returns to main)"
