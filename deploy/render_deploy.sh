#!/usr/bin/env bash
# Deploy one commit's image to the Render service by hand: a rollback to any earlier commit, or a redeploy
# (DEPLOY.md, section 6). GitHub Actions does the same after every green push to main.
#
#   RENDER_DEPLOY_HOOK_URL='<the service's deploy hook URL>' deploy/render_deploy.sh <commit>
#
# The hook URL is a secret (it can deploy the service): pass it in the environment, never write it to a file here.
set -euo pipefail
IMAGE=ghcr.io/karansanghvi/triage-api
hook=${RENDER_DEPLOY_HOOK_URL:?set RENDER_DEPLOY_HOOK_URL to the deploy hook URL from Render: Settings, Deploy Hook}
commit=$(git rev-parse --verify -q "${1:?usage: deploy/render_deploy.sh <commit>}^{commit}") || { echo "unknown commit: $1" >&2; exit 1; }

# The image must exist: only commits pushed to main whose checks passed have one
if command -v docker >/dev/null; then
  docker buildx imagetools inspect "$IMAGE:$commit" >/dev/null 2>&1 ||
    { echo "no image $IMAGE:$commit (only green pushes to main get one)" >&2; exit 1; }
fi

encoded=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$IMAGE:$commit")
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$hook&imgURL=$encoded")
[ "$code" = 200 ] || [ "$code" = 202 ] || { echo "Render refused the deploy (HTTP $code)" >&2; exit 1; }
echo "deploy of $IMAGE:$commit started: follow it in the Render Dashboard (Events)"
