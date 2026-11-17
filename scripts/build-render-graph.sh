#!/usr/bin/env bash
# Build the render-graph page: scripts/build-render-graph.sh
#
# Runs scripts/build-bricks.sh first, then the page kit (scripts/build-page.mjs) with the
# stages whose scene file examples/render-graph/scenes/<stage>.js exists, in stage rank
# order. Output: dist/render-graph/ with scenes.json and the ready build.json. The page
# root is removed first, so a failed build never leaves a usable old receipt.
# NODE=... overrides node. Arguments are not accepted.
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: scripts/build-render-graph.sh" >&2
  exit 2
fi

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/dist/render-graph"
rm -rf "$OUT"

echo "== scripts/build-bricks.sh"
bash "$ROOT/scripts/build-bricks.sh"

STAGES=""
for stage in interpreter views parameters wiring slots gi-stages post-stages compute frozen; do
  if [ -f "$ROOT/examples/render-graph/scenes/$stage.js" ]; then
    STAGES="${STAGES:+$STAGES,}$stage"
  fi
done
if [ -z "$STAGES" ]; then
  echo "build-render-graph.sh: no scene file under examples/render-graph/scenes/" >&2
  exit 1
fi

echo "== $OUT (stages: $STAGES)"
"$NODE" "$ROOT/scripts/build-page.mjs" render-graph --stages "$STAGES"

echo "build-render-graph.sh: ok ($OUT/scenes.json, $OUT/build.json)"
