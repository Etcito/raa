#!/usr/bin/env bash
# Build the P5 post page: scripts/build-post.sh
#
# Runs scripts/build-bricks.sh first, then the page kit (scripts/build-page.mjs) stages
# index.html, main.js, nmc.gpu.json, raa/, common/, lib/, scenes/ and shared/ (with
# --shared examples/virtual-geometry/lib/content.js), the sorted scenes.json and build.json.
# The earlier page root is removed first, so a failed build leaves no ready receipt.
# NODE=... overrides node. Arguments are not accepted.
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: scripts/build-post.sh" >&2
  exit 2
fi

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/dist/post"

rm -rf "$OUT"

echo "== scripts/build-bricks.sh"
bash "$ROOT/scripts/build-bricks.sh"

echo "== $OUT"
"$NODE" "$ROOT/scripts/build-page.mjs" post --stages exposure,sharpen,mips \
  --shared examples/virtual-geometry/lib/content.js

echo "build-post.sh: ok ($OUT/scenes.json, $OUT/build.json)"
