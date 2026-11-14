#!/usr/bin/env bash
# Build the P4 global-illumination page: scripts/build-global-illumination.sh
#
# Runs scripts/build-bricks.sh first, so every package declaration, JavaScript and WASM
# output it stages is the current brick. Staging, import rewriting, the shared/ helper
# copies and the sorted scenes.json, plus the final ready build.json, come from the
# page kit (scripts/build-page.mjs, contract 'build-page'). The staged layout is:
#   index.html, main.js, nmc.gpu.json   the delivered page entry
#   raa/      the staged @etcito/raa package (declarations, JS and WASM)
#   common/   web/common/*  (transport and page helpers)
#   lib/      examples/global-illumination/lib/*  (content, facts, inventory, runtime)
#   scenes/   every delivered .js scene module; imported for metadata only, never built
#   shared/   the reusable sky-weather helpers plus --shared examples/virtual-geometry/lib/content.js
#
# The volume stage shares the virtual-geometry content module (box, plane, sphere,
# roomShell, volumeRoom) as shared/content.js.
#
# Every earlier page root (and with it any earlier ready receipt) is removed before the
# bricks build, so a failed build never leaves a usable old receipt.
#
# Tools from PATH: node, cargo, rustc (wasm32-unknown-unknown) and wasm-bindgen at the
# Cargo.lock version. NODE=... overrides node. Arguments are not accepted.
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: scripts/build-global-illumination.sh" >&2
  exit 2
fi

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/dist/global-illumination"

# Remove the whole page (including build.json) up front: a failure below leaves no receipt.
rm -rf "$OUT"

echo "== scripts/build-bricks.sh"
bash "$ROOT/scripts/build-bricks.sh"

echo "== $OUT"
"$NODE" "$ROOT/scripts/build-page.mjs" global-illumination --stages volume,lighting,virtual \
  --shared examples/virtual-geometry/lib/content.js

echo "build-global-illumination.sh: ok ($OUT/scenes.json, $OUT/build.json)"
