#!/usr/bin/env bash
# Build the SDF Modeler comparison page: scripts/build-sdfm-compare.sh --mama <dir>
#
# <dir> is the MAMA checkout (e.g. ../MAMA) with its build (dist/, produced by `npm run build`
# in MAMA) and its node_modules. Nothing is built or written inside MAMA.
#
# MAMA's dist/ is tsc output: its modules import npm packages by bare specifier
# (@paramesh/*, three, ...), which a browser worker cannot resolve (no import map in
# workers). The page therefore gets a browser ESM bundle of the MAMA entry points the
# bridge imports (see the `## MAMA calls` section of packages/raa-sdfm/README.md), made with
# MAMA's own esbuild into dist/sdfm-compare/mama/ at the same paths as under MAMA's dist/,
# plus manifold.wasm next to the chunks (manifold-3d resolves it from import.meta.url).
#
# Runs scripts/build-package.sh, then rebuilds dist/sdfm-compare/ from
# examples/sdfm-compare/**, dist/package/raa/ (as raa/), the raa-sdfm sources (as
# raa-sdfm/), the MAMA browser build (as mama/) and web/common/* (as common/), and writes
#   dist/sdfm-compare/scenes.json  S14: [{id, milestone, file}] sorted by id, read by
#                                  importing every examples/sdfm-compare/scenes/*.js
#   dist/sdfm-compare/sizes.json   S7 of raa/pkg/raa_js_bg.wasm
#
# Override the node binary with NODE=...
set -euo pipefail

usage() {
  echo "usage: scripts/build-sdfm-compare.sh --mama <MAMA checkout or its browser build dir>" >&2
  exit 2
}

MAMA_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mama)
      [ $# -ge 2 ] || usage
      MAMA_ARG="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[ -n "$MAMA_ARG" ] || usage

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMA_BUILD_CMD="npm run build"
# Entry points under MAMA's dist/ (importMama in packages/raa-sdfm/src/mama-bridge.js, the
# compute worker setupBrowserHost loads, and the CPU material evaluator the MAMA reference
# viewport of examples/sdfm-compare/lib/mama-view.js reads base colours from).
MAMA_ENTRIES=(index.js core/game/mesh-assets.js core/material-graph/helpers.js browser/index.js browser/compute-worker.js
  core/material-graph/cpu-eval.js)

if [ ! -d "$MAMA_ARG" ]; then
  echo "build-sdfm-compare.sh: --mama $MAMA_ARG is not a directory" >&2
  exit 1
fi
MAMA_ABS="$(cd "$MAMA_ARG" && pwd)"
MAMA_DIST="$MAMA_ABS/dist"
for e in "${MAMA_ENTRIES[@]}"; do
  if [ ! -f "$MAMA_DIST/$e" ]; then
    echo "build-sdfm-compare.sh: MAMA build not found (dist/$e under $MAMA_ABS);" \
      "build it in the MAMA checkout with \`$MAMA_BUILD_CMD\` (this script never builds MAMA)" >&2
    exit 1
  fi
done
MAMA_ESBUILD="$MAMA_ABS/node_modules/.bin/esbuild"
MAMA_MANIFOLD_WASM="$MAMA_ABS/node_modules/manifold-3d/manifold.wasm"
for f in "$MAMA_ESBUILD" "$MAMA_MANIFOLD_WASM"; do
  if [ ! -e "$f" ]; then
    echo "build-sdfm-compare.sh: $f missing; install MAMA's dependencies in the MAMA checkout" >&2
    exit 1
  fi
done

cd "$ROOT"
bash "$ROOT/scripts/build-package.sh"

SRC="$ROOT/examples/sdfm-compare"
PKG="$ROOT/dist/package/raa"
SDFM="$ROOT/packages/raa-sdfm"
OUT="$ROOT/dist/sdfm-compare"

if [ ! -f "$PKG/pkg/raa_js_bg.wasm" ]; then
  echo "build-sdfm-compare.sh: dist/package/raa/pkg/raa_js_bg.wasm missing after build-package.sh" >&2
  exit 1
fi
for f in index.js index.d.ts worker.js src package.json; do
  if [ ! -e "$SDFM/$f" ]; then
    echo "build-sdfm-compare.sh: packages/raa-sdfm/$f missing" >&2
    exit 1
  fi
done

echo "== dist/sdfm-compare"
rm -rf "$OUT"
mkdir -p "$OUT"
if [ -d "$SRC" ]; then
  cp -R "$SRC/." "$OUT/"
fi
rm -rf "$OUT/raa" "$OUT/raa-sdfm" "$OUT/mama" "$OUT/common"
cp -R "$PKG" "$OUT/raa"
mkdir -p "$OUT/raa-sdfm"
cp -R "$SDFM/index.js" "$SDFM/index.d.ts" "$SDFM/worker.js" "$SDFM/src" "$SDFM/package.json" "$OUT/raa-sdfm/"
mkdir -p "$OUT/mama"
# node: imports are dynamic, node-only branches of MAMA; they stay external and never run in the page.
(cd "$MAMA_DIST" && "$MAMA_ESBUILD" "${MAMA_ENTRIES[@]}" --bundle --splitting --format=esm \
  --platform=browser --external:'node:*' --outbase=. --outdir="$OUT/mama" --log-level=error)
cp "$MAMA_MANIFOLD_WASM" "$OUT/mama/manifold.wasm"
for e in "${MAMA_ENTRIES[@]}"; do
  [ -f "$OUT/mama/$e" ] || { echo "build-sdfm-compare.sh: MAMA bundle lacks mama/$e" >&2; exit 1; }
done
mkdir -p "$OUT/common"
cp -R web/common/. "$OUT/common/"
# lib/mama-view.js imports the bridge and SdfmError from the repository package (so node tests
# run it in place); the staged page has those files under raa-sdfm/src/. The imports are
# rewritten to the staged copies because the GPU client serves the page under a sub-path, where
# the repository-relative URL points outside the page.
REPO_SDFM="'../../../packages/raa-sdfm/"
for f in src/mama-bridge.js src/errors.js; do
  grep -qF "$REPO_SDFM$f'" "$OUT/lib/mama-view.js" || {
    echo "build-sdfm-compare.sh: lib/mama-view.js no longer imports $REPO_SDFM$f'" >&2
    exit 1
  }
done
sed -i "s#$REPO_SDFM#'../raa-sdfm/#g" "$OUT/lib/mama-view.js"
if grep -qF "$REPO_SDFM" "$OUT/lib/mama-view.js"; then
  echo "build-sdfm-compare.sh: lib/mama-view.js still imports $REPO_SDFM after the rewrite" >&2
  exit 1
fi

echo "== dist/sdfm-compare/scenes.json"
N="$("$NODE" --input-type=module -e '
  import fs from "node:fs";
  import path from "node:path";
  import { pathToFileURL } from "node:url";
  const [dir, out] = process.argv.slice(1);
  const scenes = [];
  const files = fs.existsSync(dir)
    ? fs.readdirSync(dir).filter((f) => f.endsWith(".js")).sort()
    : [];
  for (const f of files) {
    const stem = f.slice(0, -3);
    const mod = await import(pathToFileURL(path.resolve(dir, f)).href);
    const s = mod.default;
    if (!s || typeof s !== "object") {
      throw new Error(`examples/sdfm-compare/scenes/${f}: no default export object`);
    }
    if (s.id !== stem) {
      throw new Error(
        `examples/sdfm-compare/scenes/${f}: id ${JSON.stringify(s.id)} does not match the file name stem ${JSON.stringify(stem)}`,
      );
    }
    if (typeof s.milestone !== "string" || !/^m\d\d$/.test(s.milestone)) {
      throw new Error(
        `examples/sdfm-compare/scenes/${f}: milestone ${JSON.stringify(s.milestone)} is not of the form mNN`,
      );
    }
    scenes.push({ id: s.id, milestone: s.milestone, file: `scenes/${f}` });
  }
  scenes.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  fs.writeFileSync(out, JSON.stringify(scenes, null, 2) + "\n");
  console.log(scenes.length);
' "$SRC/scenes" "$OUT/scenes.json")"

echo "== dist/sdfm-compare/sizes.json"
# Same S7 writer as scripts/build-package.sh.
"$NODE" -e '
  const fs = require("fs");
  const zlib = require("zlib");
  const crypto = require("crypto");
  const [file, rel, out] = process.argv.slice(1);
  const buf = fs.readFileSync(file);
  const sizes = {
    wasm: rel,
    raw: buf.length,
    gzip: zlib.gzipSync(buf, { level: 9 }).length,
    sha256: crypto.createHash("sha256").update(buf).digest("hex"),
  };
  fs.writeFileSync(out, JSON.stringify(sizes, null, 2) + "\n");
  console.log("wasm raw " + sizes.raw + " B, gzip " + sizes.gzip + " B");
' "$OUT/raa/pkg/raa_js_bg.wasm" "raa/pkg/raa_js_bg.wasm" "$OUT/sizes.json"

echo "build-sdfm-compare.sh: ok ($N scenes)"
