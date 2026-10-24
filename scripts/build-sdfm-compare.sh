#!/usr/bin/env bash
# Build the SDF Modeler comparison page: scripts/build-sdfm-compare.sh --mama <dir>
#
# <dir> is the MAMA checkout (e.g. ../MAMA) or directly its browser build directory
# (MAMA's dist/browser/, produced by `npm run build` in MAMA; see the `## MAMA calls`
# section of packages/raa-sdfm/README.md). Nothing is built or written inside MAMA.
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
MAMA_BUILD_REL="dist/browser"
MAMA_BUILD_CMD="npm run build"
MAMA_MARKER="compute-worker.js"

if [ ! -d "$MAMA_ARG" ]; then
  echo "build-sdfm-compare.sh: --mama $MAMA_ARG is not a directory" >&2
  exit 1
fi
MAMA_ABS="$(cd "$MAMA_ARG" && pwd)"
if [ -f "$MAMA_ABS/$MAMA_MARKER" ]; then
  MAMA_BUILD="$MAMA_ABS"
elif [ -f "$MAMA_ABS/$MAMA_BUILD_REL/$MAMA_MARKER" ]; then
  MAMA_BUILD="$MAMA_ABS/$MAMA_BUILD_REL"
else
  echo "build-sdfm-compare.sh: MAMA browser build not found ($MAMA_BUILD_REL/$MAMA_MARKER under $MAMA_ABS);" \
    "build it in the MAMA checkout with \`$MAMA_BUILD_CMD\` (this script never builds MAMA)" >&2
  exit 1
fi

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
cp -R "$MAMA_BUILD" "$OUT/mama"
mkdir -p "$OUT/common"
cp -R web/common/. "$OUT/common/"

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
