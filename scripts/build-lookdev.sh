#!/usr/bin/env bash
# Build the lookdev page: scripts/build-lookdev.sh
#
# Runs scripts/build-bricks.sh, then rebuilds dist/lookdev/ from examples/lookdev/**,
# the staged package dist/package/raa/ (as raa/), the brick dist/bricks/raa-anim/ (as
# anim/) and web/common/* (as common/), and writes
#   dist/lookdev/scenes.json  S14: [{id, milestone, file}] sorted by id, read by importing
#                             the default export of every examples/lookdev/scenes/*.js
#                             (a file whose id differs from its file name stem is refused)
#   dist/lookdev/sizes.json   S7 of raa/pkg/raa_js_bg.wasm
#
# Override the node binary with NODE=... Arguments are not accepted.
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: scripts/build-lookdev.sh" >&2
  exit 2
fi

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash "$ROOT/scripts/build-bricks.sh"

SRC="$ROOT/examples/lookdev"
PKG="$ROOT/dist/package/raa"
ANIM="$ROOT/dist/bricks/raa-anim"
OUT="$ROOT/dist/lookdev"

if [ ! -f "$PKG/pkg/raa_js_bg.wasm" ]; then
  echo "build-lookdev.sh: dist/package/raa/pkg/raa_js_bg.wasm missing after build-bricks.sh" >&2
  exit 1
fi
if [ ! -f "$ANIM/index.js" ] || [ ! -f "$ANIM/pkg/raa_anim_js_bg.wasm" ] || [ ! -f "$ANIM/pkg/raa_anim_js.js" ]; then
  echo "build-lookdev.sh: dist/bricks/raa-anim is incomplete after build-bricks.sh" >&2
  exit 1
fi

echo "== dist/lookdev"
rm -rf "$OUT"
mkdir -p "$OUT"
if [ -d "$SRC" ]; then
  cp -R "$SRC/." "$OUT/"
fi
rm -rf "$OUT/raa" "$OUT/common"
cp -R "$PKG" "$OUT/raa"
cp -R "$ANIM" "$OUT/anim"
mkdir -p "$OUT/common"
cp -R web/common/. "$OUT/common/"

echo "== dist/lookdev/scenes.json"
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
      throw new Error(`examples/lookdev/scenes/${f}: no default export object`);
    }
    if (s.id !== stem) {
      throw new Error(
        `examples/lookdev/scenes/${f}: id ${JSON.stringify(s.id)} does not match the file name stem ${JSON.stringify(stem)}`,
      );
    }
    if (typeof s.milestone !== "string" || !/^m\d\d$/.test(s.milestone)) {
      throw new Error(
        `examples/lookdev/scenes/${f}: milestone ${JSON.stringify(s.milestone)} is not of the form mNN`,
      );
    }
    scenes.push({ id: s.id, milestone: s.milestone, file: `scenes/${f}` });
  }
  scenes.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  fs.writeFileSync(out, JSON.stringify(scenes, null, 2) + "\n");
  console.log(scenes.length);
' "$SRC/scenes" "$OUT/scenes.json")"

echo "== dist/lookdev/sizes.json"
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

echo "build-lookdev.sh: ok ($N scenes)"
