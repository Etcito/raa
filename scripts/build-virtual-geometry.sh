#!/usr/bin/env bash
# Build the P3 virtual-geometry page: scripts/build-virtual-geometry.sh
#
# Runs scripts/build-bricks.sh first, so every package declaration, JavaScript and WASM
# output it stages is the current brick, then stages dist/virtual-geometry/:
#   index.html, main.js, nmc.gpu.json   the delivered page entry
#   raa/      the staged @etcito/raa package (declarations, JS and WASM)
#   common/   web/common/*  (transport and page helpers)
#   lib/      examples/virtual-geometry/lib/*  (content, preview, facts, inventory, runtime)
#   scenes/   every delivered .js scene module; imported for metadata only, never built
#   shared/   the reusable sky-weather helpers (capture, measure, readiness, evidence)
#   scenes.json  [{id, milestone, file}] sorted by id, file relative to the page root
#   build.json   the stage-independent ready BuildReceipt, written last
#
# Every earlier page root (and with it any earlier ready receipt) is removed before the
# bricks build, so a failed build never leaves a usable old receipt.
#
# Tools from PATH: node, cargo, rustc (wasm32-unknown-unknown) and wasm-bindgen at the
# Cargo.lock version. NODE=... overrides node. Arguments are not accepted.
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: scripts/build-virtual-geometry.sh" >&2
  exit 2
fi

NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PKG="$ROOT/dist/package/raa"
SRC="$ROOT/examples/virtual-geometry"
OUT="$ROOT/dist/virtual-geometry"

# Remove the whole page (including build.json) up front: a failure below leaves no receipt.
rm -rf "$OUT"

echo "== scripts/build-bricks.sh"
bash "$ROOT/scripts/build-bricks.sh"

if [ ! -f "$PKG/index.js" ] || [ ! -f "$PKG/index.d.ts" ] || [ ! -f "$PKG/pkg/raa_js_bg.wasm" ]; then
  echo "build-virtual-geometry.sh: $PKG is incomplete after build-bricks.sh" >&2
  exit 1
fi
for file in index.html main.js nmc.gpu.json; do
  if [ ! -f "$SRC/$file" ]; then
    echo "build-virtual-geometry.sh: examples/virtual-geometry/$file is missing" >&2
    exit 1
  fi
done

echo "== stage $OUT"
mkdir -p "$OUT" "$OUT/common" "$OUT/lib" "$OUT/scenes" "$OUT/shared"
cp "$SRC/index.html" "$SRC/main.js" "$SRC/nmc.gpu.json" "$OUT/"
cp -R "$PKG" "$OUT/raa"
cp -R web/common/. "$OUT/common/"
cp -R "$SRC/lib/." "$OUT/lib/"
cp "$SRC"/scenes/*.js "$OUT/scenes/"
for file in capture.js measure.js readiness.js evidence.mjs; do
  cp "$ROOT/examples/sky-weather/lib/$file" "$OUT/shared/$file"
done

# The staged copies resolve the sky-weather helpers as shared/ and web/common as common/,
# both beside main.js, lib/ and scenes/ (contract 'vg-page-layout'). A specifier that still
# names the source paths is a failed build: the browser page could never load it.
echo "== rewrite staged imports"
"$NODE" --input-type=module - "$OUT" <<'IMPORTS'
import fs from 'node:fs';
import path from 'node:path';
const out = process.argv[2];
const IMPORT = /(?:from\s*|import\s*\(\s*)['"]([^'"]+)['"]|(?:^|\n)\s*import\s+['"]([^'"]+)['"]/g;
function assertPortable(file, text) {
  for (const match of text.matchAll(IMPORT)) {
    const specifier = match[1] ?? match[2];
    if (specifier.includes('sky-weather/') || specifier.includes('web/common/')) {
      throw new Error(`${file}: staged import ${specifier} still names sky-weather/ or web/common/`);
    }
  }
}
const rewrite = (file, pairs) => {
  const text = fs.readFileSync(file, 'utf8');
  let staged = text;
  for (const [from, to] of pairs) staged = staged.split(from).join(to);
  if (staged !== text) fs.writeFileSync(file, staged);
  assertPortable(file, staged);
};
rewrite(path.join(out, 'main.js'), [
  ['../sky-weather/lib/', './shared/'], ['../../web/common/', './common/'],
]);
for (const dir of ['lib', 'scenes']) {
  const root = path.join(out, dir);
  for (const name of fs.readdirSync(root).filter((f) => f.endsWith('.js') || f.endsWith('.mjs'))) {
    rewrite(path.join(root, name), [
      ['../../sky-weather/lib/', '../shared/'], ['../../../web/common/', '../common/'],
    ]);
  }
}
IMPORTS

echo "== $OUT/scenes.json"
"$NODE" --input-type=module - "$OUT" <<'SCENES'
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const out = process.argv[2];
const stages = ['bake', 'passes'];
const name = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const directory = path.join(out, 'scenes');
const files = fs.readdirSync(directory).filter((file) => file.endsWith('.js')).sort();
const scenes = [];
const ids = new Set();
for (const file of files) {
  const module = await import(pathToFileURL(path.join(directory, file)).href);
  const scene = module?.default;
  if (!scene || typeof scene !== 'object' || Array.isArray(scene)) {
    throw new Error(`examples/virtual-geometry/scenes/${file}: no default export object`);
  }
  if (typeof scene.id !== 'string' || !name.test(scene.id)) {
    throw new Error(`examples/virtual-geometry/scenes/${file}: invalid scene id ${JSON.stringify(scene.id)}`);
  }
  if (!stages.includes(scene.milestone)) {
    throw new Error(`examples/virtual-geometry/scenes/${file}: milestone ${JSON.stringify(scene.milestone)} is not one of ${stages.join(', ')}`);
  }
  if (ids.has(scene.id)) throw new Error(`examples/virtual-geometry/scenes/${file}: duplicate scene id ${scene.id}`);
  ids.add(scene.id);
  // Metadata only: the page contract's builders and observations stay untouched at build time.
  scenes.push({ id: scene.id, milestone: scene.milestone, file: `scenes/${file}` });
}
if (scenes.length === 0) throw new Error('examples/virtual-geometry/scenes has no delivered scene module');
scenes.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
fs.writeFileSync(path.join(out, 'scenes.json'), `${JSON.stringify(scenes, null, 2)}\n`);
console.log(`${scenes.length} scene(s): ${scenes.map((scene) => `${scene.id} (${scene.milestone})`).join(', ')}`);
SCENES

echo "== $OUT/build.json (ready BuildReceipt, written last)"
"$NODE" --input-type=module - "$ROOT" "$OUT" <<'RECEIPT'
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const [root, out] = process.argv.slice(2);
const { captureInputs, containedPath, writeJsonRecord } = await import(
  pathToFileURL(path.join(root, 'examples/sky-weather/lib/evidence.mjs')).href);
function walk(directory, prefix = '') {
  const names = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) names.push(...walk(path.join(directory, entry.name), name));
    else if (entry.isFile()) { if (name !== 'build.json') names.push(name); }
    else throw new Error(`staged output ${name} is not a regular file`);
  }
  return names;
}
const preparationId = randomUUID();
const inputs = captureInputs(root, preparationId);
const files = walk(out).sort();
if (!files.includes('scenes.json')) throw new Error('scenes.json was not staged');
const receipt = {
  buildId: randomUUID(), preparationId, inputs, builtAt: new Date().toISOString(),
  page: path.resolve(out), files, status: 'ready', error: null,
};
writeJsonRecord(containedPath(out, 'build.json'), receipt);
console.log(`build ${receipt.buildId}: ${files.length} staged file(s), preparation ${preparationId}`);
RECEIPT

echo "build-virtual-geometry.sh: ok ($OUT/scenes.json, $OUT/build.json)"
