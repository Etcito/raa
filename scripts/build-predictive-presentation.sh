#!/usr/bin/env bash
# Stage the opt-in Windows browser fixture using a Kong environment and the engine brick.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <engine checkout> <ege.wasm>" >&2
  exit 2
fi
ENGINE="$(realpath "$1")"
EGE_WASM="$(realpath "$2")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/dist/predictive-presentation"
mkdir -p "$OUT/raa" "$OUT/ege"
rm -f "$OUT/build.json"

"${CARGO:-cargo}" build --locked -p raa-js --target wasm32-unknown-unknown --profile wasm-release
TARGET="$("${CARGO:-cargo}" metadata --no-deps --format-version 1 | node -p 'JSON.parse(require("node:fs").readFileSync(0)).target_directory')"
wasm-bindgen --target web --out-name raa --out-dir "$OUT/raa" "$TARGET/wasm32-unknown-unknown/wasm-release/raa_js.wasm"
node node_modules/typescript/bin/tsc --module nodenext --target es2022 --noCheck \
  --rewriteRelativeImportExtensions --rootDir "$ENGINE/packages/game-engine/src" \
  --outDir "$OUT/ege" "$ENGINE/packages/game-engine/src/loader.ts" "$ENGINE/packages/game-engine/src/rollback.ts"
cp "$EGE_WASM" "$OUT/ege.wasm"
cp "$ENGINE/examples/core/scenes/falling.scene.json" "$OUT/scene.json"
cp "$ENGINE/examples/core/scenes/demo.schema.json" "$OUT/schema.json"
cp examples/predictive-presentation/{index.html,main.js,nmc.gpu.json} "$OUT/"
node --input-type=module - "$OUT" "$(git rev-parse HEAD)" "$(git -C "$ENGINE" rev-parse HEAD)" <<'JS'
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const [root, revision, engineRevision] = process.argv.slice(2);
const files = readdirSync(root, { recursive: true, withFileTypes: true })
  .filter(entry => entry.isFile()).map(entry => join(entry.parentPath.slice(root.length), entry.name)
    .replace(/^[/\\]/, '').replaceAll('\\', '/')).sort();
const hash = createHash('sha256');
for (const name of files) hash.update(name).update('\0').update(readFileSync(join(root, name))).update('\0');
writeFileSync(join(root, 'build.json'), JSON.stringify({
  status: 'ready', buildId: hash.digest('hex'), files, inputs: { revision, engineRevision },
}));
JS
