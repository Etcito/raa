#!/usr/bin/env bash
# Stage the package, shared transport/runner and every scene, including later M3 modules.
set -euo pipefail
if [ $# -gt 0 ]; then
  echo "usage: scripts/build-lights.sh" >&2
  exit 2
fi
NODE="${NODE:-node}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/build-package.sh

OUT="$ROOT/dist/lights"
rm -rf "$OUT"
mkdir -p "$OUT/lib" "$OUT/common" "$OUT/scenes"
cp examples/lights/index.html examples/lights/main.js examples/lights/nmc.gpu.json "$OUT/"
cp -R dist/package/raa "$OUT/raa"
cp -R web/common/. "$OUT/common/"
cp examples/lookdev/lib/{geometry,objects,runner}.js "$OUT/lib/"
cp -R examples/lights/scenes/. "$OUT/scenes/"
"$NODE" --input-type=module - "$OUT" <<'JS'
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const out = process.argv[2];
const scenes = [];
for (const file of fs.readdirSync(path.join(out, 'scenes')).filter((f) => f.endsWith('.js')).sort()) {
  const scene = (await import(pathToFileURL(path.join(out, 'scenes', file)).href)).default;
  if (!scene || scene.id !== file.slice(0, -3) || !/^m\d\d$/.test(scene.milestone)) {
    throw new Error(`invalid scene metadata: ${file}`);
  }
  scenes.push({ id: scene.id, milestone: scene.milestone, file: `scenes/${file}` });
}
if (!scenes.length) throw new Error('no lights scenes');
fs.writeFileSync(path.join(out, 'scenes.json'), JSON.stringify(scenes, null, 2) + '\n');
JS
echo "build-lights.sh: ok"
