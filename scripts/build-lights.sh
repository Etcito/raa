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
# Bound package compilation as well as transport-side CPU work. No invented host
# identity variables: Git is the source commit, checked against the host revision.
CARGO_BUILD_JOBS="$("$NODE" -e 'const n=Number(process.env.NMC_CPUS ?? 1); if(!Number.isInteger(n)||n<1) throw Error("NMC_CPUS must be a positive integer"); console.log(n)')"
export CARGO_BUILD_JOBS
LIGHTS_COMMIT="$(git rev-parse --verify HEAD)"
if [ -n "${NMC_GPU_REVISION:-}" ] && [ "$NMC_GPU_REVISION" != "$LIGHTS_COMMIT" ]; then
  echo "build-lights.sh: host NMC_GPU_REVISION differs from worktree commit" >&2
  exit 1
fi
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
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';

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
const commit = execFileSync('git', ['rev-parse', '--verify', 'HEAD'], { encoding: 'utf8' }).trim();
if (!/^[0-9a-f]{40,64}$/.test(commit)) throw new Error('missing Git commit provenance');
if (process.env.NMC_GPU_REVISION && process.env.NMC_GPU_REVISION !== commit) throw new Error('host revision changed during build');
// A build UUID identifies this actual staging invocation, including uncommitted
// sources. It is not a content hash, a commit replacement or an assumed env var.
fs.writeFileSync(path.join(out, 'build.json'), JSON.stringify({
  buildId: randomUUID(), commit, builtAt: new Date().toISOString(),
  source: 'scripts/build-lights.sh', worktree: process.cwd(),
  dirty: execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8' }).trim().length > 0,
}, null, 2) + '\n');
JS
echo "build-lights.sh: ok"
