// M7 (chain milestone 1) meshes + HDR base proof. --quick and --full are identical: no timing or
// size threshold (P2), the captures are the committed records of the GPU client card.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { apiDocCheck, cargoTest, lookdevCheck, wasmCheck } from './lib/p2.mjs';
import { check, run } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TIMEOUT_MS = 30 * 60 * 1000;

function tail(text, lines = 12) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

// v0 coverage of packages/raa/index.d.ts in docs/API.md (port of the api-doc check of
// proofs/m6-api.mjs, which runs its checks at import time and so cannot be imported).
function v0ApiDoc() {
  const dts = readFileSync(join(ROOT, 'packages/raa/index.d.ts'), 'utf8');
  const names = new Set();
  for (const m of dts.matchAll(/^export (?:declare )?(?:interface|type|function|const|class) (\w+)/gm)) names.add(m[1]);
  const engine = /^export (?:declare )?interface Engine\b[^{]*\{([\s\S]*?)^\}/m.exec(dts);
  if (!engine) throw new Error('interface Engine not found in packages/raa/index.d.ts');
  for (const m of engine[1].matchAll(/^ {2}(\w+)\??\s*[(<]/gm)) names.add(m[1]);
  const docPath = join(ROOT, 'docs/API.md');
  if (!existsSync(docPath)) throw new Error('missing docs/API.md');
  const doc = readFileSync(docPath, 'utf8');
  const missing = [...names].filter((n) => !new RegExp('`' + n + '(?:`|\\()').test(doc)).sort();
  return { covered: names.size - missing.length, total: names.size, missing };
}

await check('unit', () =>
  cargoTest(['raa-mesh', 'raa-shader', 'raa-pipeline', 'raa-lit', 'raa-tonemap', 'raa-scene', 'raa-js']),
);

await check('wasm-check', () => wasmCheck());

await check('api-doc', () => {
  const m07 = apiDocCheck('m07');
  const v0 = v0ApiDoc();
  const v0Ok = v0.total > 0 && v0.missing.length === 0;
  const v0Text = `v0 ${v0.covered}/${v0.total}${v0.missing.length ? `, missing: ${v0.missing.join(',')}` : ''}`;
  return {
    measured: `m07 ${m07.measured}; ${v0Text}`,
    threshold: `${m07.threshold}; v0 index.d.ts covered == total in docs/API.md`,
    ok: m07.ok && v0Ok,
  };
});

await check('captures', () => lookdevCheck('m07'));

await check('v0-mapping', () => {
  const t = run(process.env.CARGO || 'cargo', ['test', '-p', 'raa-js', '--test', 'convert'], {
    cwd: ROOT,
    timeout: TIMEOUT_MS,
  });
  let passed = 0;
  let failed = 0;
  for (const m of t.stdout.matchAll(/test result: \w+\. (\d+) passed; (\d+) failed/g)) {
    passed += Number(m[1]);
    failed += Number(m[2]);
  }
  const measured = `exit ${t.code}, ${passed} passed, ${failed} failed (raa-js --test convert)`;
  return {
    measured: t.code === 0 ? measured : `${measured}: ${tail(t.stderr + t.stdout)}`,
    threshold: 'exit 0, 0 failed, > 0 passed',
    ok: t.code === 0 && failed === 0 && passed > 0,
  };
});
