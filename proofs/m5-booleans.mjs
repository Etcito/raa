// M5 simple booleans proof. --quick and --full are identical: every case is cheap.
// The csg-cases example fans its cases out over NMC_CPUS threads itself.
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, readJson, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BIN = join(ROOT, 'target/release/examples/csg-cases');
const TMP = join(ROOT, 'target/proofs');
const TIMEOUT_MS = 30 * 60 * 1000;
const OPTS = { cwd: ROOT, timeout: TIMEOUT_MS };
const IDENTITY_MAX = 1e-4;

function tail(text, lines = 12) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

// Each artefact is built at most once per run, lazily, so a skipped check builds nothing.
let cases = null;
function csgCases() {
  if (!cases) {
    const b = run('cargo', ['build', '--release', '-p', 'raa-csg', '--example', 'csg-cases'], OPTS);
    if (b.code !== 0) throw new Error(`cargo build csg-cases exit ${b.code}: ${tail(b.stderr)}`);
    mkdirSync(TMP, { recursive: true });
    const out = join(TMP, 'm5-csg-cases.json');
    const r = run(BIN, ['--json', out], OPTS);
    if (!existsSync(out)) throw new Error(`csg-cases exit ${r.code}, no JSON: ${tail(r.stderr + r.stdout)}`);
    cases = { code: r.code, rows: readJson(out), detail: tail(r.stderr + r.stdout) };
  }
  return cases;
}

let webBuilt = false;
function webSizes() {
  if (!webBuilt) {
    const b = run(join(ROOT, 'scripts/build-web.sh'), ['demos/m5-booleans'], OPTS);
    if (b.code !== 0) throw new Error(`build-web.sh demos/m5-booleans exit ${b.code}: ${tail(b.stderr + b.stdout)}`);
    webBuilt = true;
  }
  return readJson(join(ROOT, 'dist/m5-booleans/sizes.json'));
}

// Timing thresholds come first and alone.
await check('browser-run', () => {
  const r = browserResultCheck('m5-booleans', { requireHardware: false });
  const j = readJson(join(ROOT, 'proofs/results/m5-booleans.json'));
  return {
    measured: `${r.measured}, cpuMsMedian=${Number(j.cpuMsMedian).toFixed(2)} ms/frame`,
    threshold: `firstFrameMs <= ${VALUES.FIRST_FRAME_MAX_MS}, nonEmpty`,
    ok: r.ok,
  };
});

await check('csg-supported', () => {
  const { code, rows } = csgCases();
  const ok = rows.filter((c) => c.expected === 'ok');
  const good = ok.filter(
    (c) =>
      c.status === 'ok' &&
      c.watertight === true &&
      typeof c.volumeIdentityRelErr === 'number' &&
      c.volumeIdentityRelErr <= IDENTITY_MAX,
  );
  const per = ok.map((c) => `${c.id} tris=${c.triangles} err=${c.volumeIdentityRelErr}`);
  return {
    measured: `${good.length}/${ok.length}; ${per.join('; ')}`,
    threshold: `watertight, volume identity rel err <= ${IDENTITY_MAX}`,
    ok: code === 0 && ok.length > 0 && good.length === ok.length,
  };
});

await check('csg-refusals', () => {
  const { code, rows } = csgCases();
  const ref = rows.filter((c) => c.expected !== 'ok');
  const good = ref.filter((c) => c.code === c.expected && typeof c.message === 'string' && c.message !== '');
  return {
    measured: `${good.length}/${ref.length}; ${ref.map((c) => `${c.id} ${c.code}: ${c.message}`).join('; ')}`,
    threshold: 'code === expected, non-empty message',
    ok: code === 0 && ref.length > 0 && good.length === ref.length,
  };
});

await check('loader-integration', () => {
  const t = run('cargo', ['test', '-p', 'raa-csg', '--test', 'loader'], OPTS);
  const m = /test result: [^\n]*/.exec(t.stdout);
  return {
    measured: t.code === 0 ? (m ? m[0] : 'ok') : `cargo test exit ${t.code}: ${tail(t.stderr + t.stdout)}`,
    threshold: 'exit 0',
    ok: t.code === 0,
  };
});

await check('wasm-size', () => {
  const sizes = webSizes();
  const gz = gzipSize(join(ROOT, 'dist/m5-booleans', sizes.wasm));
  return {
    measured: `raw ${sizes.raw}, gzip ${gz}`,
    threshold: String(VALUES.WASM_GZIP_MAX),
    ok: gz <= VALUES.WASM_GZIP_MAX,
  };
});
