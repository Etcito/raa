// M3 primitives proof. The committed timing result is deliberately checked first and alone.
import { accessSync, constants, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { delimiter, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, parseArgs, pool, readJson, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ARGS = parseArgs(process.argv.slice(2));
const FIXTURE_REL = 'crates/raa-prims/tests/fixtures/sdf-leaf-reference.json';
const REPORT_BIN = join(ROOT, 'target/release/examples/prims-report');
const TABLE_REL = 'target/proofs/m3-prims-table.json';
// PrimKind::ALL order (S3 names).
const KINDS = ['sphere', 'box', 'roundBox', 'capsule', 'cylinder', 'cone', 'torus', 'ellipsoid'];
const QUALITIES = [
  ['LOW', VALUES.QUALITY.LOW],
  ['LOD0', VALUES.QUALITY.LOD0],
  ['HIGH', VALUES.QUALITY.HIGH],
];
// Cold release, test and wasm builds of the prims stack fit well inside 30 minutes.
const BUILD_TIMEOUT_MS = 30 * 60 * 1000;

// Cargo must use the PATH rustc (m1 dogfood item 1). An explicit RUSTC wins.
function pathRustc() {
  for (const dir of (process.env.PATH ?? '').split(delimiter)) {
    if (!dir) continue;
    const candidate = join(dir, 'rustc');
    try {
      if (!statSync(candidate).isFile()) continue;
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Not present or not executable in this PATH entry.
    }
  }
  return null;
}

function cargoEnv() {
  const env = { ...process.env };
  if (!env.RUSTC) {
    const rustc = pathRustc();
    if (rustc) env.RUSTC = rustc;
  }
  return env;
}

const ENV = cargoEnv();

function tail(text, lines = 15) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

// `--only` is forwarded to the kind names. A pattern that matches no kind (for example a
// check label such as `sdf-reference`) selects every kind.
function selectedKinds() {
  if (!ARGS.only) return KINDS;
  const picked = KINDS.filter((k) => ARGS.only.test(k));
  return picked.length > 0 ? picked : KINDS;
}

// Builds prims-report once and runs it per kind through `pool`; shared by two checks.
let report = null;
async function primsReport() {
  if (report) return report;
  const b = run('cargo', ['build', '--release', '-p', 'raa-prims', '--example', 'prims-report'], {
    cwd: ROOT,
    env: ENV,
    timeout: BUILD_TIMEOUT_MS,
  });
  if (b.code !== 0) throw new Error(`cargo build prims-report exit ${b.code}: ${tail(b.stderr)}`);
  mkdirSync(join(ROOT, 'target/proofs'), { recursive: true });
  const kinds = selectedKinds();
  const perKind = await pool(kinds, async (kind) => {
    const out = join(ROOT, `target/proofs/m3-prims-${kind}.json`);
    const args = ['--only', `^${kind}$`, '--compare', FIXTURE_REL, '--json', out];
    const r = run(REPORT_BIN, args, { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
    if (!existsSync(out)) throw new Error(`prims-report ${kind} exit ${r.code}, no JSON: ${tail(r.stderr + r.stdout)}`);
    return { kind, code: r.code, stdout: r.stdout, rows: readJson(out) };
  });
  const rows = perKind.flatMap((p) => p.rows);
  writeFileSync(join(ROOT, TABLE_REL), `${JSON.stringify(rows, null, 2)}\n`);
  const fixture = readJson(join(ROOT, FIXTURE_REL));
  report = { kinds, perKind, rows, fixture };
  return report;
}

function failLines(perKind) {
  return perKind
    .filter((p) => p.code !== 0)
    .flatMap((p) => p.stdout.split(/\r?\n/).filter((l) => /FAIL$|^\s+(ref )?fail:/.test(l)))
    .slice(0, 6)
    .join(' / ');
}

function worst(rows, key) {
  let w = null;
  for (const row of rows) {
    const v = row.ref ? row.ref[key] : null;
    if (typeof v !== 'number') return { text: 'null', ok: false };
    if (w === null || v > w.v) w = { v, at: `${row.kind} ${row.set} q${row.quality}` };
  }
  return w === null ? { text: 'none', ok: false } : { text: `${w.v} (${w.at})`, ok: true };
}

function cell(fixture, row) {
  const c = fixture.cases.find((x) => x.kind === row.kind && x.set === row.set && x.quality === row.quality);
  const counts = c && c.counts && Object.keys(c.counts).length > 0
    ? Object.entries(c.counts).map(([k, v]) => `${k}=${v}`).join(',')
    : '-';
  return `${row.triangles} [${counts}]`;
}

function printTable({ kinds, rows, fixture }) {
  const head = ['kind', ...['A', 'B'].flatMap((s) => QUALITIES.map(([n]) => `${s} ${n}`))];
  const lines = kinds.map((kind) => [
    kind,
    ...['A', 'B'].flatMap((set) =>
      QUALITIES.map(([, q]) => {
        const row = rows.find((r) => r.kind === kind && r.set === set && r.quality === q);
        return row ? cell(fixture, row) : 'missing';
      }),
    ),
  ]);
  const width = head.map((h, i) => Math.max(h.length, ...lines.map((l) => l[i].length)));
  const fmt = (l) => l.map((c, i) => c.padEnd(width[i])).join(' | ');
  console.log('triangle table (triangles [derived counts]) per kind x set x quality:');
  console.log(fmt(head));
  console.log(width.map((w) => '-'.repeat(w)).join('-|-'));
  for (const l of lines) console.log(fmt(l));
}

await check('browser-run', () => {
  const result = browserResultCheck('m3-prims', {});
  const path = join(ROOT, 'proofs/results/m3-prims.json');
  if (existsSync(path)) {
    try {
      const r = readJson(path);
      result.measured += `, cpuMsMedian=${r.cpuMsMedian} ms, p95=${r.cpuMsP95} ms, instances=${r.instances}, triangles=${r.triangles} (information)`;
    } catch {
      // browserResultCheck already reports the malformed committed record.
    }
  }
  return result;
});

await check('prims-closed-normals', () => {
  const r = run('cargo', ['test', '-p', 'raa-prims', '--test', 'closed'], { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
  let passed = 0;
  let failed = 0;
  for (const m of r.stdout.matchAll(/test result: \w+\. (\d+) passed; (\d+) failed/g)) {
    passed += Number(m[1]);
    failed += Number(m[2]);
  }
  const measured = `exit ${r.code}, ${passed} passed; ${failed} failed`;
  return {
    measured: r.code === 0 ? measured : `${measured}: ${tail(r.stdout + r.stderr, 8)}`,
    threshold: 'exit 0, 0 failed',
    ok: r.code === 0 && failed === 0 && passed > 0,
  };
});

await check('prims-deterministic', async () => {
  const { perKind, rows } = await primsReport();
  const det = rows.filter((r) => r.deterministic === true).length;
  const fails = rows.filter((r) => r.deterministic !== true).map((r) => `${r.kind} ${r.set} q${r.quality}`);
  const measured = `${det}/${rows.length} hashes identical across two runs${fails.length ? `; differ: ${fails.join(', ')}` : ''}`;
  return {
    measured: perKind.every((p) => p.code === 0) ? measured : `${measured}; report: ${failLines(perKind)}`,
    threshold: `all (${rows.length}/${rows.length})`,
    ok: rows.length > 0 && det === rows.length,
  };
});

await check('sdf-reference', async () => {
  const { perKind, rows, fixture } = await primsReport();
  const okCount = rows.filter((r) => r.ref && r.ref.ok === true).length;
  const bb = worst(rows, 'bboxMaxAbsErr');
  const ar = worst(rows, 'areaRelErr');
  const vr = worst(rows, 'volumeRelErr');
  const commit = fixture.mapping?.reference?.commit ?? 'unknown';
  let measured = `${okCount}/${rows.length} within SDF_REF_TOL; worst bboxMaxAbsErr=${bb.text}, areaRelErr=${ar.text}, volumeRelErr=${vr.text}; sdfm commit ${commit}`;
  if (okCount !== rows.length) measured += `; report: ${failLines(perKind)}`;
  return {
    measured,
    threshold: `all (${rows.length}/${rows.length}); uniquePositions and triangles equal, bbox <= 1e-5 x max(1, diag), area/volume rel <= 1e-4`,
    ok: rows.length > 0 && okCount === rows.length && bb.ok && ar.ok && vr.ok,
  };
});

if (report) printTable(report);

await check('wasm-size', () => {
  const threshold = `<= ${VALUES.WASM_GZIP_MAX}`;
  const b = run('bash', ['scripts/build-web.sh', 'demos/m3-prims'], { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
  if (b.code !== 0) return { measured: `build-web.sh exit ${b.code}: ${tail(b.stderr)}`, threshold, ok: false };
  const sizesPath = join(ROOT, 'dist/m3-prims/sizes.json');
  const sizes = readJson(sizesPath);
  const gzip = gzipSize(resolve(dirname(sizesPath), sizes.wasm));
  return {
    measured: `gzip ${gzip} B (raw ${sizes.raw} B)`,
    threshold,
    ok: Number.isFinite(gzip) && gzip <= VALUES.WASM_GZIP_MAX,
  };
});
