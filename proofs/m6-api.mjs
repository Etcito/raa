// M6 extensibility and public API proof. --quick and --full are identical: every check is cheap.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, readJson, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TMP = join(ROOT, 'target/proofs');
const TIMEOUT_MS = 30 * 60 * 1000;
const OPTS = { cwd: ROOT, timeout: TIMEOUT_MS };
const TGZ = join(ROOT, 'dist/etcito-raa-0.1.0.tgz');

function tail(text, lines = 12) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

// Each artefact is built at most once per run, lazily, so a skipped check builds nothing.
let packageBuilt = false;
function buildPackage() {
  if (!packageBuilt) {
    const b = run('bash', [join(ROOT, 'scripts/build-package.sh')], OPTS);
    if (b.code !== 0) throw new Error(`build-package.sh exit ${b.code}: ${tail(b.stderr + b.stdout)}`);
    packageBuilt = true;
  }
}

let demosBuilt = false;
function buildDemos() {
  if (!demosBuilt) {
    const dirs = readdirSync(join(ROOT, 'demos'))
      .filter((d) => statSync(join(ROOT, 'demos', d)).isDirectory())
      .sort();
    for (const d of dirs) {
      const b = run('bash', [join(ROOT, 'scripts/build-web.sh'), `demos/${d}`], OPTS);
      if (b.code !== 0) throw new Error(`build-web.sh demos/${d} exit ${b.code}: ${tail(b.stderr + b.stdout)}`);
    }
    demosBuilt = true;
  }
}

// Timing thresholds come first and alone.
await check('browser-run', () => {
  const r = browserResultCheck('example-basic', { requireHardware: false });
  const j = readJson(join(ROOT, 'proofs/results/example-basic.json'));
  const hw = j.hardware === true ? 'hardware' : j.hardware === false ? 'software' : 'hardware-unstated';
  return {
    measured: `${r.measured}, cpuMsMedian=${Number(j.cpuMsMedian).toFixed(2)} ms/frame (${hw})`,
    threshold: `firstFrameMs <= ${VALUES.FIRST_FRAME_MAX_MS}, nonEmpty`,
    ok: r.ok,
  };
});

await check('extension', () => {
  const t = run('cargo', ['test', '-p', 'raa-ext-test'], OPTS);
  if (t.code !== 0) throw new Error(`cargo test -p raa-ext-test exit ${t.code}: ${tail(t.stderr + t.stdout)}`);
  // The integration test binary's summary is the meaningful line; fall back to the last one.
  const summaries = t.stdout.match(/test result: [^\n]*/g) ?? [];
  const summary = summaries.find((s) => !/ 0 passed/.test(s)) ?? summaries.at(-1) ?? 'ok';
  const m = run('cargo', ['metadata', '--format-version', '1', '--no-deps'], OPTS);
  if (m.code !== 0) throw new Error(`cargo metadata exit ${m.code}: ${tail(m.stderr)}`);
  const pkgs = JSON.parse(m.stdout).packages;
  const ext = pkgs.find((p) => p.name === 'raa-ext-test');
  if (!ext) throw new Error('package raa-ext-test not in cargo metadata');
  const demoNames = new Set(
    pkgs.filter((p) => relative(ROOT, p.manifest_path).split(/[\\/]/)[0] === 'demos').map((p) => p.name),
  );
  const deps = [...new Set(ext.dependencies.map((d) => d.name))].sort();
  const bad = deps.filter((n) => !n.startsWith('raa-') || n === 'raa-js' || demoNames.has(n));
  const dependents = pkgs.filter((p) => p.dependencies.some((d) => d.name === 'raa-ext-test')).map((p) => p.name);
  const parts = [summary, `deps=${deps.join(',')}`, `dependents=${dependents.length}`];
  if (bad.length) parts.push(`not allowed: ${bad.join(',')}`);
  if (dependents.length) parts.push(`dependents: ${dependents.join(',')}`);
  return {
    measured: parts.join(', '),
    threshold: 'exit 0, deps only raa-* libraries (not raa-js, not demos), dependents=0',
    ok: bad.length === 0 && dependents.length === 0,
  };
});

await check('api-doc', () => {
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
  const covered = names.size - missing.length;
  return {
    measured: `${covered}/${names.size}${missing.length ? `, missing: ${missing.join(',')}` : ''}`,
    threshold: 'covered == total',
    ok: names.size > 0 && missing.length === 0,
  };
});

await check('npm-pack', () => {
  buildPackage();
  if (!existsSync(TGZ)) throw new Error(`missing ${relative(ROOT, TGZ)}`);
  const t = run('tar', ['-tzf', TGZ], OPTS);
  if (t.code !== 0) throw new Error(`tar -tzf exit ${t.code}: ${tail(t.stderr)}`);
  const entries = t.stdout.split(/\r?\n/).filter(Boolean);
  const want = [
    ['package/package.json', (e) => e === 'package/package.json'],
    ['package/index.js', (e) => e === 'package/index.js'],
    ['package/index.d.ts', (e) => e === 'package/index.d.ts'],
    ['package/README.md', (e) => e === 'package/README.md'],
    ['package/pkg/*_bg.wasm', (e) => /^package\/pkg\/[^/]+_bg\.wasm$/.test(e)],
    ['package/pkg/*.js', (e) => /^package\/pkg\/[^/]+\.js$/.test(e)],
  ];
  const missing = want.filter(([, f]) => !entries.some(f)).map(([n]) => n);
  return {
    measured: `${relative(ROOT, TGZ)}: ${entries.length} entries${missing.length ? `, missing: ${missing.join(',')}` : ', all 6 required present'}`,
    threshold: 'exit 0, package.json index.js index.d.ts README.md pkg/*_bg.wasm pkg/*.js',
    ok: missing.length === 0,
  };
});

await check('wasm-size-table', () => {
  buildPackage();
  buildDemos();
  const sources = readdirSync(join(ROOT, 'dist'))
    .filter((d) => existsSync(join(ROOT, 'dist', d, 'sizes.json')))
    .map((d) => ({ name: d, base: join(ROOT, 'dist', d), sizes: join(ROOT, 'dist', d, 'sizes.json') }));
  sources.push({ name: 'raa', base: join(ROOT, 'packages/raa'), sizes: join(ROOT, 'dist/raa-sizes.json') });
  const rows = sources
    .map(({ name, base, sizes }) => {
      const s = readJson(sizes);
      const path = join(base, s.wasm);
      const bytes = readFileSync(path);
      return {
        name,
        wasm: relative(ROOT, path),
        raw: bytes.length,
        gzip: gzipSize(path),
        sha256: createHash('sha256').update(bytes).digest('hex'),
      };
    })
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const w = Math.max(4, ...rows.map((r) => r.name.length));
  console.log(`${'name'.padEnd(w)}  ${'raw'.padStart(9)}  ${'gzip'.padStart(9)}  ok`);
  for (const r of rows) {
    const ok = r.gzip <= VALUES.WASM_GZIP_MAX ? 'yes' : 'NO';
    console.log(`${r.name.padEnd(w)}  ${String(r.raw).padStart(9)}  ${String(r.gzip).padStart(9)}  ${ok}`);
  }
  mkdirSync(TMP, { recursive: true });
  writeFileSync(join(TMP, 'm6-sizes.json'), `${JSON.stringify(rows, null, 2)}\n`);
  const max = rows.reduce((a, r) => (r.gzip > a.gzip ? r : a), rows[0]);
  return {
    measured: `${rows.length} wasm, max gzip ${max.gzip} (${max.name})`,
    threshold: `every gzip <= ${VALUES.WASM_GZIP_MAX}`,
    ok: rows.length > 0 && rows.every((r) => r.gzip <= VALUES.WASM_GZIP_MAX),
  };
});
