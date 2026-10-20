// M2 scene proof. The committed timing result is deliberately checked first and alone.
import { accessSync, constants, existsSync, statSync } from 'node:fs';
import { delimiter, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, readJson, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RESULT_REL = 'proofs/results/m2-scene.json';
// Cold release and wasm builds of the whole scene stack fit well inside 30 minutes.
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

await check('cpu-ms-1000', () => {
  const max = VALUES.CPU_MS_MEDIAN_MAX;
  const instances = VALUES.ALLOC_RUN.instances;
  const threshold = `<= ${max.toFixed(1)} ms median, hardware true, instances ${instances}`;
  const path = join(ROOT, RESULT_REL);
  if (!existsSync(path)) return { measured: `missing ${RESULT_REL}`, threshold, ok: false };
  const r = readJson(path);
  const ad = r.adapter && typeof r.adapter === 'object' ? r.adapter : {};
  const adapterName = [ad.device, ad.description].filter((s) => typeof s === 'string' && s.trim() !== '').join(' / ');
  const hwWord = r.hardware === true ? 'hardware' : 'software';
  const ok =
    r.hardware === true &&
    r.instances === instances &&
    typeof r.cpuMsMedian === 'number' &&
    Number.isFinite(r.cpuMsMedian) &&
    r.cpuMsMedian <= max;
  return {
    measured: `cpuMsMedian=${r.cpuMsMedian} ms (p95 ${r.cpuMsP95}), ${hwWord} (${adapterName || 'adapter unnamed'}), instances ${r.instances}`,
    threshold,
    ok,
  };
});

await check('browser-run', () => {
  const result = browserResultCheck('m2-scene', {});
  const path = join(ROOT, RESULT_REL);
  let resizeOk = false;
  if (existsSync(path)) {
    try {
      const r = readJson(path);
      const resize = r.resize && typeof r.resize === 'object' ? r.resize : null;
      resizeOk = resize !== null && resize.ok === true;
      const to = resize && Array.isArray(resize.to) ? resize.to.join('x') : 'none';
      const back = resize && Array.isArray(resize.back) ? resize.back.join('x') : 'none';
      result.measured += `, resize ok=${resize ? resize.ok : null} to=${to} back=${back}`;
      result.measured += `, allocsPerFrameTotal=${r.allocsPerFrameTotal} (information)`;
    } catch {
      // browserResultCheck already reports the malformed committed record.
    }
  }
  return { measured: result.measured, threshold: `${result.threshold}, resize ok`, ok: result.ok && resizeOk };
});

await check('frame-allocs', () => {
  const { instances, warmup, frames, allowed } = VALUES.ALLOC_RUN;
  const threshold = `allocations == ${allowed} over ${frames} frames after ${warmup} warm-up, ${instances} instances`;
  const args = ['run', '--release', '-p', 'raa-bench', '--bin', 'frame-allocs', '--'];
  args.push('--instances', String(instances), '--warmup', String(warmup), '--frames', String(frames));
  const r = run('cargo', args, { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
  const line = r.stdout.split(/\r?\n/).filter((l) => l.startsWith('{')).at(-1);
  if (r.code !== 0 || line === undefined) {
    return { measured: `exit ${r.code}, no JSON line: ${tail(r.stderr)}`, threshold, ok: false };
  }
  const j = JSON.parse(line);
  const ok = j.instances === instances && j.warmup === warmup && j.frames === frames && j.allocations === allowed;
  return {
    measured: `allocations=${j.allocations}, bytes=${j.bytes}, prepCpuMsMedian=${j.prepCpuMsMedian} ms (instances ${j.instances}, warmup ${j.warmup}, frames ${j.frames})`,
    threshold,
    ok,
  };
});

await check('wasm-size', () => {
  const threshold = `<= ${VALUES.WASM_GZIP_MAX}`;
  const b = run('bash', ['scripts/build-web.sh', 'demos/m2-scene'], { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
  if (b.code !== 0) return { measured: `build-web.sh exit ${b.code}: ${tail(b.stderr)}`, threshold, ok: false };
  const sizesPath = join(ROOT, 'dist/m2-scene/sizes.json');
  const sizes = readJson(sizesPath);
  const gzip = gzipSize(resolve(dirname(sizesPath), sizes.wasm));
  return {
    measured: `gzip ${gzip} B (raw ${sizes.raw} B)`,
    threshold,
    ok: Number.isFinite(gzip) && gzip <= VALUES.WASM_GZIP_MAX,
  };
});

await check('scene-tests', () => {
  const crates = ['raa-math', 'raa-geom', 'raa-material', 'raa-scene', 'raa-render'];
  const r = run('cargo', ['test', ...crates.flatMap((c) => ['-p', c])], { cwd: ROOT, env: ENV, timeout: BUILD_TIMEOUT_MS });
  let passed = 0;
  let failed = 0;
  for (const m of r.stdout.matchAll(/test result: \w+\. (\d+) passed; (\d+) failed/g)) {
    passed += Number(m[1]);
    failed += Number(m[2]);
  }
  const skips = r.stderr.split(/\r?\n/).filter((l) => l.startsWith('SKIP ')).length;
  const measured = `exit ${r.code}, ${passed} passed; ${failed} failed, ${skips} SKIP lines (shown, not judged)`;
  return { measured: r.code === 0 ? measured : `${measured}: ${tail(r.stderr, 8)}`, threshold: 'exit 0', ok: r.code === 0 };
});
