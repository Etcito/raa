// M1 boot proof. The committed browser result is deliberately checked first.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { Agent, get } from 'node:http';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, readJson, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NODE = process.env.NODE || '/root/nmc/engines/bin/node';

await check('browser-run', () => {
  const result = browserResultCheck('m1-boot', {});
  const path = join(ROOT, 'proofs/results/m1-boot.json');
  if (existsSync(path)) {
    try {
      const browser = readJson(path);
      result.measured += `, cpuMsMedian=${browser.cpuMsMedian} ms/frame (information only)`;
    } catch {
      // browserResultCheck already reports the malformed committed record.
    }
  }
  return result;
});

await check('check-sh', async () => {
  const child = spawn('bash', ['scripts/check.sh'], {
    cwd: ROOT,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  const stream = (chunk, isError) => {
    const value = chunk.toString();
    if (isError) stderr += value;
    else stdout += value;
    (isError ? process.stderr : process.stdout).write(value);
  };
  child.stdout.on('data', (chunk) => stream(chunk, false));
  child.stderr.on('data', (chunk) => stream(chunk, true));
  const code = await new Promise((resolveCode, reject) => {
    child.once('error', reject);
    child.once('close', (status) => resolveCode(status ?? 128));
  });
  const steps = `${stdout}\n${stderr}`.split(/\r?\n/).filter((line) => line.startsWith('== '));
  const lastStep = steps.at(-1) ?? '(no step line)';
  return { measured: `exit ${code}, ${lastStep}`, threshold: 'exit 0', ok: code === 0 };
});

await check('wasm-size', () => {
  const sizesPath = join(ROOT, 'dist/m1-boot/sizes.json');
  const sizes = readJson(sizesPath);
  const wasmPath = resolve(dirname(sizesPath), sizes.wasm);
  const raw = sizes.raw;
  const gzip = gzipSize(wasmPath);
  return {
    measured: `gzip ${gzip} B (raw ${raw} B)`,
    threshold: `<= ${VALUES.WASM_GZIP_MAX}`,
    ok: Number.isFinite(gzip) && gzip <= VALUES.WASM_GZIP_MAX,
  };
});

// GET from the local server with a plain agent: with NODE_USE_ENV_PROXY set, fetch() and the
// global http agent route even 127.0.0.1 through the env proxy unless NO_PROXY lists it.
function getLocal(port, path) {
  return new Promise((resolveGet, rejectGet) => {
    const request = get({ host: '127.0.0.1', port, path, agent: new Agent() }, (response) => {
      response.resume();
      response.once('end', () => resolveGet({ status: response.statusCode, headers: response.headers }));
    });
    request.once('error', rejectGet);
    request.setTimeout(10000, () => request.destroy(new Error(`GET ${path} timed out`)));
  });
}

await check('serve', async () => {
  const child = spawn(NODE, ['web/serve.mjs', 'dist/m1-boot', '--port', '0'], {
    cwd: ROOT,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  let errors = '';
  let portResolve;
  let portReject;
  const portLine = new Promise((resolvePort, rejectPort) => {
    portResolve = resolvePort;
    portReject = rejectPort;
  });
  const collect = (chunk, isError) => {
    const value = chunk.toString();
    if (isError) errors += value;
    else output += value;
    const match = output.match(/serving .* on http:\/\/127\.0\.0\.1:(\d+)\//);
    if (match) portResolve(Number(match[1]));
  };
  child.stdout.on('data', (chunk) => collect(chunk, false));
  child.stderr.on('data', (chunk) => collect(chunk, true));
  child.once('error', portReject);
  child.once('exit', (code) => portReject(new Error(`server exited ${code}: ${errors || output}`)));
  try {
    let timeoutId;
    const timeout = new Promise((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error('server startup timed out')), 10000);
    });
    let port;
    try {
      port = await Promise.race([portLine, timeout]);
    } finally {
      clearTimeout(timeoutId);
    }
    const paths = ['/index.html', '/main.js', '/pkg/m1_boot.js', '/pkg/m1_boot_bg.wasm', '/nmc.gpu.json'];
    const statuses = [];
    for (const path of paths) {
      const response = await getLocal(port, path);
      const contentType = response.headers['content-type'] ?? '';
      statuses.push(`${path}=${response.status}${path.endsWith('.wasm') ? ` ${contentType}` : ''}`);
      if (path.endsWith('.wasm') && !contentType.toLowerCase().startsWith('application/wasm')) {
        return { measured: statuses.join(', '), threshold: 'all 200; wasm application/wasm', ok: false };
      }
      if (response.status !== 200) {
        return { measured: statuses.join(', '), threshold: 'all 200; wasm application/wasm', ok: false };
      }
    }
    return { measured: statuses.join(', '), threshold: 'all 200; wasm application/wasm', ok: true };
  } finally {
    child.kill('SIGTERM');
    await new Promise((resolveClose) => {
      if (child.exitCode !== null || child.signalCode !== null) resolveClose();
      else child.once('exit', resolveClose);
    });
  }
});
