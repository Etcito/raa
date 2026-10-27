#!/usr/bin/env node
// Functional Chrome runner for examples/anim-brick: det-selftest uses local Chrome;
// clip-sample publishes the staged brick through NMC_GPU_CLIENT. Records are
// checked for their exact shape and requested seed/ticks only; no hash or SHA value is
// compared with anything (owner rule: no hash proofs).
import { spawn } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const DET_CASES = Object.freeze([[0, 0], [1, 60], [305419896, 600]]);
const STAGE = join(ROOT, 'dist/anim-brick');
const BRICKS = join(ROOT, 'dist/bricks');
const U32_MAX = 0xffffffff;
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
};

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const isU32 = (value) => Number.isInteger(value) && value >= 0 && value <= U32_MAX;
const label = (adapter) => ['vendor', 'architecture', 'device', 'description']
  .map((key) => adapter?.[key]).filter((value) => typeof value === 'string' && value.trim() !== '')
  .join(' / ');

function stagePage() {
  const anim = join(BRICKS, 'raa-anim');
  if (!existsSync(join(anim, 'index.js')) || !existsSync(join(anim, 'pkg', 'raa_anim_js_bg.wasm'))) {
    throw new Error('missing dist/bricks/raa-anim; run bash scripts/build-bricks.sh first');
  }
  rmSync(STAGE, { recursive: true, force: true });
  cpSync(join(ROOT, 'examples/anim-brick'), STAGE, { recursive: true });
  cpSync(join(ROOT, 'web/common'), join(STAGE, 'common'), { recursive: true });
  cpSync(anim, join(STAGE, 'raa-anim'), { recursive: true });
}

function serve(root) {
  const server = http.createServer((req, res) => {
    const raw = (req.url ?? '/').split('?')[0].split('#')[0];
    let pathname;
    try { pathname = decodeURIComponent(raw); } catch { res.writeHead(400); res.end('bad request\n'); return; }
    if (pathname.includes('\0')) { res.writeHead(400); res.end('bad request\n'); return; }
    let file = resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (file !== root && !file.startsWith(root + sep)) { res.writeHead(403); res.end('forbidden\n'); return; }
    try {
      if (statSync(file).isDirectory()) file = join(file, 'index.html');
      const body = readFileSync(file);
      res.writeHead(200, {
        'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end('not found\n');
    }
  });
  return new Promise((ready, fail) => {
    server.once('error', fail);
    server.listen(0, '127.0.0.1', () => ready({ server, port: server.address().port }));
  });
}

class Pipe {
  constructor(child) {
    this.child = child;
    this.buffer = '';
    this.nextId = 0;
    this.pending = new Map();
    child.stdio[4].on('data', (chunk) => this.receive(chunk));
    child.on('exit', (code, signal) => this.failAll(`Chrome exited (${signal ?? code})`));
  }

  receive(chunk) {
    this.buffer += chunk.toString('utf8');
    for (let end = this.buffer.indexOf('\0'); end >= 0; end = this.buffer.indexOf('\0')) {
      const text = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 1);
      if (!text) continue;
      let message;
      try { message = JSON.parse(text); } catch { continue; }
      const entry = this.pending.get(message.id);
      if (!entry) continue;
      this.pending.delete(message.id);
      if (message.error) entry.fail(new Error(message.error.message));
      else entry.ready(message.result);
    }
  }

  failAll(message) {
    for (const entry of this.pending.values()) entry.fail(new Error(message));
    this.pending.clear();
  }

  send(method, params = {}, sessionId) {
    return new Promise((ready, fail) => {
      const message = { id: ++this.nextId, method, params };
      if (sessionId) message.sessionId = sessionId;
      this.pending.set(message.id, { ready, fail });
      this.child.stdio[3].write(JSON.stringify(message) + '\0');
    });
  }
}

function openChrome() {
  const chrome = process.env.CHROME_PATH ?? process.env.NMC_APP_CHROME;
  if (!chrome || !existsSync(chrome)) throw new Error('CHROME_PATH is unavailable');
  const profile = mkdtempSync(join(tmpdir(), 'raa-anim-chrome-'));
  const child = spawn(chrome, [
    '--no-sandbox', '--disable-dev-shm-usage', '--remote-debugging-pipe',
    `--user-data-dir=${profile}`, 'about:blank',
  ], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
  const exit = new Promise((done) => child.once('exit', done));
  return { child, profile, pipe: new Pipe(child), exit };
}

async function evaluate(pipe, sessionId, expression) {
  const response = await pipe.send('Runtime.evaluate',
    { expression, awaitPromise: true, returnByValue: true }, sessionId);
  if (response.exceptionDetails) {
    const detail = response.exceptionDetails.exception?.description ?? response.exceptionDetails.text;
    throw new Error(`page run failed: ${detail ?? 'unknown error'}`);
  }
  return response.result?.value;
}

async function waitReady(pipe, sessionId, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  let last = 'no response';
  while (Date.now() < deadline) {
    try {
      const state = await evaluate(pipe, sessionId,
        '({ready: !!(window.nmcGpu && window.nmcGpu.ready), status: (document.getElementById("status")||{}).textContent, error: (document.getElementById("error")||{}).textContent})');
      if (state?.error) throw new Error(`page error: ${state.error}`);
      if (state?.ready) return state;
      last = state?.status ?? last;
    } catch (error) {
      if (/page error:/.test(error.message)) throw error;
      last = error.message;
    }
    await sleep(250);
  }
  throw new Error(`page did not become ready: ${last}`);
}

function validateOne(op, seed, ticks, caseName = 'det-selftest') {
  if (!op || typeof op !== 'object' || Array.isArray(op)) throw new Error('browser run returned no object');
  if (op.case !== caseName || op.ticks !== ticks || !Array.isArray(op.seeds)
      || op.seeds.length !== 1 || op.seeds[0] !== seed) {
    throw new Error('browser run inputs differ from the request');
  }
  if (!op.brick || Object.keys(op.brick).sort().join(',') !== 'name,version'
      || op.brick.name !== 'raa-anim' || op.brick.version !== '0.5.0') {
    throw new Error('browser brick metadata is not raa-anim 0.5.0');
  }
  const adapter = op.adapter;
  const fields = ['vendor', 'architecture', 'device', 'description'];
  if (!adapter || !fields.every((key) => typeof adapter[key] === 'string')
      || typeof adapter.isFallbackAdapter !== 'boolean') {
    throw new Error('browser adapter is malformed');
  }
  if (typeof op.hardware !== 'boolean') throw new Error('browser hardware classification is missing');
  if (!Array.isArray(op.results) || op.results.length !== 1) throw new Error('expected one browser record');
  const record = op.results[0];
  if (!record || Object.keys(record).sort().join(',') !== 'case,hash,seed,ticks'
      || record.case !== caseName || record.seed !== seed || record.ticks !== ticks
      || !isU32(record.seed) || !isU32(record.ticks)
      || typeof record.hash !== 'string' || !/^[0-9a-f]{8}$/.test(record.hash)) {
    throw new Error('browser selftest record is malformed');
  }
  return {
    adapter,
    hardware: op.hardware,
    record: { case: record.case, seed: record.seed, ticks: record.ticks, hash: record.hash },
  };
}

export async function captureAnimBrick({ cases = DET_CASES } = {}) {
  stagePage();
  const { server, port } = await serve(STAGE);
  const chrome = openChrome();
  try {
    const target = await chrome.pipe.send('Target.createTarget', {
      url: `http://127.0.0.1:${port}/index.html`,
    });
    const attached = await chrome.pipe.send('Target.attachToTarget',
      { targetId: target.targetId, flatten: true });
    const sessionId = attached.sessionId;
    await waitReady(chrome.pipe, sessionId);
    const runs = [];
    for (const [seed, ticks] of cases) {
      const op = await evaluate(chrome.pipe, sessionId,
        `window.nmcGpu.ops.run({case:"det-selftest",seeds:[${seed}],ticks:${ticks}})`);
      runs.push(validateOne(op, seed, ticks));
    }
    const adapter = runs[0]?.adapter;
    if (!label(adapter)) throw new Error('browser did not name its adapter');
    for (const run of runs) {
      if (JSON.stringify(run.adapter) !== JSON.stringify(adapter) || run.hardware !== runs[0].hardware) {
        throw new Error('browser adapter changed between cases');
      }
    }
    return {
      adapter,
      hardware: runs[0].hardware,
      adapterText: label(adapter),
      records: runs.map((run) => run.record),
    };
  } finally {
    try { await chrome.pipe.send('Browser.close'); } catch { /* already closed */ }
    chrome.child.kill('SIGKILL');
    await Promise.race([chrome.exit, sleep(2000)]);
    server.close();
    rmSync(chrome.profile, { recursive: true, force: true });
  }
}

// Only clip-sample uses the managed client; the det-selftest runner above retains
// its existing local Chrome behavior and output. Each client call is bounded.
function gpuClient(client, args) {
  return new Promise((done, fail) => {
    const child = spawn(process.execPath, [client, ...args], {
      cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 60_000);
    child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
    child.once('error', error => { clearTimeout(timer); fail(error); });
    child.once('close', (code, signal) => {
      clearTimeout(timer);
      if (code === 0) done(stdout);
      else {
        const token = process.env.NMC_GPU_TOKEN;
        const detail = (token ? stderr.replaceAll(token, '[REDACTED]') : stderr).slice(-4000).trim();
        fail(new Error(`GPU client ${args[0]} failed (${signal ?? code}); no record written`
          + (detail ? `\n${detail}` : '')));
      }
    });
  });
}

function pageRef(stdout) {
  for (const text of [stdout, ...stdout.split(/\r?\n/)]) {
    try {
      const doc = JSON.parse(text);
      for (const value of [doc, doc?.result, doc?.data]) {
        const ref = value?.page_ref ?? value?.pageRef;
        if (typeof ref === 'string' && ref.trim()) return ref.trim();
      }
    } catch { /* The client may also print progress lines. */ }
  }
  throw new Error('GPU publish returned no page_ref; no record written');
}

function opResult(doc, accept = value => value.adapter && Array.isArray(value.results)) {
  const queue = [doc];
  for (let i = 0; i < queue.length && i < 32; i++) {
    const value = queue[i];
    if (!value || typeof value !== 'object') continue;
    if (accept(value)) return value;
    for (const key of ['json', 'result', 'value', 'data', 'output', 'opResult', 'op_result', 'run']) {
      if (value[key] && typeof value[key] === 'object') queue.push(value[key]);
    }
  }
  throw new Error('GPU client returned no animation operation result; no record written');
}

export async function captureClipSample() {
  const client = process.env.NMC_GPU_CLIENT;
  if (!client || !existsSync(client)) throw new Error('NMC_GPU_CLIENT is required; no record written');
  const work = mkdtempSync(join(tmpdir(), 'raa-clip-sample-'));
  try {
    // Establish the client protocol with a blank page before loading the brick.
    const probe = join(work, 'probe');
    mkdirSync(probe);
    writeFileSync(join(probe, 'nmc.gpu.json'), JSON.stringify({ page: 'index.html', ops: ['render'] }));
    writeFileSync(join(probe, 'index.html'), '<script>window.nmcGpu={ops:{render:async()=>({json:{ok:1},files:[]})}}</script>');
    const probeRef = pageRef(await gpuClient(client, ['publish', probe]));
    const probeInput = join(work, 'probe.json');
    const probeOut = join(work, 'probe-out');
    writeFileSync(probeInput, '{}');
    await gpuClient(client, ['render', probeInput, '--page-ref', probeRef, '-o', probeOut]);
    const probeResult = JSON.parse(readFileSync(join(probeOut, 'result.json'), 'utf8'));
    opResult(probeResult, value => value.ok === 1);
    stagePage();
    const ref = pageRef(await gpuClient(client, ['publish', STAGE]));
    const runs = [];
    // Start with the smallest canonical request; do not proceed if it fails.
    for (const [index, [seed, ticks]] of DET_CASES.entries()) {
      const input = join(work, `${index}.json`);
      const out = join(work, String(index));
      writeFileSync(input, JSON.stringify({ args: { case: 'clip-sample', seeds: [seed], ticks } }));
      await gpuClient(client, ['render', input, '--page-ref', ref, '-o', out]);
      const op = opResult(JSON.parse(readFileSync(join(out, 'result.json'), 'utf8')));
      const run = validateOne(op, seed, ticks, 'clip-sample');
      if (!run.hardware || run.adapter.isFallbackAdapter || !label(run.adapter)
          || /swiftshader|llvmpipe|lavapipe|software|basic render/i.test(label(run.adapter))) {
        throw new Error('clip-sample requires a named hardware adapter; no record written');
      }
      if (runs.length && JSON.stringify(run.adapter) !== JSON.stringify(runs[0].adapter)) {
        throw new Error('GPU adapter changed between cases; no record written');
      }
      runs.push(run);
    }
    const result = {
      adapter: runs[0].adapter,
      hardware: runs[0].hardware,
      adapterText: label(runs[0].adapter),
      records: runs.map(run => run.record),
    };
    const output = join(ROOT, 'proofs/results/anim/clip-sample.json');
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
    return result;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

export const USAGE = 'usage: node scripts/anim-capture.mjs --case det-selftest\n'
  + '       node scripts/anim-capture.mjs clip-sample (or --case clip-sample; requires NMC_GPU_CLIENT)';

export function parseArgs(argv) {
  const text = argv.join(' ');
  if (text === '--case det-selftest') return 'det-selftest';
  if (text === 'clip-sample' || text === '--case clip-sample') return 'clip-sample';
  return null;
}

const invoked = process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (invoked) {
  const mode = parseArgs(process.argv.slice(2));
  if (!mode) {
    console.error(USAGE);
    process.exitCode = 2;
  } else if (mode === 'clip-sample') {
    captureClipSample().then(result => {
      console.log(JSON.stringify(result));
      console.error('anim-capture: 3/3 client GPU records; wrote proofs/results/anim/clip-sample.json');
    }).catch(error => {
      console.error(`anim-capture: ${error.message}`);
      process.exitCode = 1;
    });
  } else {
    captureAnimBrick().then((result) => {
      console.log(JSON.stringify(result));
      console.error(`anim-capture: 3/3 Chrome records; adapter ${result.adapterText} (`
        + `${result.hardware ? 'hardware' : 'software'}); hash values are data, never compared`);
    }).catch((error) => {
      console.error(`anim-capture: ${error.message}`);
      process.exitCode = 1;
    });
  }
}
