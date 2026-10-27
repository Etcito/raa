#!/usr/bin/env node
// Functional Chrome runner for examples/anim-brick: loads the staged animation brick
// in a real Chrome and calls run() for each canonical det-selftest input. Records are
// checked for their exact shape and requested seed/ticks only; no hash or SHA value is
// compared with anything (owner rule: no hash proofs).
import { spawn } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
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

function validateOne(op, seed, ticks) {
  if (!op || typeof op !== 'object' || Array.isArray(op)) throw new Error('browser run returned no object');
  if (op.case !== 'det-selftest' || op.ticks !== ticks || !Array.isArray(op.seeds)
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
      || record.case !== 'det-selftest' || record.seed !== seed || record.ticks !== ticks
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

const invoked = process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (invoked) {
  if (process.argv.slice(2).join(' ') !== '--case det-selftest') {
    console.error('usage: node scripts/anim-capture.mjs --case det-selftest');
    process.exitCode = 2;
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
