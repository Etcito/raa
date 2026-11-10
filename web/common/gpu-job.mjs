#!/usr/bin/env node
// Managed GPU-page transport: verified publication and one bounded page operation.
// The managed client inherits the caller's environment (including proxy settings)
// and performs the network calls; this module never starts a browser itself.
// A job's queue wait, running clock and output download are attributed separately
// so waiting for the GPU slot never consumes the operation's own work budget.
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const MAX_WORK_TIMEOUT_MS = 300000;
const nonblank = (v) => typeof v === 'string' && v.trim().length > 0;
const hostId = (v) => nonblank(v) || (Number.isSafeInteger(v) && v > 0);
const ensure = (ok, message) => { if (!ok) throw new Error(message); };
const redact = (text) => {
  const token = process.env.NMC_GPU_TOKEN;
  const value = String(text ?? '');
  return token ? value.split(token).join('[redacted]') : value;
};

// Queue availability is a separate allowance from the bounded operation work.
function waitSeconds() {
  const n = process.env.NMC_GPU_WAIT_S === undefined ? 60 : Number(process.env.NMC_GPU_WAIT_S);
  ensure(Number.isFinite(n) && n > 0, 'NMC_GPU_WAIT_S must be positive seconds');
  return n;
}

function workTimeout(options) {
  const ms = options?.timeoutMs;
  ensure(typeof ms === 'number' && Number.isFinite(ms) && ms > 0 && ms <= MAX_WORK_TIMEOUT_MS,
    `work timeoutMs must be a positive number of at most ${MAX_WORK_TIMEOUT_MS} ms`);
  return ms;
}

// A page is publishable only when the build actually produced its client
// manifest and the declared entry page; the caller never publishes a bare dir.
function verifyBuiltPage(directory) {
  ensure(existsSync(directory) && statSync(directory).isDirectory(),
    `page directory is missing or not a directory: ${directory}: build the page before publishing`);
  const manifest = join(directory, 'nmc.gpu.json');
  ensure(existsSync(manifest) && lstatSync(manifest).isFile(),
    `${manifest} is missing: publish only a verified built page directory`);
  let doc;
  try { doc = JSON.parse(readFileSync(manifest, 'utf8')); } catch (e) {
    throw new Error(`${manifest} is not valid JSON (${e.message}): publish only a verified built page directory`);
  }
  ensure(doc !== null && typeof doc === 'object' && !Array.isArray(doc) && nonblank(doc.page) && Array.isArray(doc.ops),
    `${manifest} must declare {page, ops}: publish only a verified built page directory`);
  const pageFile = join(directory, doc.page);
  ensure(existsSync(pageFile) && lstatSync(pageFile).isFile(),
    `declared page ${doc.page} is missing: publish only a verified built page directory`);
}

async function managedClient(waitS) {
  const client = process.env.NMC_GPU_CLIENT;
  ensure(nonblank(client), 'NMC_GPU_CLIENT is missing: managed Windows Chrome is required');
  const until = Date.now() + waitS * 1000;
  while (!existsSync(client) && Date.now() < until) await delay(Math.min(100, Math.max(1, until - Date.now())));
  ensure(existsSync(client) && lstatSync(client).isFile(), `NMC_GPU_CLIENT unavailable after ${waitS}s`);
  return client;
}

// The installed CLI reports its queue/running transitions on stderr. Its running
// clock is separate from slot availability; never pass a wall-clock spawn timeout.
function clientCall(client, args, { waitS, workMs, job }) {
  return new Promise((resolveCall, reject) => {
    const child = spawn(process.execPath, [client, ...args], { stdio: ['ignore', 'pipe', 'pipe'], shell: false, env: process.env });
    let stdout = '', stderr = '', lineBuffer = '', failure = null, running = false;
    const stop = (message) => { failure ??= new Error(message); child.kill('SIGKILL'); };
    const seconds = (ms) => `${ms / 1000}`;
    let timer = setTimeout(() => stop(job
      ? `managed client GPU waiting timeout after ${waitS}s`
      : `managed client publication timeout after ${seconds(workMs)}s`), job ? waitS * 1000 : workMs);
    child.stdout.on('data', (data) => { stdout += data; if (stdout.length > 1024 * 1024) stop('managed client stdout too large'); });
    child.stderr.on('data', (data) => {
      stderr = (stderr + data).slice(-16000);
      lineBuffer += data;
      const lines = lineBuffer.split(/\r?\n/);
      lineBuffer = lines.pop().slice(-16000);
      for (const line of lines) {
        if (job && /^GPU job .+: running \(/.test(line) && !running) {
          running = true;
          clearTimeout(timer);
          timer = setTimeout(() => stop(`GPU job running timeout after ${seconds(workMs)}s`), workMs);
        } else if (job && /^GPU job .+: done$/.test(line)) {
          // Downloads happen after the job finishes. Do not truncate a successful
          // job just because its separate file transfer crosses the running limit.
          clearTimeout(timer);
          timer = setTimeout(() => stop('managed output download timeout'), waitS * 1000);
        }
      }
    });
    child.on('error', (cause) => { failure = new Error(`managed client could not start: ${cause.message}`, { cause }); });
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      if (failure) return reject(failure);
      const details = redact(stderr);
      if (code !== 0) return reject(new Error(`managed client failed (${code ?? signal}): ${details.trim()}`));
      resolveCall(stdout);
    });
  });
}

function pageReference(stdout) {
  const pick = (o) => [o, o?.result, o?.data].map((v) => v?.page_ref ?? v?.pageRef).find(nonblank);
  for (const text of [stdout, ...stdout.split(/\r?\n/)]) {
    try { const ref = pick(JSON.parse(text)); if (ref) return ref.trim(); } catch { /* CLI progress line */ }
    const match = /^\s*(?:page_ref|pageRef)\s*[:=]\s*["']?([^\s"']+)["']?\s*$/.exec(text);
    if (match) return match[1];
  }
  throw new Error('managed publication returned no page_ref');
}

function readResult(path) {
  ensure(existsSync(path), `${path} is missing: managed job must return its complete result`);
  let doc;
  try { doc = JSON.parse(readFileSync(path, 'utf8')); } catch (e) {
    throw new Error(`${path} is not valid JSON (${e.message}): managed job returned a malformed response`);
  }
  ensure(doc !== null && typeof doc === 'object' && !Array.isArray(doc),
    `${path} is not a host reply object: managed job returned a malformed response`);
  return doc;
}

/** Publishes one verified built page directory and returns the host page reference. */
export async function publishGpuPage(directory, options) {
  const timeoutMs = workTimeout(options);
  ensure(nonblank(directory), 'publishGpuPage requires a page directory');
  const built = resolve(directory);
  verifyBuiltPage(built);
  const waitS = waitSeconds();
  const client = await managedClient(waitS);
  const stdout = await clientCall(client, ['publish', built], { waitS, workMs: timeoutMs, job: false });
  const pageRef = pageReference(stdout);
  ensure(nonblank(pageRef), 'managed publication returned an empty page reference');
  return { pageRef };
}

/** Runs one declared page operation and returns its nonempty host identities with the complete reply.result.
 *  The legacy lights transport also passes options.out so its attempt retains input.json/result.json. */
export async function runGpuPageOp(pageRef, op, input, options) {
  const timeoutMs = workTimeout(options);
  ensure(nonblank(pageRef), 'runGpuPageOp requires a nonempty host-issued page reference');
  ensure(nonblank(op), 'runGpuPageOp requires a nonempty declared operation name');
  let serialized;
  try { serialized = JSON.stringify(input); } catch (e) {
    throw new Error(`operation input is not serializable JSON (${e.message})`);
  }
  ensure(typeof serialized === 'string', 'operation input is not serializable JSON');
  const waitS = waitSeconds();
  const client = await managedClient(waitS);
  const provided = nonblank(options?.out) ? resolve(options.out) : null;
  const scratch = provided ? null : mkdtempSync(join(tmpdir(), 'nmc-gpu-page-'));
  const outDir = provided ?? join(scratch, 'out');
  mkdirSync(outDir, { recursive: true });
  const inputPath = join(outDir, 'input.json');
  try {
    // The input is forwarded as JSON, never reinterpreted (e.g. remainingMs).
    writeFileSync(inputPath, `${serialized}\n`);
    rmSync(join(outDir, 'result.json'), { force: true });
    await clientCall(client, [op, inputPath, '--page-ref', pageRef, '-o', outDir], { waitS, workMs: timeoutMs, job: true });
    const reply = readResult(join(outDir, 'result.json'));
    ensure(hostId(reply.job_id) && hostId(reply.client_id),
      'managed job response is missing nonempty managed job/client provenance');
    ensure(Object.hasOwn(reply, 'result'), 'managed job response is missing its complete result');
    return { jobId: String(reply.job_id), clientId: String(reply.client_id), result: reply.result };
  } finally {
    if (scratch) rmSync(scratch, { recursive: true, force: true });
  }
}
