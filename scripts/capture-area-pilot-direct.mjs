#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [chromeArg, pageArg, outputArg] = process.argv.slice(2);
if (process.platform !== 'win32' || !chromeArg || !pageArg || !outputArg) {
  throw new Error('Usage on Windows: node scripts/capture-area-pilot-direct.mjs <chrome.exe> <built GI page> <new output>');
}
const chrome = path.resolve(chromeArg), page = path.resolve(pageArg), output = path.resolve(outputArg);
const receiptBytes = await readFile(path.join(page, 'build.json'));
const receipt = JSON.parse(receiptBytes);
if (receipt.status !== 'ready' || !receipt.buildId) throw new Error('A ready GI build is required');

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const served = new Set(receipt.files);
const server = createServer(async (request, response) => {
  const name = new URL(request.url, 'http://127.0.0.1').pathname.slice(1) || 'index.html';
  if (request.method !== 'GET' || !served.has(name) || name.split('/').some(part => !part || part === '..' || part.includes('\\'))) {
    response.writeHead(404).end();
    return;
  }
  const file = path.join(page, name);
  try {
    const info = await stat(file);
    response.writeHead(200, { 'content-type': types[path.extname(file)] ?? 'application/octet-stream', 'content-length': info.size });
    createReadStream(file).on('error', () => response.destroy()).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
const profile = await mkdtemp(path.join(tmpdir(), 'raa-area-pilot-'));
let browser;
let socket;
let launchError;
const stderr = [];
try {
  browser = spawn(chrome, [
    '--headless=new', '--do-not-de-elevate', '--enable-logging=stderr', '--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--disable-software-rasterizer',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true });
  browser.on('error', error => { launchError = error; });
  browser.stderr.on('data', chunk => {
    stderr.push(chunk.toString());
    if (stderr.length > 12) stderr.shift();
  });
  const profilePort = path.join(profile, 'DevToolsActivePort');
  let port;
  for (let attempt = 0; attempt < 100; attempt++) {
    if (launchError) throw launchError;
    if (browser.exitCode !== null) throw new Error(`Chrome exited ${browser.exitCode}`);
    try { [port] = (await readFile(profilePort, 'utf8')).split(/\r?\n/); break; }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    await delay(100);
  }
  if (!/^\d+$/.test(port ?? '')) throw new Error('Chrome DevTools did not start');
  const debug = `http://127.0.0.1:${port}`;
  const version = await (await fetch(`${debug}/json/version`)).json();
  socket = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let serial = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    const reply = pending.get(message.id);
    if (reply) {
      pending.delete(message.id);
      if (message.error) reply.reject(new Error(message.error.message));
      else reply.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown' && browserErrors.length < 8)
      browserErrors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
  });
  socket.addEventListener('close', () => {
    for (const reply of pending.values()) reply.reject(new Error('Chrome DevTools connection closed'));
    pending.clear();
  });
  const browserErrors = [];
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params, sessionId }));
  });
  const { targetInfos } = await call('Target.getTargets');
  const { sessionId } = await call('Target.attachToTarget', { targetId: targetInfos.find(info => info.type === 'page').targetId, flatten: true });
  const send = (method, params) => call(method, params, sessionId);
  const evalPage = async (expression, timeout = 10000) => {
    const { result, exceptionDetails } = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
    return result.value;
  };
  await send('Runtime.enable');
  await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
  let ready;
  for (let attempt = 0; attempt < 100; attempt++) {
    ready = await evalPage("typeof window.nmcGpu?.ops?.['area-pilot'] === 'function'");
    if (ready || browserErrors.length) break;
    await delay(100);
  }
  if (!ready) throw new Error(`GI capture page did not initialize: ${browserErrors.join('; ')}`);
  const { gpu } = await call('SystemInfo.getInfo');
  const renderer = gpu.auxAttributes?.glRenderer ?? '';
  const nvidia = gpu.devices.find(device => device.vendorId === 0x10de && /RTX 3090/i.test(device.deviceString));
  if (!nvidia || !/RTX 3090/i.test(renderer) || /swiftshader|software|warp/i.test(renderer)) {
    throw new Error(`Expected native RTX 3090 Chrome rendering, got ${renderer}`);
  }
  const webgpu = await evalPage(`(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    return info && {
      vendor: info.vendor, device: info.device, description: info.description,
      isFallbackAdapter: info.isFallbackAdapter,
    };
  })()`);
  if (!webgpu || webgpu.isFallbackAdapter !== false || !/nvidia|10de/i.test(webgpu.vendor) ||
      !/3090|2204/i.test(`${webgpu.device} ${webgpu.description}`)) {
    throw new Error(`Expected native RTX 3090 WebGPU, got ${JSON.stringify(webgpu)}`);
  }
  console.log(`Windows ${version.Browser}: ${renderer}; WebGPU ${JSON.stringify(webgpu)}`);
  const capture = await evalPage(`(async () => {
    const result = await window.nmcGpu.ops['area-pilot'](${JSON.stringify({ buildId: receipt.buildId, remainingMs: 280000 })});
    const files = await Promise.all(result.files.map(async ({ name, blob }) => {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const chunks = [];
      for (let i = 0; i < bytes.length; i += 16384) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 16384)));
      return { name, base64: btoa(chunks.join('')) };
    }));
    return { json: result.json, files };
  })()`, 300000);
  const data = capture.json;
  const adapter = data.facts?.adapter;
  if (!data.facts?.hardware || !/Windows/.test(data.facts.browser) ||
      adapter?.isFallbackAdapter !== false || !/3090|2204/i.test(adapter.device) || data.buildId !== receipt.buildId) {
    throw new Error(`Area WebGPU adapter or build provenance mismatch: ${JSON.stringify(data.facts)}`);
  }
  await mkdir(path.dirname(output), { recursive: true });
  await mkdir(output);
  const files = new Map(capture.files.map(file => [file.name, Buffer.from(file.base64, 'base64')]));
  for (const frame of data.frames) {
    for (const record of [frame.input, frame.reference, ...Object.values(frame.native)]) {
      const bytes = files.get(record.file);
      if (!/^[a-z0-9-]+\.bin$/.test(record.file) || !bytes ||
          bytes.length !== record.byteLength || createHash('sha256').update(bytes).digest('hex') !== record.sha256) {
        throw new Error(`Capture length or hash mismatch: ${record.file}`);
      }
      await writeFile(path.join(output, record.file), bytes, { flag: 'wx' });
    }
  }
  const source = { revision: receipt.inputs.revision, receiptSha256: createHash('sha256').update(receiptBytes).digest('hex') };
  const manifest = path.join(output, 'manifest.json');
  await writeFile(manifest, JSON.stringify({ ...data, source, windows: { browser: version.Browser, renderer, device: nvidia, webgpu } }, null, 2) + '\n', { flag: 'wx' });
  console.log(manifest);
} catch (error) {
  if (stderr.length) console.error(stderr.join('').trim());
  throw error;
} finally {
  socket?.close();
  if (browser && browser.exitCode === null) browser.kill();
  server.closeAllConnections();
  server.close();
  await rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 150 }).catch(error => console.error(error.message));
}
