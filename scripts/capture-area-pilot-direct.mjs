#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [chromeArg, pageArg, outputArg, mode = 'area-pilot', comparison = 'fixed'] = process.argv.slice(2);
const op = mode === '--prediction' ? 'prediction' : mode;
const capture = op === 'area-pilot' || op === 'gallery-path-capture';
const gallery = op.startsWith('gallery-');
const spherical = op === 'gallery-spherical';
const sequence = op === 'gallery-sequence';
const liveSpherical = spherical && comparison === 'live';
if (process.platform !== 'win32' || !chromeArg || !pageArg || !outputArg
    || !['area-pilot', 'prediction', 'gallery-path-capture', 'gallery-path-measure', 'gallery-path-diffuse', 'gallery-panel-noise', 'gallery-spherical', 'gallery-sequence'].includes(op)) {
  throw new Error('Usage on Windows: node scripts/capture-area-pilot-direct.mjs <chrome.exe> <built page> <new output> [--prediction|gallery-path-capture|gallery-path-measure|gallery-path-diffuse|gallery-panel-noise|gallery-spherical|gallery-sequence] [baseline|fixed|live|train|test|both]');
}
const chrome = path.resolve(chromeArg), page = path.resolve(pageArg), output = path.resolve(outputArg);
const receiptBytes = await readFile(path.join(page, 'build.json'));
const receipt = JSON.parse(receiptBytes);
if (receipt.status !== 'ready' || !receipt.buildId) throw new Error('A ready page build is required');

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const served = new Set([...receipt.files, 'build.json']);
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
const url = liveSpherical ? 'http://127.0.0.1:8081/' : `http://127.0.0.1:${server.address().port}/`;
const profile = await mkdtemp(path.join(tmpdir(), 'raa-area-pilot-'));
let browser;
let socket;
let launchError;
const stderr = [];
try {
  if (liveSpherical) {
    const response = await fetch(new URL('build.json', url), { signal: AbortSignal.timeout(10000) });
    if (!response.ok || !receiptBytes.equals(Buffer.from(await response.arrayBuffer()))) {
      throw new Error('The live 8081 receipt does not match the staged spherical build');
    }
  }
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
  await send('Page.navigate', { url });
  let ready;
  for (let attempt = 0; attempt < 100; attempt++) {
    ready = await evalPage(`typeof window.nmcGpu?.ops?.[${JSON.stringify(op)}] === 'function'`);
    if (ready || browserErrors.length) break;
    await delay(100);
  }
  if (!ready) throw new Error(`Capture page did not initialize: ${browserErrors.join('; ')}`);
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
  const result = liveSpherical ? null : await evalPage(`(async () => {
    const result = await window.nmcGpu.ops[${JSON.stringify(op)}](${JSON.stringify({
      buildId: receipt.buildId, remainingMs: 280000, requireFixed: comparison !== 'baseline',
      ...(sequence && { split: comparison === 'fixed' ? 'both' : comparison }),
    })});
    const files = await Promise.all(result.files.map(async ({ name, blob, base64 }) => {
      if (base64) return { name, base64 };
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const chunks = [];
      for (let i = 0; i < bytes.length; i += 16384) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 16384)));
      return { name, base64: btoa(chunks.join('')) };
    }));
    return { json: result.json, files };
  })()`, 300000);
  const ui = spherical ? await evalPage(`import('/lib/gallery-spherical-evidence.js').then(module => module.verifyGallerySphericalUi())`, 120000) : null;
  if ((spherical || sequence) && browserErrors.length) throw new Error(browserErrors.join('\n'));
  const windows = { browser: version.Browser, renderer, device: nvidia, webgpu };
  if (liveSpherical) {
    await mkdir(path.dirname(output), { recursive: true });
    await mkdir(output);
    const manifest = path.join(output, 'live-ui.json');
    await writeFile(manifest, JSON.stringify({ revision: receipt.inputs.revision, buildId: receipt.buildId,
      url, windows, ui }, null, 2) + '\n', { flag: 'wx' });
    console.log(manifest);
  } else {
    const data = result.json;
    const facts = gallery ? data.source?.facts : data.facts;
    const adapter = facts?.adapter;
    if (!facts?.hardware || !/Windows/.test(facts.browser) ||
        adapter?.isFallbackAdapter !== false || !/3090|2204/i.test(adapter.device) || (gallery ? data.source.buildId : data.buildId) !== receipt.buildId) {
      throw new Error(`WebGPU adapter or build provenance mismatch: ${JSON.stringify(facts)}`);
    }
    await mkdir(path.dirname(output), { recursive: true });
    await mkdir(output);
    const files = new Map(result.files.map(file => [file.name, Buffer.from(file.base64, 'base64')]));
    if (capture) {
      for (const frame of data.frames) {
        const records = gallery ? Object.values(frame.signals).flatMap(signal => [signal.input, signal.reference])
          : [frame.input, frame.reference, ...Object.values(frame.native)];
        for (const record of records) {
          const bytes = files.get(record.file);
          if (!/^[a-z0-9-]+\.bin$/.test(record.file) || !bytes ||
              bytes.length !== record.byteLength || createHash('sha256').update(bytes).digest('hex') !== record.sha256) {
            throw new Error(`Capture length or hash mismatch: ${record.file}`);
          }
          await writeFile(path.join(output, record.file), bytes, { flag: 'wx' });
        }
      }
    } else if (op === 'gallery-panel-noise') {
      for (const { planes } of data.frames) {
        for (const plane of Object.values(planes)) {
          const bytes = files.get(plane.file);
          if (!/^pan-[0-9]+-[a-z]+\.bin$/.test(plane.file) || !bytes ||
              bytes.length !== plane.byteLength || createHash('sha256').update(bytes).digest('hex') !== plane.sha256) {
            throw new Error(`Panel plane length or hash mismatch: ${plane.file}`);
          }
          await writeFile(path.join(output, plane.file), bytes, { flag: 'wx' });
        }
      }
    } else if (op === 'gallery-path-measure') {
      for (const row of data.cases) {
        const bytes = files.get(row.file);
        if (!/^[a-z0-9-]+\.png$/.test(row.file) || !bytes?.length) throw new Error(`Missing measurement image ${row.file}`);
        row.sha256 = createHash('sha256').update(bytes).digest('hex');
        await writeFile(path.join(output, row.file), bytes, { flag: 'wx' });
      }
    } else if (sequence) {
      for (const clip of data.sequences) {
        for (const frame of clip.frames) {
          for (const record of Object.values(frame.planes)) {
            const bytes = files.get(record.file);
            if (!/^[a-z0-9-]+\.bin$/.test(record.file) || record.dtype !== 'f32le' || !bytes
                || bytes.length !== record.byteLength || bytes.length !== record.shape.reduce((a, b) => a * b, 4)
                || createHash('sha256').update(bytes).digest('hex') !== record.sha256) {
              throw new Error(`Sequence plane length or hash mismatch: ${record.file}`);
            }
            await writeFile(path.join(output, record.file), bytes, { flag: 'wx' });
          }
        }
      }
    } else if (spherical) {
      for (const record of data.captures) {
        const bytes = files.get(record.file);
        if (!/^[a-z0-9-]+\.rgba16f$/.test(record.file) || !bytes || bytes.length !== record.byteLength
            || createHash('sha256').update(bytes).digest('hex') !== record.sha256) {
          throw new Error(`Spherical capture length or hash mismatch: ${record.file}`);
        }
        await writeFile(path.join(output, record.file), bytes, { flag: 'wx' });
      }
    }
    const source = {
      ...(gallery ? data.source : {}), revision: receipt.inputs.revision,
      receiptSha256: createHash('sha256').update(receiptBytes).digest('hex'),
    };
    const manifest = path.join(output, op === 'prediction' ? 'prediction.json' : capture || sequence ? 'manifest.json' : op === 'gallery-path-diffuse' ? 'diffuse.json'
      : op === 'gallery-panel-noise' ? 'panel-noise.json' : spherical ? 'spherical.json' : 'measurement.json');
    await writeFile(manifest, JSON.stringify({ ...data, source, windows, ...(ui && { ui }) }, null, 2) + '\n', { flag: 'wx' });
    console.log(manifest);
  }
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
