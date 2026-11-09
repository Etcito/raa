import { loadRaa } from './raa/index.js';
import { registerNmcGpu } from './common/nmc-gpu.js';
import * as geometry from './lib/geometry.js';
import * as objects from './lib/objects.js';
import { runScene, renderAt, variantTimeMs } from './lib/runner.js';

const canvas = document.getElementById('c');
const status = document.getElementById('status');
const gpuErrors = [];
const preparation = { status: 'pending', error: null };
let engine = null;
let residents = [];
let last = null;
let running = false;

function installGpuErrors() {
  const gpu = navigator.gpu;
  if (!gpu) return;
  const requestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options) => {
    const adapter = await requestAdapter(options);
    if (!adapter) return adapter;
    const requestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (descriptor) => {
      const device = await requestDevice(descriptor);
      device.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      return device;
    };
    return adapter;
  };
}

const pbr = (engine, color, opts = {}) => engine.addMaterial({ ...opts, color, model: 'pbr' });
const flat = (engine, color, opts = {}) => engine.addMaterial({ ...opts, color, model: 'flat' });
const lib = Object.freeze({ geometry, objects, pbr, flat, materials: Object.freeze({ pbr, flat }) });

/**
 * Handoff: default.build(engine, lib, variant) and optional default.frame(engine, timeMs,
 * variant), exported by examples/lights/scenes/<id>.js, attach through loadScenes().
 * The registry is Map<scene.id, default>; variant.id selects on/off. runScene receives
 * the frozen lib above and a fresh engine; scene lifecycle state stays keyed by engine.
 * Motion, point/opaque/transparent/fur, spot, directional/area, attached-light,
 * many-lights and later M3 modules use this same discovery entry point.
 */
async function loadScenes() {
  const response = await fetch('./scenes.json');
  if (!response.ok) throw new Error(`scenes.json: HTTP ${response.status}`);
  const scenes = new Map();
  for (const entry of await response.json()) {
    const scene = (await import(`./${entry.file}`)).default;
    scenes.set(entry.id, scene);
  }
  return scenes;
}

const ready = (async () => {
  installGpuErrors();
  const [raa, scenes] = await Promise.all([loadRaa(), loadScenes()]);
  return { raa, scenes };
})();

// Chrome/Windows metadata for a measurement, recorded next to the adapter identity.
function browserMetadata() {
  const nav = typeof navigator === 'object' && navigator ? navigator : {};
  return {
    userAgent: String(nav.userAgent ?? ''),
    platform: String(nav.platform ?? ''),
    devicePixelRatio: typeof globalThis.devicePixelRatio === 'number' ? globalThis.devicePixelRatio : 1,
  };
}

/**
 * Handoff: `runMeasured(raa, scene, variant)` in `examples/lights/main.js` is the
 * many-lights operation. It calls `runScene(raa, canvas, scene, variant, args)` from
 * `examples/lookdev/lib/runner.js` once per on/off variant with the scene's `measure`
 * configuration (`warmup`, `frames`, `width`, `height`), keeps both engines resident for
 * the operation and returns `{ runs: { on, off }, adapter, hardware, browser,
 * measurement_for, gpuErrors }`. The `loadScenes()` Map<scene.id, default> and variant.id
 * on/off registry are unchanged by this operation.
 */
async function runMeasured(raa, scene, variant) {
  const config = scene.measure;
  const runs = {};
  const devices = new Map();
  // The requested variant runs last so its engine owns the canvas configuration for the
  // capture; both engines stay resident for the operation.
  for (const candidate of [...scene.variants.filter((v) => v.id !== variant.id), variant]) {
    window.__nmcGpuProgress = `measure:${candidate.id}`;
    runs[candidate.id] = await runScene(raa, canvas, scene, candidate, {
      width: config.width, height: config.height, lib,
      measure: { warmup: config.warmup, frames: config.frames },
      onEngine: (created) => {
        devices.set(candidate.id, created);
        residents.push(created);
      },
    });
  }
  engine = devices.get(variant.id);
  const on = runs.on;
  const problems = [];
  for (const [id, result] of Object.entries(runs)) {
    if (!Array.isArray(result.canvas) || result.canvas[0] !== config.width || result.canvas[1] !== config.height) {
      problems.push(`${id}: canvas ${JSON.stringify(result.canvas)} is not ${config.width}x${config.height}`);
    }
    if (!(Number.isFinite(result.fpsMean) && result.fpsMean > 0)) problems.push(`${id}: fpsMean is not a frame rate`);
    for (const key of ['frameIntervalMsMedian', 'frameIntervalMsP95', 'fpsMedian', 'fpsP95']) {
      if (!Number.isFinite(result[key])) problems.push(`${id}: ${key} is not a number`);
    }
  }
  if (!(on.fpsMean >= config.minFpsMean)) problems.push(`on: fpsMean ${on.fpsMean} is below ${config.minFpsMean}`);
  if (problems.length > 0) throw new Error(`many-lights: ${problems.join('; ')}`);
  last = { scene, variant, preparedCapture: false, captures: [] };
  if (scene.legend) status.textContent = scene.legend;
  if (gpuErrors.length) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
  return {
    ...runs[variant.id], runs, adapter: on.adapter, hardware: on.hardware,
    browser: browserMetadata(), measurement_for: config.measurementFor, gpuErrors: gpuErrors.slice(),
  };
}

async function run(args) {
  window.__nmcGpuProgress = 'preparation';
  const { raa, scenes } = await ready;
  if (running) throw new Error('run already in progress');
  const scene = scenes.get(args.scene);
  const variant = scene?.variants.find((v) => v.id === args.variant);
  if (!variant) throw new Error(`unknown scene/variant: ${args.scene}/${args.variant}`);
  running = true;
  try {
    for (const resident of residents) resident.dispose();
    residents = [];
    engine = null;
    last = null;
    gpuErrors.length = 0;
    if (scene.measure) return await runMeasured(raa, scene, variant);
    window.__nmcGpuProgress = 'runScene';
    const result = await runScene(raa, canvas, scene, variant, {
      width: args.width, height: args.height, lib,
      onEngine: (created) => { engine = created; residents.push(created); },
    });
    const preparedCapture = typeof scene.prepareCapture === 'function';
    const captures = [];
    if (preparedCapture) {
      window.__nmcGpuProgress = 'prepareCapture';
      result.lifecycle = await scene.prepareCapture(engine, variant, {
        width: canvas.width, height: canvas.height,
        capture: (name) => {
          window.__nmcGpuProgress = `capture:${name}`;
          captures.push({ name, base64: canvas.toDataURL('image/png').split(',')[1] });
        },
      });
    }
    result.gpuErrors = gpuErrors.slice();
    if (gpuErrors.length) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    last = { scene, variant, preparedCapture, captures };
    if (scene.legend) status.textContent = scene.legend;
    return result;
  } finally {
    running = false;
  }
}

registerNmcGpu({
  run,
  screenshot: async () => {
    window.__nmcGpuProgress = 'capture';
    if (!engine || !last) throw new Error('no completed run');
    if (!last.preparedCapture) {
      renderAt(engine, last.scene, last.variant, variantTimeMs(last.variant));
    }
    if (gpuErrors.length) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    return canvas.toDataURL('image/png').split(',')[1];
  },
});
const render = window.nmcGpu.ops.render;
window.nmcGpu.ops.render = async (input) => {
  const result = await render(input);
  result.files.push(...last.captures);
  return result;
};
window.nmcGpu.ready = false;
window.nmcGpu.preparation = preparation;
ready.then(() => {
  preparation.status = 'ready';
  window.nmcGpu.ready = true;
  status.textContent = 'Lights scenes ready';
}).catch((error) => {
  preparation.status = 'failed';
  preparation.error = error.message;
  status.textContent = `Preparation failed: ${error.message}`;
});
