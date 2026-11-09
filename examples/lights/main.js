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

async function run(args) {
  window.__nmcGpuProgress = 'preparation';
  const { raa, scenes } = await ready;
  if (running) throw new Error('run already in progress');
  const scene = scenes.get(args.scene);
  const variant = scene?.variants.find((v) => v.id === args.variant);
  if (!variant) throw new Error(`unknown scene/variant: ${args.scene}/${args.variant}`);
  running = true;
  try {
    if (engine) engine.dispose();
    engine = null;
    last = null;
    gpuErrors.length = 0;
    window.__nmcGpuProgress = 'runScene';
    const result = await runScene(raa, canvas, scene, variant, {
      width: args.width, height: args.height, lib,
      onEngine: (created) => { engine = created; },
    });
    const preparedCapture = typeof scene.prepareCapture === 'function';
    if (preparedCapture) {
      window.__nmcGpuProgress = 'prepareCapture';
      result.lifecycle = await scene.prepareCapture(engine, variant, {
        width: canvas.width, height: canvas.height,
      });
    }
    result.gpuErrors = gpuErrors.slice();
    if (gpuErrors.length) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    last = { scene, variant, preparedCapture };
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
