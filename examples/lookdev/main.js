// Lookdev page of @etcito/raa: renders one (scene, variant) of examples/lookdev/scenes/*
// per run (S14). The engine comes ONLY from the staged package (./raa/index.js);
// ./common/nmc-gpu.js registers the ops of nomorechat's GPU client. The page requests a
// core adapter through the package and never substitutes a compatibility adapter.
import { loadRaa } from './raa/index.js';
import { registerNmcGpu } from './common/nmc-gpu.js';
import * as geometry from './lib/geometry.js';
import * as objects from './lib/objects.js';
import { runScene, renderAt, variantTimeMs } from './lib/runner.js';

const canvas = document.getElementById('c');
const picker = document.getElementById('picker');
const statusLine = document.getElementById('status');
const errorBox = document.getElementById('error');

// Uncaptured WebGPU errors of the current run (validation errors do not throw from
// render()). A non-empty list rejects the op instead of accepting a wrong capture.
const gpuErrors = [];

// The package does not expose its device, so the adapter request is wrapped ONLY to
// attach an 'uncapturederror' listener to every device it creates. The adapter request
// itself (core, the engine's options) is passed through unchanged.
function installGpuErrorListener() {
  const gpu = navigator.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return;
  const requestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options) => {
    const adapter = await requestAdapter(options);
    if (!adapter) return adapter;
    const requestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (descriptor) => {
      const device = await requestDevice(descriptor);
      device.addEventListener('uncapturederror', (event) => {
        const message = event.error && event.error.message ? event.error.message : String(event.error);
        gpuErrors.push(message);
        console.error(message);
      });
      return device;
    };
    return adapter;
  };
}

function showError(error) {
  const code = error && error.code ? error.code : 'page.error';
  let message = error && error.message ? error.message : String(error);
  // RaaError messages already start with "<code>: ".
  if (message.startsWith(`${code}: `)) message = message.slice(code.length + 2);
  errorBox.textContent = `${code}: ${message}`;
}

// Material helpers for scenes: v0 materials, colours sRGB in [0, 1].
function pbr(engine, color, opts = {}) {
  return engine.addMaterial({ ...opts, color, model: 'pbr' });
}

function flat(engine, color, opts = {}) {
  return engine.addMaterial({ ...opts, color, model: 'flat' });
}

const lib = Object.freeze({ geometry, objects, materials: Object.freeze({ pbr, flat }), pbr, flat });

// scenes.json (S14): [{ id, milestone, file }], file relative to the page ('scenes/x.js').
async function loadScenes() {
  const response = await fetch('./scenes.json');
  if (!response.ok) throw new Error(`scenes.json: HTTP ${response.status}`);
  const index = await response.json();
  const scenes = new Map();
  for (const entry of index) {
    const file = entry.file.startsWith('scenes/') ? entry.file : `scenes/${entry.file}`;
    const mod = await import(`./${file}`);
    scenes.set(entry.id, mod.default);
  }
  return scenes;
}

const ready = (async () => {
  installGpuErrorListener();
  const [raa, scenes] = await Promise.all([loadRaa(), loadScenes()]);
  return { raa, scenes };
})();

let engine = null;
let last = null; // { scene, variant } of the last run, for the screenshot
let running = false;

async function run(args) {
  const { raa, scenes } = await ready;
  if (running) throw new Error('run: a run is already in progress');
  const scene = scenes.get(args.scene);
  if (!scene) throw new Error(`run: unknown scene ${JSON.stringify(args.scene)}`);
  const variant = scene.variants.find((v) => v.id === args.variant);
  if (!variant) throw new Error(`run: scene ${scene.id} has no variant ${JSON.stringify(args.variant)}`);

  running = true;
  try {
    if (engine) engine.dispose();
    engine = null;
    last = null;
    gpuErrors.length = 0;
    const result = await runScene(raa, canvas, scene, variant, {
      width: args.width,
      height: args.height,
      lib,
      onEngine: (created) => { engine = created; },
    });
    last = { scene, variant };
    result.gpuErrors = gpuErrors.slice();
    if (gpuErrors.length > 0) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    return result;
  } finally {
    running = false;
  }
}

registerNmcGpu({
  run,
  // Draw the capture frame (same fixed time) and read the canvas in the same task,
  // before it is presented.
  screenshot: async () => {
    if (!engine || !last) throw new Error('screenshot: no completed run');
    renderAt(engine, last.scene, last.variant, variantTimeMs(last.variant));
    if (gpuErrors.length > 0) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    return canvas.toDataURL('image/png').split(',')[1];
  },
});

// Human use: pick a scene variant; each pick is one run.
async function runPicked() {
  const [scene, variant] = picker.value.split('--');
  errorBox.textContent = '';
  statusLine.textContent = `running ${picker.value}`;
  try {
    const r = await run({ scene, variant, width: 1280, height: 720 });
    statusLine.textContent = `${picker.value}: ${r.hardware ? 'hardware' : 'software'}, nonEmpty ${r.nonEmpty}, `
      + `${r.instances} instances, ${r.triangles} triangles`;
  } catch (error) {
    console.error(error);
    statusLine.textContent = '';
    showError(error);
  }
}

ready.then(({ scenes }) => {
  for (const scene of scenes.values()) {
    for (const variant of scene.variants) {
      const option = document.createElement('option');
      option.value = `${scene.id}--${variant.id}`;
      option.textContent = `${scene.id} / ${variant.id}`;
      picker.appendChild(option);
    }
  }
  picker.addEventListener('change', runPicked);
  if (picker.options.length === 0) statusLine.textContent = 'no scenes';
}).catch((error) => {
  console.error(error);
  showError(error);
});
