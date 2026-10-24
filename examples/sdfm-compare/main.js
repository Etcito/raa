// Side-by-side compare page: the MAMA reference viewport (left: the meshes and base colours
// MAMA 864ebc3 produces, drawn with plain WebGPU, because MAMA has no viewport host) and RAA
// through @etcito/raa-sdfm (right) show the same sdfm.game-asset fixture with the same camera.
// The engine comes ONLY from the staged package (./raa/index.js); ./common/nmc-gpu.js
// registers the ops of nomorechat's GPU client. Both panes request a core adapter and never
// substitute a compatibility adapter.
import { loadRaa } from './raa/index.js';
import { registerNmcGpu } from './common/nmc-gpu.js';
import { runCompare, renderCapture } from './lib/runner.js';
import { PANE_LABEL as MAMA_LABEL } from './lib/mama-view.js';

const RAA_LABEL = 'RAA (@etcito/raa-sdfm)';

const mamaCanvas = document.getElementById('mama');
const raaCanvas = document.getElementById('raa');
const picker = document.getElementById('picker');
const statusLine = document.getElementById('status');
const errorBox = document.getElementById('error');
document.getElementById('mama-label').textContent = MAMA_LABEL;
document.getElementById('raa-label').textContent = RAA_LABEL;

// Uncaptured WebGPU errors of the current run (validation errors do not throw from
// render()). A non-empty list rejects the op instead of accepting a wrong capture.
const gpuErrors = [];

// The package does not expose its device, so the adapter request is wrapped ONLY to attach
// an 'uncapturederror' listener to every device it creates (RAA and MAMA). The adapter
// request itself (core, the engine's options) is passed through unchanged.
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
  // RaaError and SdfmError messages already start with "<code>: ".
  if (message.startsWith(`${code}: `)) message = message.slice(code.length + 2);
  errorBox.textContent = `${code}: ${message}`;
}

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

let state = null; // completed run, kept for the screenshot
let running = false;

function disposeState() {
  if (!state) return;
  try {
    if (state.mamaView && typeof state.mamaView.dispose === 'function') state.mamaView.dispose();
  } catch (error) {
    console.error(error);
  }
  try {
    state.engine.dispose();
  } catch (error) {
    console.error(error);
  }
  state = null;
}

async function run(args) {
  const { raa, scenes } = await ready;
  if (running) throw new Error('run: a run is already in progress');
  const scene = scenes.get(args.scene);
  if (!scene) throw new Error(`run: unknown scene ${JSON.stringify(args.scene)}`);
  const variant = scene.variants.find((v) => v.id === args.variant);
  if (!variant) throw new Error(`run: scene ${scene.id} has no variant ${JSON.stringify(args.variant)}`);

  running = true;
  try {
    disposeState();
    gpuErrors.length = 0;
    const result = await runCompare(raa, { mama: mamaCanvas, raa: raaCanvas }, scene, variant, {
      width: args.width,
      height: args.height,
      onState: (completed) => { state = completed; },
    });
    result.gpuErrors = gpuErrors.slice();
    if (gpuErrors.length > 0) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    return result;
  } catch (error) {
    disposeState();
    throw error;
  } finally {
    running = false;
  }
}

// One 1280x720 image of both panes, each labelled: MAMA reference viewport left, RAA right.
function composeCapture(width, height) {
  const shot = document.createElement('canvas');
  shot.width = width;
  shot.height = height;
  const ctx = shot.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(mamaCanvas, 0, 0, width / 2, height);
  ctx.drawImage(raaCanvas, width / 2, 0, width / 2, height);
  ctx.font = '16px sans-serif';
  ctx.textBaseline = 'top';
  for (const [label, x] of [[MAMA_LABEL, 0], [RAA_LABEL, width / 2]]) {
    const w = ctx.measureText(label).width + 16;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(x + 8, 8, w, 26);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 16, 13);
  }
  return shot;
}

registerNmcGpu({
  run,
  // Draw the capture frame (same fixed time) and read both canvases in the same task,
  // before they are presented.
  screenshot: async () => {
    if (!state) throw new Error('screenshot: no completed run');
    await renderCapture(state);
    if (gpuErrors.length > 0) throw new Error(`GPU errors: ${gpuErrors.join('\n')}`);
    const shot = composeCapture(state.width, state.height);
    return shot.toDataURL('image/png').split(',')[1];
  },
});

// Human use: pick a fixture variant; each pick is one run.
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
