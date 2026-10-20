// m5-booleans page: every supported CSG case in a rotating grid, driven by requestAnimationFrame, with
// the `run` and `render` ops for nomorechat's GPU client (window.nmcGpu.ops, see
// common/nmc-gpu.js).
import init, { start } from './pkg/demo_m5_booleans.js';
import { nextFrame, runMeasured } from './common/run-loop.js';
import { registerNmcGpu } from './common/nmc-gpu.js';

const canvas = document.getElementById('c');
const gpuErrors = [];

// Some Chrome builds (M1 dogfood item 6) return SwiftShader for a core adapter and the
// hardware GPU only at featureLevel 'compatibility'. Prefer that hardware adapter. The
// demo's hardware flag still comes only from the real adapter info.
function installHardwareAdapterWrapper() {
  const gpu = navigator.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') return;
  const original = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options = {}) => {
    const core = await original(options);
    if (core && !(core.info && core.info.isFallbackAdapter)) {
      console.info('requestAdapter: using the core adapter');
      return core;
    }
    const compat = await original({ ...options, featureLevel: 'compatibility' });
    if (!compat || (compat.info && compat.info.isFallbackAdapter)) {
      console.info('requestAdapter: no hardware compatibility adapter, using the core result');
      return core;
    }
    // Clamp every required limit to what the compatibility adapter supports.
    const requestDevice = compat.requestDevice.bind(compat);
    compat.requestDevice = async (descriptor = {}) => {
      const required = descriptor.requiredLimits;
      const limits = {};
      for (const [name, value] of Object.entries(required ?? {})) {
        const max = compat.limits[name];
        if (value === undefined || typeof max !== 'number') {
          limits[name] = value;
        } else if (name.startsWith('min')) {
          // Alignment limits are minimums: the adapter value is the tightest allowed.
          limits[name] = Math.max(value, max);
        } else {
          limits[name] = Math.min(value, max);
        }
      }
      const device = await requestDevice({ ...descriptor, requiredLimits: limits });
      // WebGPU validation errors are asynchronous and do not throw from frame().
      // Surface them through the client op instead of accepting a black capture.
      device.addEventListener('uncapturederror', (event) => {
        if (gpuErrors.length === 0) gpuErrors.push(event.error.message);
        console.error(event.error.message);
      });
      return device;
    };
    console.info('requestAdapter: using the compatibility adapter (core was missing or a fallback)');
    return compat;
  };
}

function showError(error) {
  const pre = document.createElement('pre');
  const code = error && error.code ? error.code : 'page.start_failed';
  const message = error && error.message ? error.message : String(error);
  pre.textContent = `${code}: ${message}`;
  document.body.appendChild(pre);
}

// The ops are registered before the demo starts, so the client never finds them missing;
// they wait for the start to settle and reject with its error when it failed.
let opRunning = false;
const demoReady = main();

registerNmcGpu({
  run: async (args) => {
    const demo = await demoReady;
    if (opRunning) throw new Error('run: an op run is already in progress');
    opRunning = true;
    try {
      const result = await runMeasured(demo, canvas, args);
      if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
      return result;
    } finally {
      opRunning = false;
    }
  },
  // Draw one frame and read the canvas in the same task, before it is presented.
  screenshot: async () => {
    const demo = await demoReady;
    demo.frame(performance.now());
    return canvas.toDataURL('image/png').split(',')[1];
  },
});

async function main() {
  let demo;
  try {
    installHardwareAdapterWrapper();
    await init();
    demo = await start(canvas, {});
  } catch (error) {
    console.error(error);
    showError(error);
    throw error;
  }

  // The idle loop pauses while an op run drives frames itself, so no frame is doubled.
  let loggedFrameError = false;
  const loop = (now) => {
    if (!opRunning) {
      try {
        demo.frame(now);
      } catch (error) {
        if (!loggedFrameError) console.error(error);
        loggedFrameError = true;
      }
    }
    nextFrame().then(loop);
  };
  nextFrame().then(loop);
  return demo;
}
